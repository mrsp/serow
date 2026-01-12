import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import RolloutReturn


def linear_schedule(initial_value, final_value):
    """Linear learning rate schedule."""

    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


def compute_rolling_average(data, window_size):
    """Helper to compute rolling average, padding the start."""
    if len(data) == 0:
        return []
    series = pd.Series(data)
    # Use .rolling().mean() with min_periods to start from the first data point
    rolling_avg = series.rolling(window=window_size, min_periods=1).mean()
    return rolling_avg.tolist()


class PreStepDQN(DQN):
    def collect_rollouts(
        self,
        env,
        callback,
        train_freq,
        replay_buffer: ReplayBuffer,
        action_noise=None,
        learning_starts: int = 0,
        log_interval=None,
    ) -> RolloutReturn:
        """
        Custom rollout collection:
        - Always get observation from env.get_observation_for_action()
        - Only store transitions where info['valid'] == True
        """
        # Switch to eval mode to avoid dropout/batchnorm training
        self.policy.set_training_mode(False)
        n_steps = 0
        total_rewards = []
        completed_episodes = 0

        # Reset buffer for new rollout
        callback.on_rollout_start()

        while n_steps < train_freq[0]:
            # 1. Get obs for action selection
            obs_for_action = []
            if hasattr(env, "envs"):
                # Vectorized env
                for e in env.envs:
                    obs_for_action.append(e.get_observation_for_action())
                obs_for_action = np.array(obs_for_action)
            else:
                obs_for_action = np.array([env.get_observation_for_action()])

            self._last_obs = obs_for_action
            # 2. Predict action
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )

            # 3. Step environment
            new_obs, rewards, dones, infos = env.step(actions)

            # 4. Nullify invalid samples
            for idx, info in enumerate(infos):
                if not info.get("valid", True):
                    rewards[idx] = np.nan
                    new_obs[idx] = np.zeros_like(new_obs[idx])
                    self._last_obs[idx] = np.zeros_like(self._last_obs[idx])
                    buffer_actions[idx] = np.zeros_like(buffer_actions[idx])

            replay_buffer.add(
                self._last_obs,
                new_obs,
                buffer_actions,
                rewards,
                dones,
                infos,
            )
            self._update_info_buffer(infos, dones)

            # 5. Update counters
            n_steps += 1
            self.num_timesteps += env.num_envs
            total_rewards.extend(rewards)

            # Count completed episodes
            completed_episodes += sum(dones)

            # 6. Handle episode ends
            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(
                    episode_timesteps=n_steps,
                    n_episodes=completed_episodes,
                    continue_training=False,
                )

        callback.on_rollout_end()

        return RolloutReturn(
            episode_timesteps=n_steps,
            n_episodes=completed_episodes,
            continue_training=True,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # Filter out invalid samples
            valid_mask = ~torch.isnan(replay_data.rewards.flatten())
            num_valid = valid_mask.sum()

            # Skip if too few valid samples (less than 25% of batch)
            min_valid_samples = max(1, batch_size // 4)
            if num_valid < min_valid_samples:
                self.logger.record("train/skipped_batches", 1, exclude="tensorboard")
                continue

            # Create filtered data instead of modifying the original object
            filtered_observations = replay_data.observations[valid_mask]
            filtered_next_observations = replay_data.next_observations[valid_mask]
            filtered_actions = replay_data.actions[valid_mask]
            filtered_rewards = replay_data.rewards[valid_mask]
            filtered_dones = replay_data.dones[valid_mask]
            filtered_discounts = (
                replay_data.discounts[valid_mask]
                if replay_data.discounts is not None
                else None
            )

            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = (
                filtered_discounts if filtered_discounts is not None else self.gamma
            )

            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(filtered_next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = (
                    filtered_rewards + (1 - filtered_dones) * discounts * next_q_values
                )

            # Get current Q-values estimates
            current_q_values = self.q_net(filtered_observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = torch.gather(
                current_q_values, dim=1, index=filtered_actions.long()
            )

            # Compute Huber loss (less sensitive to outliers)
            loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.dump(step=self.num_timesteps)

    def forward(self, obs, deterministic=False):
        return self.policy.forward(obs, deterministic)


class KalmanFilterEnv(gym.Env):
    """Custom Gym environment integrating a Kalman filter."""

    def __init__(
        self,
        measurement,
        u,
        gt,
        process_noise=0.1,
        measurement_noise=0.1,
    ):
        super(KalmanFilterEnv, self).__init__()

        # Environment parameters
        self.measurement = measurement
        self.gt = gt
        self.u = u
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_steps = len(measurement) - 1

        # Kalman filter parameters
        self.dt = 0.001  # time step

        # State: [position, velocity]
        self.state_dim = 2
        self.measurement_dim = 1  # we only measure position
        self.history_length = 100
        self.measurement_history = []
        self.measurement_noise_history = []
        self.prev_action = 0.0

        # Initialize Kalman filter matrices
        self.F = np.array([[1, self.dt], [0, 1]])  # State transition matrix
        self.H = np.array([[1, 0]])  # Measurement matrix
        self.B = np.array([0.0, self.dt])

        self.Q = np.array([[self.process_noise**2]])
        self.R = np.array([[self.measurement_noise**2]])  # Measurement noise

        # Action space - discrete choices for measurement noise scaling
        self.discrete_actions = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
        self.action_space = spaces.Discrete(len(self.discrete_actions))

        # Observation space: [position, velocity, position covariance,
        # velocity covariance, measurement noise, innovation]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5 + self.history_length * 2,),
            dtype=np.float32,
        )

        self.reset()

    def get_available_actions(self):
        """Return the available discrete action values."""
        return self.discrete_actions.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset Kalman filter state
        self.x = np.array([0.0, 0.0])  # Initial state [position, velocity]
        self.P = np.eye(self.state_dim) * np.random.uniform(
            0.1, 5.0
        )  # Initial covariance
        self.step_count = 0
        self.reward = 0

        self.measurement_history = [0.0] * self.history_length
        self.measurement_noise_history = [float(self.R[0, 0])] * self.history_length
        self.prev_action = 0.0

        # Get initial observation
        obs = self._get_observation()
        return obs, {}

    def get_observation_for_action(self):
        """Get the observation that should be used for action computation."""
        # Run prediction step with current control input
        if self.step_count < len(self.u):
            self.predict(self.u[self.step_count])

        # Get the observation that the policy should use
        obs = self._get_observation()
        return obs

    def step(self, action):
        next_state, y, S = self.update(self.measurement[self.step_count], action)

        position_error = abs(next_state[0] - self.gt[self.step_count])
        position_reward = -position_error / 10.0

        # Innovation consistency reward (clipped)
        nis = float(y @ np.linalg.inv(S) @ y.T)
        nis = np.clip(nis, 0, 10.0)
        innovation_reward = -nis / 10.0

        # Small action penalty to encourage smoothness
        current_action_value = self.discrete_actions[action]
        action_penalty = abs(current_action_value - self.prev_action)

        self.reward = (
            4.0 * position_reward + 0.5 * innovation_reward - 0.005 * action_penalty
        )

        # Check termination conditions
        terminated = position_error > 10.0 or self.step_count == self.max_steps - 1
        truncated = False

        info = {
            "nis": nis,
            "position_error": position_error,
            "step_count": self.step_count,
            "reward": self.reward,
            "valid": (
                True if np.random.rand() > 0.5 else False
            ),  # Make 50% of samples invalid
        }

        self.step_count += 1
        self.prev_action = current_action_value

        # Get the final observation for the next step
        obs = self._get_observation()
        return obs, self.reward, bool(terminated), bool(truncated), info

    def predict(self, control_input):
        """Kalman filter prediction step"""
        # Control input matrix (acceleration affects position and velocity)
        u = np.array([control_input])

        # Predict state
        self.x = self.F @ self.x + self.B * u

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement, action):
        """Kalman filter update step"""
        # Update step - convert discrete action index to actual scaling value
        action_scaling = self.discrete_actions[action]
        R = self.R * action_scaling

        # Innovation
        y = measurement - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state estimate
        self.x += K @ y

        # Updated covariance
        self.P = self.P - K @ self.H @ self.P

        self.measurement_history.append(measurement)
        self.measurement_noise_history.append(self.R[0, 0])
        while len(self.measurement_history) > self.history_length:
            self.measurement_history.pop(0)
            self.measurement_noise_history.pop(0)

        return self.x, y, S

    def _get_observation(self):
        """Get current observation for the agent"""

        measurement_history = np.array(self.measurement_history).flatten()
        measurement_noise_history = np.array(self.measurement_noise_history).flatten()
        obs = np.concatenate(
            [
                [self.x[1]],  # current velocity
                [self.P[0, 0]],  # position covariance
                [self.P[1, 1]],  # velocity covariance
                [self.R[0, 0]],  # measurement noise
                measurement_history,
                measurement_noise_history,
                [self.measurement[self.step_count] - self.x[0]],  # innovation
            ],
            axis=0,
            dtype=np.float32,
        )

        return obs

    def render(self, mode="human"):
        if mode == "human":
            print(
                f"Step: {self.step_count}, Position: {self.x[0]},  "
                f"Target: {self.gt[self.step_count]}, Reward: {self.reward}"
            )


class TrainingCallback(BaseCallback):
    """Custom callback for monitoring training progress"""

    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.step_rewards = []  # Track rewards per step as fallback
        self.episode_reward_sum = 0
        self.episode_length = 0
        self.last_dones = None
        self.total_steps = 0

    def _on_step(self) -> bool:
        self.total_steps += 1

        # Get dones and truncated to detect episode completions
        dones = self.locals.get("dones", [False] * len(self.locals.get("infos", [])))
        truncated = self.locals.get(
            "truncated", [False] * len(self.locals.get("infos", []))
        )

        # Since invalid samples are now filtered out, all samples are valid
        if len(self.locals["infos"]) > 0:
            for i, info in enumerate(self.locals["infos"]):
                reward = info["reward"]
                self.step_rewards.append(reward)
                self.episode_reward_sum += reward
                self.episode_length += 1

        # Check if any episode completed (either done or truncated)
        episode_completed = False
        for _, (done, trunc) in enumerate(zip(dones, truncated)):
            if done or trunc:
                episode_completed = True
                break

        if episode_completed and self.episode_length > 0:
            self.episode_rewards.append(self.episode_reward_sum)
            self.episode_lengths.append(self.episode_length)
            if self.verbose > 0:
                print(
                    f"Episode completed: reward={self.episode_reward_sum:.3f}, "
                    f"length={self.episode_length}"
                )

            # Reset for next episode
            self.episode_reward_sum = 0
            self.episode_length = 0

        return True


def generate_dataset(
    n_points=1000,
    t_max=1.0,
    measurement_noise_std=0.1,
    control_noise_std=0.05,
    seed=None,
):
    """Generate a random dataset for Kalman filter training."""
    if seed is not None:
        np.random.seed(seed)

    # Time vector
    t = np.linspace(0, t_max, n_points)

    # Generate random parameters for different trajectories
    # This ensures each dataset has a different ground truth signal
    freq1 = np.random.uniform(1.0, 3.0)  # Random frequency for primary
    freq2 = np.random.uniform(2.0, 6.0)  # Random frequency for secondary
    freq3 = np.random.uniform(3.0, 8.0)  # Random frequency for tertiary

    amp1 = np.random.uniform(1.0, 3.0)  # Random amplitude for primary
    amp2 = np.random.uniform(0.2, 1.0)  # Random amplitude for secondary
    amp3 = np.random.uniform(0.1, 0.5)  # Random amplitude for tertiary

    quad_coeff = np.random.uniform(0.1, 0.8)  # Random quadratic coefficient
    phase1 = np.random.uniform(0, 2 * np.pi)  # Random phase shifts
    phase2 = np.random.uniform(0, 2 * np.pi)
    phase3 = np.random.uniform(0, 2 * np.pi)

    # Generate a smooth ground truth signal using random parameters
    ground_truth = (
        amp1 * np.sin(2 * np.pi * freq1 * t + phase1)  # Primary oscillation
        + amp2 * np.sin(2 * np.pi * freq2 * t + phase2)  # Secondary
        + quad_coeff * t**2  # Quadratic trend
        + amp3 * np.cos(2 * np.pi * freq3 * t + phase3)  # Additional complexity
    )

    # Compute the second derivative (acceleration) analytically
    true_acceleration = (
        -4
        * np.pi**2
        * freq1**2
        * amp1
        * np.sin(2 * np.pi * freq1 * t + phase1)  # Second derivative of primary
        - -4
        * np.pi**2
        * freq2**2
        * amp2
        * np.sin(2 * np.pi * freq2 * t + phase2)  # Second derivative of secondary
        + 2 * quad_coeff  # Second derivative of quadratic term
        + -4
        * np.pi**2
        * freq3**2
        * amp3
        * np.cos(2 * np.pi * freq3 * t + phase3)  # Second derivative of cosine
    )

    # Add varying measurement noise (zero-mean Gaussian with varying std)
    time_varying_std = measurement_noise_std * (1 + 0.5 * t / t_max)
    measurement_noise = np.random.normal(0, time_varying_std, n_points)
    measurement = ground_truth + measurement_noise

    # Add varying control noise (zero-mean Gaussian with varying std)
    control_varying_std = control_noise_std * (
        1 + 0.2 * np.abs(true_acceleration) / np.max(np.abs(true_acceleration))
    )

    control_noise = np.random.normal(0, control_varying_std, n_points)
    control = true_acceleration + control_noise

    return measurement, control, ground_truth


def visualize_dataset(measurement, control, ground_truth, save_plot=False):
    """Visualize the generated dataset."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot position signals
    t = np.linspace(0, 1, len(ground_truth))
    axes[0].plot(t, ground_truth, "b-", linewidth=2, label="Ground Truth")
    axes[0].plot(t, measurement, "r.", markersize=1, alpha=0.6, label="Measurement")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Position")
    axes[0].set_title("Position Signal")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot control signals
    axes[1].plot(t, control, "g-", linewidth=1, label="Control (Acceleration)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Acceleration")
    axes[1].set_title("Control Signal")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        plt.savefig("generated_dataset.png", dpi=300, bbox_inches="tight")

    plt.show()


def main():
    # Number of parallel environments
    n_envs = 8
    measurement_noise_std = 0.1
    control_noise_std = 0.25
    scale = 1e3
    gen_scale = 1e-3

    # Generate random datasets
    datasets = []
    for i in range(n_envs + 1):
        measurement_signal, acceleration_signal, position_signal = generate_dataset(
            n_points=1000,
            t_max=1.0,
            measurement_noise_std=gen_scale * measurement_noise_std,
            control_noise_std=control_noise_std,
            seed=42 + i,  # Different seed for each dataset
        )
        dataset = {
            "control": acceleration_signal,
            "measurement": measurement_signal,
            "ground_truth": position_signal,
        }
        datasets.append(dataset)

        # Print some statistics to verify datasets are different
        print(
            f"Dataset {i}: GT range [{position_signal.min():.3f}, "
            f"{position_signal.max():.3f}], GT std: {position_signal.std():.3f}"
        )

        # Only visualize the first dataset to avoid too many plots
        # visualize_dataset(measurement_signal, acceleration_signal, position_signal)

    # Create vectorized environment
    def make_env(dataset_idx):
        """Helper function to create a single environment with specific dataset"""
        dataset = datasets[dataset_idx]
        base_env = KalmanFilterEnv(
            measurement=dataset["measurement"],
            u=dataset["control"],
            gt=dataset["ground_truth"],
            measurement_noise=scale * measurement_noise_std,
            process_noise=control_noise_std,
        )
        return base_env

    # Create vectorized environment with different datasets for each environment
    env = DummyVecEnv([lambda i=i: make_env(i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # For testing, create a single environment using the last dataset
    test_env = KalmanFilterEnv(
        measurement=datasets[-1]["measurement"],
        u=datasets[-1]["control"],
        gt=datasets[-1]["ground_truth"],
        measurement_noise=scale * measurement_noise_std,
        process_noise=control_noise_std,
    )

    baseline_env = KalmanFilterEnv(
        measurement=datasets[-1]["measurement"],
        u=datasets[-1]["control"],
        gt=datasets[-1]["ground_truth"],
        measurement_noise=scale * measurement_noise_std,
        process_noise=control_noise_std,
    )

    # Check environment
    check_env(test_env)
    print("Environment check passed!")
    print(f"Training with {n_envs} parallel environments")

    model = PreStepDQN(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        learning_rate=linear_schedule(3e-4, 1e-5),
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=(4, "step"),
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[512, 512, 256, 128],
            activation_fn=nn.ReLU,
        ),
    )

    # Train the model
    print("Starting DQN training...")
    training_callback = TrainingCallback()
    model.learn(total_timesteps=100000, callback=training_callback)

    stats = None
    try:
        # Extract the observation normalization statistics
        stats = {
            "obs_mean": env.obs_rms.mean,
            "obs_var": env.obs_rms.var,
            "obs_count": env.obs_rms.count,
        }
        print("Observation normalization stats:")
        print(f"  Mean: {stats['obs_mean']}")
        print(f"  Variance: {stats['obs_var']}")
        print(f"  Count: {stats['obs_count']}")
    except Exception as e:
        print(f"Error extracting observation normalization stats: {e}")

    # Save the model
    # model.save("kalman_dqn_model")
    # print("Model saved!")

    # Test the trained model
    print("\nTesting trained model...")
    obs, _ = test_env.reset()
    baseline_env.reset()

    episode_rewards = []
    positions = []
    positions_baseline = []

    for step in range(len(test_env.gt)):
        # Call get_observation_for_action manually for testing
        obs = test_env.get_observation_for_action()
        if stats is not None:
            obs = (obs - stats["obs_mean"]) / np.sqrt(stats["obs_var"])
        action, _ = model.predict(obs, deterministic=True)
        print(f"step {step} action: {test_env.discrete_actions[action]}")
        obs, reward, terminated, truncated, _ = test_env.step(action)

        # Run the baseline
        baseline_env.get_observation_for_action()
        baseline_env.step(np.where(test_env.discrete_actions == 1.0)[0][0])
        if len(episode_rewards) == 0:
            episode_rewards.append(reward)
        else:
            episode_rewards.append(episode_rewards[-1] + reward)
        positions.append(test_env.x[0])
        positions_baseline.append(baseline_env.x[0])
        if step % 20 == 0:
            test_env.render()

        if terminated or truncated:
            break

    print(f"\nTest episode reward: {episode_rewards[-1]:.2f}")

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(positions, color="b", label="agent")
    plt.plot(test_env.gt, color="r", linestyle="--", label="gt")
    plt.plot(positions_baseline, color="g", linestyle="--", label="baseline")
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.title("Agent Position Over Time")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)

    # Debug information
    print("Training callback stats:")
    print(f"Episode rewards collected: {len(training_callback.episode_rewards)}")
    print(f"Step rewards collected: {len(training_callback.step_rewards)}")

    # Plot the step rewards
    step_rewards = training_callback.step_rewards
    step_rewards_avg = compute_rolling_average(step_rewards, 100)
    plt.plot(step_rewards_avg, label="Average Rewards", alpha=1.0, color="blue")
    plt.plot(
        step_rewards,
        label="Rewards",
        alpha=0.35,
        color="lightblue",
    )
    plt.xlabel("Samples")
    plt.ylabel("Normalized Rewards")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
