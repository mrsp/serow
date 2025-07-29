import numpy as np
import gymnasium as gym
import pandas as pd

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import torch.nn as nn


def compute_rolling_average(data, window_size):
    """Helper to compute rolling average, padding the start."""
    if len(data) == 0:
        return []
    series = pd.Series(data)
    # Use .rolling().mean() with min_periods to start from the first data point
    rolling_avg = series.rolling(window=window_size, min_periods=1).mean()
    return rolling_avg.tolist()


class AutoPreStepWrapper(gym.Wrapper):
    """A wrapper that automatically handles pre-step logic."""

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_observation_for_action(self):
        """Delegate to the wrapped environment's get_observation_for_action method"""
        return self.env.get_observation_for_action()

    def _get_observation(self):
        """Override to automatically call get_observation_for_action"""
        return self.env.get_observation_for_action()

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ValidSampleCallback(BaseCallback):
    """Callback to track and handle valid/invalid samples during training."""

    def __init__(self, verbose=0):
        super(ValidSampleCallback, self).__init__(verbose)
        self.valid_samples_count = 0
        self.total_samples_count = 0
        self.invalid_samples_count = 0

    def _on_step(self) -> bool:
        # Count valid vs invalid samples
        infos = self.locals.get("infos", [])
        for info in infos:
            self.total_samples_count += 1
            if info.get("valid", True):
                self.valid_samples_count += 1
            else:
                self.invalid_samples_count += 1

        # Log statistics periodically
        if self.total_samples_count % 100 == 0:
            valid_ratio = (
                self.valid_samples_count / self.total_samples_count
                if self.total_samples_count > 0
                else 0.0
            )
            if self.verbose > 0:
                print(
                    f"Sample validity: {valid_ratio:.2%} valid "
                    f"({self.valid_samples_count}/{self.total_samples_count})"
                )

        return True


class PreStepPPO(PPO):
    """Custom PPO model that handles pre-step logic during training and evaluation."""

    def predict(self, observation, state=None, deterministic=False):
        return self.policy.predict(observation, state, deterministic)

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ):
        """Override collect_rollouts to use get_observation_for_action during training"""
        # Store original step method to restore later
        original_step_methods = []

        # Create wrapper environments that use get_observation_for_action
        if hasattr(env, "envs"):
            # Vectorized environment
            for i, single_env in enumerate(env.envs):
                # Store original step method
                original_step_methods.append(single_env.step)

                # Create a wrapper that intercepts step calls
                def make_step_wrapper(env, original_step):
                    def step_wrapper(action):
                        # Call get_observation_for_action before the step
                        env.get_observation_for_action()
                        return original_step(action)

                    return step_wrapper

                single_env.step = make_step_wrapper(
                    single_env, original_step_methods[-1]
                )
        else:
            # Single environment
            original_step_methods.append(env.step)

            def step_wrapper(action):
                # Call get_observation_for_action before the step
                env.get_observation_for_action()
                return original_step_methods[0](action)

            env.step = step_wrapper

        try:
            # Call the parent collect_rollouts method
            result = super().collect_rollouts(
                env, callback, rollout_buffer, n_rollout_steps
            )

        finally:
            # Restore original step methods
            if hasattr(env, "envs"):
                for i, single_env in enumerate(env.envs):
                    if i < len(original_step_methods):
                        single_env.step = original_step_methods[i]
            else:
                if len(original_step_methods) > 0:
                    env.step = original_step_methods[0]

        return result

    def forward(self, obs, deterministic=False):
        return self.policy.forward(obs, deterministic)


class KalmanFilterEnv(gym.Env):
    """Custom Gym environment integrating a Kalman filter."""

    def __init__(
        self,
        measurement,
        u,
        gt,
        min_action,
        max_action,
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

        # Initialize Kalman filter matrices
        self.F = np.array([[1, self.dt], [0, 1]])  # State transition matrix
        self.H = np.array([[1, 0]])  # Measurement matrix

        self.Q = (
            np.array(
                [
                    [0.25 * self.dt**4, 0.5 * self.dt**3],  # Process noise
                    [0.5 * self.dt**3, self.dt**2],
                ]
            )
            * self.process_noise**2
        )

        self.R = np.array([[self.measurement_noise**2]])  # Measurement noise

        # Action space
        self.action_space = spaces.Box(
            low=min_action, high=max_action, shape=(1,), dtype=np.float32
        )

        # Observation space: [position, velocity]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset Kalman filter state
        self.x = np.array([0.0, 0.0])  # Initial state [position, velocity]
        self.P = np.eye(self.state_dim) * 1.0  # Initial covariance
        self.step_count = 0
        self.reward = 0

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

        # Clipped position error to prevent explosion
        position_error = np.clip(position_error, 0, 2.5)

        # Simple quadratic reward (more stable than exponential)
        position_reward = 1.0 / (1.0 + position_error**2)

        # Innovation consistency reward (clipped)
        nis = float(y @ np.linalg.inv(S) @ y.T)
        nis = np.clip(nis, 0, 10.0)  # Prevent explosion
        innovation_reward = 1.0 / (1.0 + 0.1 * nis)

        # Small action penalty to encourage smoothness
        action_penalty = 0.01 * action[0] ** 2

        # Total reward (bounded)
        self.reward = position_reward + 0.1 * innovation_reward - action_penalty
        self.reward = np.clip(self.reward, -2.0, 3.0)  # Bound reward

        self.step_count += 1

        # Check termination conditions
        terminated = position_error > 2.5
        truncated = self.step_count == self.max_steps - 1
        # Get the final observation for the next step
        obs = self._get_observation()

        info = {
            "nis": nis,
            "position_error": position_error,
            "step_count": self.step_count,
            "reward": self.reward,  # Store original reward in info
            "valid": True if np.random.rand() > 0.2 else False,
        }
        return obs, self.reward, bool(terminated), bool(truncated), info

    def predict(self, control_input):
        """Kalman filter prediction step"""
        # Control input matrix (acceleration affects position and velocity)
        B = np.array([[0.5 * self.dt**2], [self.dt]])
        u = np.array([control_input])

        # Predict state
        self.x = self.F @ self.x + B.flatten() * u

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement, action):
        """Kalman filter update step"""
        # Update step
        R = self.R * float(action[0])

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

        return self.x, y, S

    def _get_observation(self):
        """Get current observation for the agent"""
        obs = np.array(
            [
                self.x[0],  # current position
                self.x[1],  # current velocity
                self.P[0, 0],  # position covariance
                self.P[1, 1],  # velocity covariance
                self.R[0, 0],  # measurement noise
                self.measurement[self.step_count] - self.x[0],  # innovation
            ],
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

        # Only accumulate episode rewards for valid steps
        valid_steps = 0
        total_reward = 0.0
        episode_completed = False
        if len(self.locals["infos"]) > 0:
            for i, info in enumerate(self.locals["infos"]):
                if info["valid"]:
                    reward = info["reward"]
                    valid_steps += 1
                    total_reward += reward
                    self.step_rewards.append(reward)
        # Only add to episode if there were valid steps
        if valid_steps > 0:
            avg_valid_reward = total_reward / valid_steps
            self.episode_reward_sum += avg_valid_reward
            self.episode_length += 1

        # Check if any episode completed (either done or truncated)
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

    # Add measurement noise (zero-mean Gaussian)
    measurement_noise = np.random.normal(0, measurement_noise_std, n_points)
    measurement = ground_truth + measurement_noise

    # Add control noise (zero-mean Gaussian)
    control_noise = np.random.normal(0, control_noise_std, n_points)
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
    n_envs = 8  # You can adjust this based on your CPU cores
    min_action = 1e-8
    max_action = 1e2

    # Generate random datasets
    datasets = []
    for i in range(n_envs + 1):
        measurement_signal, acceleration_signal, position_signal = generate_dataset(
            n_points=500,
            t_max=1.0,
            measurement_noise_std=0.1,
            control_noise_std=0.25,
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
            min_action=min_action,
            max_action=max_action,
        )
        # Wrap with AutoPreStepWrapper to automatically use pre-step logic
        return AutoPreStepWrapper(base_env)

    # Create vectorized environment with different datasets for each environment
    env = DummyVecEnv([lambda i=i: make_env(i) for i in range(n_envs)])

    # For testing, create a single environment using the last dataset
    test_env = AutoPreStepWrapper(
        KalmanFilterEnv(
            measurement=datasets[-1]["measurement"],
            u=datasets[-1]["control"],
            gt=datasets[-1]["ground_truth"],
            min_action=min_action,
            max_action=max_action,
        )
    )
    baseline_env = KalmanFilterEnv(
        measurement=datasets[-1]["measurement"],
        u=datasets[-1]["control"],
        gt=datasets[-1]["ground_truth"],
        min_action=min_action,
        max_action=max_action,
    )

    # Check environment
    check_env(test_env)
    print("Environment check passed!")
    print(f"Training with {n_envs} parallel environments")

    model = PreStepPPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        learning_rate=1e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64, 32], vf=[64, 64, 32]),
            activation_fn=nn.ReLU,
        ),
    )

    # Create callbacks
    training_callback = TrainingCallback(verbose=1)
    valid_sample_callback = ValidSampleCallback(verbose=0)

    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList

    callback = CallbackList([training_callback, valid_sample_callback])

    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=300000, callback=callback)

    # Save the model
    model.save("kalman_ppo_model")
    print("Model saved!")

    # Test the trained model
    print("\nTesting trained model...")
    obs, _ = test_env.reset()
    episode_rewards = []
    positions = []
    positions_baseline = []

    for step in range(len(test_env.env.gt)):
        # The model will automatically use get_observation_for_action
        obs = test_env.get_observation_for_action()
        action, _ = model.predict(obs, deterministic=True)
        print(f"step {step} action: {action}")
        obs, reward, terminated, truncated, info = test_env.step(action)
        baseline_env.step(np.array([1.0]))
        if len(episode_rewards) == 0:
            episode_rewards.append(reward)
        else:
            episode_rewards.append(episode_rewards[-1] + reward)
        positions.append(test_env.env.x[0])
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
    plt.plot(test_env.env.gt, color="r", linestyle="--", label="gt")
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
    valid_count = valid_sample_callback.valid_samples_count
    total_count = valid_sample_callback.total_samples_count
    valid_ratio = valid_count / total_count
    print(f"Valid sample ratio: {valid_count}/{total_count} ({valid_ratio:.2%})")

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
