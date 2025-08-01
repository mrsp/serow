import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


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


# Inverted Pendulum Environment
class InvertedPendulum(gym.Env):
    def __init__(
        self,
        min_action,
        max_action,
    ):
        super().__init__()

        self.g = 9.81
        self.length = 1.0
        self.m = 1.0
        self.max_angular_vel = 8.0
        self.dt = 0.05
        self.upright_steps = 0  # Track consecutive upright steps
        self.max_steps = 1000
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=min_action, high=max_action, shape=(1,), dtype=np.float32
        )

        self.reset()

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start with a small random angle near upright
        theta = np.random.uniform(-np.pi / 6, np.pi / 6)
        theta_dot = np.random.uniform(-0.1, 0.1)

        # Convert to [cos(theta), sin(theta), theta_dot] representation
        self.state = np.array(
            [np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32
        ).flatten()
        self.upright_steps = 0
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        # Recover theta from cos(theta) and sin(theta)
        cos_theta, sin_theta, theta_dot = self.state
        theta = np.arctan2(sin_theta, cos_theta)

        # Calculate next state
        theta_ddot = (self.g / self.length) * np.sin(theta) + (
            action / (self.m * self.length**2)
        )
        theta_dot = theta_dot + theta_ddot * self.dt
        theta = theta + theta_dot * self.dt
        self.state = np.array(
            [np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32
        ).flatten()

        # Normalize angle to [-π, π]
        theta = self.angle_normalize(theta)

        # Primary reward: exponential reward for being upright
        angle_reward = 5.0 * np.exp(-(theta**2))

        # Velocity penalty: encourage low angular velocity
        velocity_penalty = 0.01 * theta_dot**2

        # Control penalty: encourage efficient control
        control_penalty = 0.001 * action**2

        # Bonus for sustained balancing
        if abs(theta) < 0.1 and abs(theta_dot) < 0.5:
            sustained_bonus = 1.0
        else:
            sustained_bonus = 0.0

        reward = angle_reward.item()
        reward -= velocity_penalty.item()
        reward -= control_penalty.item()
        reward += sustained_bonus

        # Termination condition - only terminate for extreme angular velocities
        done = False
        truncated = False
        if abs(theta_dot) > self.max_angular_vel:
            done = True
            reward = -5.0

        if self.step_count >= self.max_steps:
            done = False
            truncated = True

        self.step_count += 1
        self.reward = reward

        info = {
            "valid": True,
            "reward": reward,
        }

        return self._get_observation(), reward, done, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            print(
                f"Step: {self.step_count}, "
                f"Position: {np.arctan2(self.state[1], self.state[0]) * 180.0 / np.pi},  "
                f"Target: {0.0}, Reward: {self.reward}"
            )

    def _get_observation(self):
        return self.state.flatten()

    def get_observation_for_action(self):
        """Get the observation that should be used for action computation."""
        # Get the observation that the policy should use
        obs = self._get_observation()
        return obs


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
                    f"Episode completed: reward={self.episode_reward_sum}, "
                    f"length={self.episode_length}"
                )

            # Reset for next episode
            self.episode_reward_sum = 0
            self.episode_length = 0

        return True


def main():
    # Number of parallel environments
    n_envs = 8  # You can adjust this based on your CPU cores
    min_action = -1.0
    max_action = 1.0

    # Create vectorized environment
    def make_env():
        """Helper function to create a single environment with specific dataset"""
        base_env = InvertedPendulum(
            min_action=min_action,
            max_action=max_action,
        )
        # Wrap with AutoPreStepWrapper to automatically use pre-step logic
        return AutoPreStepWrapper(base_env)

    # Create vectorized environment with different datasets for each environment
    env = DummyVecEnv([lambda i=i: make_env() for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # For testing, create a single environment using the last dataset
    test_env = AutoPreStepWrapper(
        InvertedPendulum(
            min_action=min_action,
            max_action=max_action,
        )
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
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.035,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh,
            ortho_init=True,
        ),
    )

    # Create callbacks
    training_callback = TrainingCallback(verbose=1)
    valid_sample_callback = ValidSampleCallback(verbose=0)

    # Combine callbacks
    callback = CallbackList([training_callback, valid_sample_callback])

    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=300000, callback=callback)

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
    model.save("inverted_pendulum_ppo_model")
    print("Model saved!")

    # Test the trained model
    print("\nTesting trained model...")
    test_env.reset()
    episode_rewards = []
    positions = []
    velocities = []
    actions = []
    for step in range(test_env.env.max_steps):
        # The model will automatically use get_observation_for_action
        obs = test_env.get_observation_for_action()
        if stats is not None:
            obs = (obs - stats["obs_mean"]) / np.sqrt(stats["obs_var"])
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        if len(episode_rewards) == 0:
            episode_rewards.append(reward)
        else:
            episode_rewards.append(episode_rewards[-1] + reward)
        positions.append(np.arctan2(obs[1], obs[0]) * 180.0 / np.pi)
        velocities.append(obs[2] * 180.0 / np.pi)
        if step % 20 == 0:
            test_env.render()

        if terminated or truncated:
            break

    print(f"\nTest episode reward: {episode_rewards[-1]:.2f}")

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.plot(positions, color="b", label="angle (deg)")
    plt.xlabel("Time Step")
    plt.ylabel("Angle")
    plt.title("Agent Angle Over Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(actions, color="r", label="torque (Nm)")
    plt.xlabel("Time Step")
    plt.ylabel("Action")
    plt.title("Agent Action Over Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(velocities, color="g", label="velocity (deg/s)")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.title("Agent Velocity Over Time")
    plt.legend()
    plt.grid(True)

    # Debug information
    print("Training callback stats:")
    print(f"Episode rewards collected: " f"{len(training_callback.episode_rewards)}")
    print(f"Step rewards collected: {len(training_callback.step_rewards)}")
    valid_count = valid_sample_callback.valid_samples_count
    total_count = valid_sample_callback.total_samples_count
    valid_ratio = valid_count / total_count
    print(f"Valid sample ratio: {valid_count}/{total_count} " f"({valid_ratio:.2%})")

    # Plot the step rewards
    step_rewards = training_callback.step_rewards
    step_rewards_avg = compute_rolling_average(step_rewards, 200)
    plt.subplot(1, 4, 4)
    plt.plot(step_rewards_avg, label="Average Rewards", alpha=1.0, color="blue")
    plt.plot(
        step_rewards,
        label="Rewards",
        alpha=0.35,
        color="lightblue",
    )
    plt.xlabel("Samples")
    plt.ylabel("Rewards")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
