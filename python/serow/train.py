import numpy as np
import gymnasium as gym
import matplotlib

matplotlib.use("TkAgg")  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import json
import os
import torch
import torch.nn as nn
import pandas as pd

from env import SerowEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList
from utils import export_model_to_onnx


def compute_rolling_average(data, window_size):
    """Helper to compute rolling average, padding the start."""
    if data.size == 0:
        return []
    series = pd.Series(data)
    # Use .rolling().mean() with min_periods to start from the first data point
    rolling_avg = series.rolling(window=window_size, min_periods=1).mean()
    return rolling_avg.tolist()


def linear_schedule(initial_value, final_value):
    """Linear learning rate schedule."""

    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


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
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))
        truncated = self.locals.get("truncated", [False] * len(infos))

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
                reward_str = f"reward={self.episode_reward_sum:.3f}"
                length_str = f"length={self.episode_length}"
                print(f"Episode completed: {reward_str}, {length_str}")

            # Reset for next episode
            self.episode_reward_sum = 0
            self.episode_length = 0

        return True


class ExplainedVarianceCallback(BaseCallback):
    """Callback to monitor explained variance and warn if it's too negative."""

    def __init__(self, min_explained_var=-1.0, verbose=0):
        super(ExplainedVarianceCallback, self).__init__(verbose)
        self.min_explained_var = min_explained_var

    def _on_step(self) -> bool:
        # Check if we have explained variance info
        if hasattr(self.model, "logger") and self.model.logger.name_to_value:
            explained_var = self.model.logger.name_to_value.get(
                "train/explained_variance", 0
            )
            if explained_var < self.min_explained_var:
                if self.verbose > 0:
                    print(
                        f"Warning: Explained variance {explained_var:.3f} is very "
                        f"negative (below {self.min_explained_var:.3f}). "
                        f"Value function may be learning poorly."
                    )
        return True


class PerformanceDegradationCallback(BaseCallback):
    """Callback to detect performance degradation and stop training."""

    def __init__(self, window_size=10, degradation_threshold=0.1, verbose=0):
        super(PerformanceDegradationCallback, self).__init__(verbose)
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.recent_rewards = []

    def _on_step(self) -> bool:
        # This callback will be called by the training callback
        return True

    def on_episode_end(self, episode_reward):
        """Called when an episode ends with the episode reward."""
        self.recent_rewards.append(episode_reward)

        # Keep only recent rewards
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)

        # Check for degradation if we have enough data
        if len(self.recent_rewards) >= self.window_size:
            recent_avg = np.mean(self.recent_rewards[-self.window_size // 2 :])
            earlier_avg = np.mean(self.recent_rewards[: self.window_size // 2])

            if earlier_avg > 0 and recent_avg < earlier_avg * (
                1 - self.degradation_threshold
            ):
                if self.verbose > 0:
                    print(
                        f"Performance degradation detected: "
                        f"recent avg {recent_avg:.0f} vs earlier avg "
                        f"{earlier_avg:.0f}"
                    )
                return False

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


class ActorCriticONNX(nn.Module):
    def __init__(self, policy_model):
        super().__init__()
        self.policy = policy_model

    def forward(self, x):
        # Ensure we're in eval mode and detach gradients
        action, value, _ = self.policy.forward(x, deterministic=True)

        return action, value


if __name__ == "__main__":
    # Load and preprocess the data
    robot = "go2"
    n_envs = 4
    n_contacts = 2
    total_samples = 300000
    device = "cpu"
    history_size = 100
    datasets = []
    for i in range(n_envs):
        dataset = np.load(f"datasets/{robot}_log_{i}.npz", allow_pickle=True)
        datasets.append(dataset)

    test_dataset = np.load(f"{robot}_log.npz", allow_pickle=True)
    contact_states = test_dataset["contact_states"]
    contact_frame = list(contact_states[0].contacts_status.keys())
    print(f"Contact frames: {contact_frame}")

    state_dim = 3 + 9 + 3 + 4 + 3 * history_size + 1 * history_size
    print(f"State dimension: {state_dim}")
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()

    # Create vectorized environment
    def make_env(i, j):
        """Helper function to create a single environment with specific dataset"""
        ds = datasets[i]
        base_env = SerowEnv(
            contact_frame[
                np.random.randint(0, len(contact_frame))
            ],  # random choice of contact frame
            robot,
            ds["joint_states"][0],
            ds["base_states"][0],
            ds["contact_states"][0],
            action_dim,
            state_dim,
            ds["imu"],
            ds["joints"],
            ds["ft"],
            ds["base_pose_ground_truth"],
            history_size,
        )
        # Wrap with AutoPreStepWrapper to automatically use pre-step logic
        return AutoPreStepWrapper(base_env)

    test_env = SerowEnv(
        contact_frame[0],
        robot,
        test_dataset["joint_states"][0],
        test_dataset["base_states"][0],
        test_dataset["contact_states"][0],
        action_dim,
        state_dim,
        test_dataset["imu"],
        test_dataset["joints"],
        test_dataset["ft"],
        test_dataset["base_pose_ground_truth"],
        history_size,
    )

    # Create vectorized environment with different datasets for each environment
    env = DummyVecEnv(
        [
            lambda i=i, j=j: make_env(i, j)
            for i in range(n_envs)
            for j in range(n_contacts)
        ]
    )

    # Add normalization for observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    lr_schedule = linear_schedule(3e-4, 1e-5)
    model = PreStepPPO(
        "MlpPolicy",
        env,
        device=device,
        verbose=1,
        learning_rate=lr_schedule,
        n_steps=512,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.035,
        vf_coef=0.35,
        ent_coef=0.005,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[1024, 1024, 1024, 512, 256, 128],
                vf=[1024, 1024, 1024, 512, 256, 128],
            ),
            activation_fn=nn.ReLU,
            ortho_init=True,
        ),
    )

    # Create callbacks
    training_callback = TrainingCallback(verbose=1)
    valid_sample_callback = ValidSampleCallback(verbose=1)
    explained_var_callback = ExplainedVarianceCallback(
        min_explained_var=-1.0, verbose=1
    )
    performance_degradation_callback = PerformanceDegradationCallback(
        window_size=10, degradation_threshold=0.1, verbose=1
    )
    callback = CallbackList(
        [
            training_callback,
            valid_sample_callback,
            explained_var_callback,
            performance_degradation_callback,
        ]
    )

    # Train the model
    print(f"Training with {n_envs * len(contact_frame)} parallel environments")
    print("Starting training...")
    model.learn(total_timesteps=total_samples, callback=callback)
    print("Training completed")

    stats = None
    try:
        # Extract the observation normalization statistics
        obs_mean = env.obs_rms.mean
        obs_var = env.obs_rms.var
        obs_count = env.obs_rms.count
        stats = {
            "obs_mean": obs_mean,
            "obs_var": obs_var,
            "obs_count": obs_count,
        }
        print("Observation normalization stats:")
        print(f"  Mean: {stats['obs_mean']}")
        print(f"  Variance: {stats['obs_var']}")
        print(f"  Count: {stats['obs_count']}")

        # Convert numpy arrays to lists for JSON serialization
        json_stats = {
            "obs_mean": stats["obs_mean"].tolist(),
            "obs_var": stats["obs_var"].tolist(),
            "obs_count": int(stats["obs_count"]),
        }
        stats_file = f"models/{robot}_stats.json"
        with open(stats_file, "w") as f:
            json.dump(json_stats, f, indent=2)
    except Exception as e:
        print(f"Error saving stats: {e}")

    # Check if the models directory exists, if not create it
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save(f"models/{robot}_ppo")

    try:
        # Create a wrapper class to match the expected interface for export_model_to_onnx
        class PPOModelWrapper:
            def __init__(self, ppo_model, device):
                self.ppo_model = ppo_model
                self.device = device
                self.policy = ActorCriticONNX(ppo_model)
                self.name = "PPO"

        # Create the wrapper
        model_wrapper = PPOModelWrapper(model.policy, device)

        # Define parameters for ONNX export
        export_params = {
            "state_dim": state_dim,
            "action_dim": action_dim,
        }

        # Export to ONNX
        print("Exporting model to ONNX...")
        export_model_to_onnx(model_wrapper, robot, export_params, "models")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")

    # Debug information
    print("Training callback stats:")
    episode_count = len(training_callback.episode_rewards)
    print(f"Episode rewards collected: {episode_count}")
    print(f"Step rewards collected: {len(training_callback.step_rewards)}")
    valid_count = valid_sample_callback.valid_samples_count
    total_count = valid_sample_callback.total_samples_count
    valid_ratio = valid_count / total_count
    ratio_str = f"Valid sample ratio: {valid_count}/{total_count} ({valid_ratio:.2%})"
    print(ratio_str)

    # Plot the step rewards
    step_rewards = np.array(training_callback.step_rewards)
    # Normalize step rewards to 0-1
    step_rewards_norm = (step_rewards - np.min(step_rewards)) / (
        np.max(step_rewards) - np.min(step_rewards)
    ).tolist()
    step_rewards_avg = compute_rolling_average(step_rewards_norm, 100)
    plt.plot(step_rewards_avg, label="Average Rewards", alpha=1.0, color="blue")
    plt.plot(
        step_rewards_norm,
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

    test_env.evaluate(model, stats)
