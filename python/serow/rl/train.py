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
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import RolloutReturn


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
        - Only use valid transitions where info['valid'] == True
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

            # 4. Nullify invalid samples - Set rewards to nan so we can filter them out in train()
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


if __name__ == "__main__":
    # Load and preprocess the data
    robot = "go2"
    n_envs = 4
    n_contacts = 3
    total_samples = 100000
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
        return base_env

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

    lr_schedule = linear_schedule(5e-4, 1e-5)
    model = PreStepDQN(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        learning_rate=lr_schedule,
        buffer_size=1000000,
        learning_starts=5000,
        batch_size=128,
        gamma=0.99,
        train_freq=(8, "step"),
        gradient_steps=4,
        target_update_interval=2000,
        exploration_fraction=0.2,
        exploration_initial_eps=0.9,
        exploration_final_eps=0.02,
        policy_kwargs=dict(
            net_arch=[1024, 1024, 512, 256, 128],
            activation_fn=nn.ReLU,
        ),
        max_grad_norm=10.0,
        tau=0.005,
    )

    # Create callbacks
    training_callback = TrainingCallback(verbose=1)
    valid_sample_callback = ValidSampleCallback(verbose=1)
    callback = CallbackList(
        [
            training_callback,
            valid_sample_callback,
        ]
    )

    # Train the model
    print(f"Training with {n_envs * n_contacts} parallel environments")
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
    model.save(f"models/{robot}_dqn")

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
