from env import SerowEnv
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList
from ac import CustomActorCritic


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

    def _on_step(self) -> bool:
        # Get dones to detect episode completions
        dones = self.locals.get("dones", [False] * len(self.locals.get("infos", [])))

        # Track episode rewards manually
        if "rewards" in self.locals:
            rewards = self.locals["rewards"]
            avg_reward = np.mean(rewards)
            self.step_rewards.append(avg_reward)

            # Add to current episode
            self.episode_reward_sum += avg_reward
            self.episode_length += 1

            # Check if any episode completed
            for i, done in enumerate(dones):
                if done and self.episode_length > 0:
                    self.episode_rewards.append(self.episode_reward_sum)
                    self.episode_lengths.append(self.episode_length)

                    # Reset for next episode
                    self.episode_reward_sum = 0
                    self.episode_length = 0

        # Also try the original method
        if len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            if "episode" in info:
                # Use the framework's episode info if available
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                if self.verbose > 0:
                    print(
                        f"Framework episode: reward={info['episode']['r']:.3f}, "
                        f"length={info['episode']['l']}"
                    )

        return True


class PreStepPPO(PPO):
    """Custom PPO model that handles pre-step logic during training and evaluation."""

    def predict(self, observation, state=None, deterministic=False):
        return super().predict(observation, state, deterministic)

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
        """Override forward to use get_observation_for_action if available"""
        # This method is called by the policy during training
        # We need to ensure the environment has run its pre-step logic
        if hasattr(self.env, "envs"):
            # Vectorized environment - ensure each env has run pre-step
            for env in self.env.envs:
                # The observation should already be from
                # get_observation_for_action since we modified the step
                # method in collect_rollouts
                pass
        else:
            # Single environment
            # The observation should already be from
            # get_observation_for_action since we modified the step
            # method in collect_rollouts
            pass

        # Call the parent forward method
        return super().forward(obs, deterministic)


if __name__ == "__main__":
    # Load and preprocess the data
    robot = "go2"
    dataset = np.load(f"{robot}_log.npz", allow_pickle=True)
    datasets = []
    datasets.append(dataset)

    test_dataset = dataset
    contact_states = dataset["contact_states"]
    contact_frame = list(contact_states[0].contacts_status.keys())[0]

    state_dim = 2 + 3 + 9 + 3 + 4
    print(f"State dimension: {state_dim}")
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    min_action = np.array([1e-8], dtype=np.float32)
    max_action = np.array([1e2], dtype=np.float32)

    # Create vectorized environment
    def make_env(dataset_idx):
        """Helper function to create a single environment with specific dataset"""
        ds = datasets[dataset_idx]
        base_env = SerowEnv(
            robot,
            contact_frame,
            ds["joint_states"][0],
            ds["base_states"][0],
            ds["contact_states"][0],
            action_dim,
            state_dim,
            min_action,
            max_action,
            ds["imu"],
            ds["joints"],
            ds["ft"],
            ds["base_pose_ground_truth"],
        )
        # Wrap with AutoPreStepWrapper to automatically use pre-step logic
        return AutoPreStepWrapper(base_env)

    # Create vectorized environment with different datasets for each environment
    n_envs = 1
    env = DummyVecEnv([lambda i=i: make_env(i) for i in range(n_envs)])

    model = PreStepPPO(
        CustomActorCritic,
        env,
        device="cuda",
        verbose=1,
        learning_rate=5e-5,
        n_steps=512,
        batch_size=128,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(action_min=1e-8, action_max=1e2),
    )

    # Create callbacks
    training_callback = TrainingCallback(verbose=1)
    valid_sample_callback = ValidSampleCallback(verbose=1)
    callback = CallbackList([training_callback, valid_sample_callback])
    # Train the model
    print(f"Training with {n_envs} parallel environments")
    print("Starting training...")
    model.learn(total_timesteps=100000, callback=callback)

    test_env = make_env(0)
    test_env.env.evaluate(model)
