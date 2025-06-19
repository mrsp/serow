import gymnasium as gym
import torch as th
import numpy as np
import warnings
import serow

from torch import nn
from typing import Callable, Optional, List, Union, Dict, Type, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from read_mcap import(
    read_base_states, 
    read_contact_states, 
    read_force_torque_measurements, 
    read_joint_measurements, 
    read_imu_measurements, 
    read_base_pose_ground_truth,
    read_joint_states
)

from utils import(
    quaternion_to_rotation_matrix, 
    logMap
)

class InvalidSampleRemover(BaseCallback):
    """
    Callback that removes invalid samples from the PPO rollout buffer before training.
    This ensures that only valid data is used for policy updates.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.valid_indices = []  # Track valid sample indices
        self.current_step = 0
        self.filtered_count = 0
        self.total_count = 0
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        if self.verbose >= 1:
            print("InvalidSampleRemover callback initialized")
    
    def _on_step(self) -> bool:
        """Called after each environment step during rollout collection."""
        try:
            # Get the current step information
            infos = self.locals.get('infos', [])
            
            # Process each environment's info
            for env_idx, info in enumerate(infos):
                self.total_count += 1
                is_valid = info.get('valid_step', True)
                
                if is_valid:
                    # Store the global index of this valid sample
                    global_idx = self.current_step * len(infos) + env_idx
                    self.valid_indices.append(global_idx)
                else:
                    self.filtered_count += 1
                    if self.verbose >= 1:
                        reason = info.get('invalid_reason', 'unknown')
                        print(f"Invalid sample detected: step {self.current_step}, env {env_idx}, reason: {reason}")
            
            self.current_step += 1
            
            if self.verbose >= 2 and self.total_count % 100 == 0:
                print(f"Step tracking: total={self.total_count}, filtered={self.filtered_count}, valid={len(self.valid_indices)}")
            
        except Exception as e:
            if self.verbose >= 1:
                warnings.warn(f"Error in sample tracking: {e}")
            # On error, assume all samples in this step are valid to prevent buffer corruption
            infos = self.locals.get('infos', [])
            for env_idx in range(len(infos)):
                global_idx = self.current_step * len(infos) + env_idx
                self.valid_indices.append(global_idx)
            self.current_step += 1
        
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of rollout collection to filter the buffer."""
        try:
            if self.verbose >= 1:
                print(f"Rollout ended. Total samples: {self.total_count}, Valid samples: {len(self.valid_indices)}, Filtered: {self.filtered_count}")
            
            # Get the rollout buffer
            rollout_buffer = self.model.rollout_buffer
            
            # Check if we have valid samples
            if not self.valid_indices:
                if self.verbose >= 1:
                    print("No valid samples found, resetting buffer")
                rollout_buffer.reset()
                self._reset_tracking()
                return
            
            # Check minimum sample requirement
            min_valid_samples = max(32, self.model.batch_size)
            if len(self.valid_indices) < min_valid_samples:
                if self.verbose >= 1:
                    print(f"Not enough valid samples ({len(self.valid_indices)}) for training (minimum {min_valid_samples}), resetting buffer")
                rollout_buffer.reset()
                self._reset_tracking()
                return
            
            # Convert valid indices to boolean mask
            buffer_size = rollout_buffer.buffer_size
            if buffer_size != self.total_count:
                if self.verbose >= 1:
                    print(f"Buffer size mismatch: expected {self.total_count}, got {buffer_size}")
                    print("This might indicate a problem with step counting. Proceeding with available data.")
                # Adjust valid indices to match actual buffer size
                self.valid_indices = [idx for idx in self.valid_indices if idx < buffer_size]
            
            # Create boolean mask
            valid_mask = np.zeros(buffer_size, dtype=bool)
            valid_mask[self.valid_indices] = True
            
            if self.verbose >= 1:
                print(f"Filtering buffer: {buffer_size} -> {np.sum(valid_mask)} samples")
            
            # Filter all buffer components
            self._filter_buffer_data(rollout_buffer, valid_mask)
            
            # Update buffer size
            new_size = np.sum(valid_mask)
            rollout_buffer.buffer_size = new_size
            rollout_buffer.pos = new_size
            
            if hasattr(rollout_buffer, 'full'):
                rollout_buffer.full = True
            
            if self.verbose >= 1:
                print(f"Buffer filtering completed. New size: {rollout_buffer.buffer_size}")
                
        except Exception as e:
            if self.verbose >= 1:
                warnings.warn(f"Error in filtering rollout buffer: {e}")
                import traceback
                traceback.print_exc()
        finally:
            # Always reset tracking for next rollout
            self._reset_tracking()

    def _filter_buffer_data(self, rollout_buffer: RolloutBuffer, valid_mask: np.ndarray) -> None:
        """Filter all data arrays in the rollout buffer."""
        # List of attributes that contain data arrays
        data_attributes = [
            'observations', 'actions', 'rewards', 'episode_starts', 
            'values', 'log_probs', 'advantages', 'returns'
        ]
        
        for attr_name in data_attributes:
            if hasattr(rollout_buffer, attr_name):
                attr_value = getattr(rollout_buffer, attr_name)
                if attr_value is not None:
                    try:
                        # Handle both tensor and numpy array cases
                        if isinstance(attr_value, th.Tensor):
                            filtered_value = attr_value[valid_mask]
                        else:
                            filtered_value = attr_value[valid_mask]
                        setattr(rollout_buffer, attr_name, filtered_value)
                        
                        if self.verbose >= 2:
                            print(f"Filtered {attr_name}: {len(attr_value)} -> {len(filtered_value)}")
                            
                    except Exception as e:
                        if self.verbose >= 1:
                            print(f"Warning: Could not filter {attr_name}: {e}")

    def _reset_tracking(self) -> None:
        """Reset tracking variables for the next rollout."""
        self.valid_indices = []
        self.current_step = 0
        self.filtered_count = 0
        self.total_count = 0

    def _on_rollout_start(self) -> None:
        """Called at the start of rollout collection."""
        self._reset_tracking()
        if self.verbose >= 2:
            print("Starting new rollout collection")

class CustomPPO(PPO):
    """
    Custom PPO that handles invalid states and integrates with the InvalidSampleRemover callback.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer, 
        n_rollout_steps: int,
    ) -> bool:
        """
        Custom rollout collection that properly integrates with the filtering callback.
        """
        # Switch to eval mode
        self.policy.set_training_mode(False)
        
        n_steps = 0
        rollout_buffer.reset()
        
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        
        callback.on_rollout_start()
        
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            
            with th.no_grad():
                # Get observations using the custom pre_step method
                if hasattr(env, 'pre_step'):
                    obs = env.pre_step()
                else:
                    obs = self._last_obs
                
                obs = np.array(obs)
                obs_tensor = obs_as_tensor(obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                actions_np = actions.cpu().numpy()

            # Step the environment
            new_obs, rewards, dones, infos = env.step(actions_np)
            
            # Ensure infos is always a list
            if not isinstance(infos, list):
                infos = [infos]
            
            # Update callback locals for step tracking
            callback.locals.update({
                'new_obs': new_obs,
                'rewards': rewards, 
                'dones': dones,
                'infos': infos,
                'actions': actions_np,
                'values': values,
                'log_probs': log_probs
            })
            
            # Convert episode_starts from dones
            episode_starts = dones.copy()
            
            # Add to rollout buffer
            rollout_buffer.add(
                obs,
                actions_np,
                rewards,
                episode_starts,
                values,
                log_probs,
            )
            
            # Call callback step
            if callback.on_step() is False:
                return False
            
            # Update for next iteration
            self._last_obs = new_obs
            n_steps += env.num_envs
            
            if self.verbose >= 2 and n_steps % (env.num_envs * 10) == 0:
                print(f"Collected {n_steps}/{n_rollout_steps} steps")

        # Compute returns and advantages before filtering
        with th.no_grad():
            obs_tensor = obs_as_tensor(new_obs, self.device) 
            _, values, _ = self.policy(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        
        # This will trigger the filtering in the callback
        callback.on_rollout_end()
        
        # Check if we still have enough data after filtering
        if rollout_buffer.size() == 0:
            if self.verbose >= 1:
                print("Warning: Rollout buffer is empty after filtering. Skipping this training iteration.")
            return False
        
        if rollout_buffer.size() < self.batch_size:
            if self.verbose >= 1:
                print(f"Warning: Rollout buffer too small ({rollout_buffer.size()}) after filtering. Skipping this training iteration.")
            return False
        
        return True

    def train(self) -> None:
        """
        Override train method to handle filtered buffers gracefully.
        """
        # Check if we have enough data to train
        if self.rollout_buffer.size() == 0:
            if self.verbose >= 1:
                print("Warning: No data in rollout buffer, skipping training step")
            return
        
        if self.rollout_buffer.size() < self.batch_size:
            if self.verbose >= 1:
                print(f"Warning: Buffer size ({self.rollout_buffer.size()}) smaller than batch size ({self.batch_size}), skipping training step")
            return
        
        # Call parent train method
        super().train()

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    """
    def __init__(self, feature_dim: int, last_layer_dim_pi: int = 6, last_layer_dim_vf: int = 1):
        super().__init__()
        
        # Required attributes for Stable Baselines3
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Action transformation parameters
        self.n_variances = 3
        self.n_correlations = 3
        self.min_var = 1e-10
        
        # Actor (Policy) network
        self.actor_layer1 = nn.Linear(feature_dim, 64)
        self.actor_layer2 = nn.Linear(64, 64)
        self.actor_layer3 = nn.Linear(64, 32)
        self.actor_mean_layer = nn.Linear(32, last_layer_dim_pi)
        self.actor_log_std_layer = nn.Linear(32, last_layer_dim_pi)
        
        # Critic (Value) network
        self.critic_layer1 = nn.Linear(feature_dim, 128)
        self.critic_layer2 = nn.Linear(128, 128)
        self.critic_value_layer = nn.Linear(128, last_layer_dim_vf)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for PPO."""
        # Actor weights
        for layer in [self.actor_layer1, self.actor_layer2, self.actor_layer3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        
        # Actor output layers
        nn.init.orthogonal_(self.actor_mean_layer.weight, gain=0.01)
        nn.init.constant_(self.actor_mean_layer.bias, 0.0)
        nn.init.orthogonal_(self.actor_log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.actor_log_std_layer.bias, -1.0)
        
        # Critic weights
        nn.init.orthogonal_(self.critic_layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.critic_layer2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.critic_layer1.bias, 0.0)
        nn.init.constant_(self.critic_layer2.bias, 0.0)
        
        nn.init.orthogonal_(self.critic_value_layer.weight, gain=1.0)
        nn.init.constant_(self.critic_value_layer.bias, 0.0)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """Forward pass for actor network."""
        x = th.tanh(self.actor_layer1(features))
        x = th.tanh(self.actor_layer2(x))
        x = th.tanh(self.actor_layer3(x))
        
        mean = self.actor_mean_layer(x)
        log_std = self.actor_log_std_layer(x).clamp(-20, 2)
        
        return th.cat([mean, log_std], dim=-1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """Forward pass for critic network."""
        x = self.critic_layer1(features)
        x = th.relu(x)
        x = self.critic_layer2(x)
        x = th.relu(x)
        return self.critic_value_layer(x)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass returning both actor and critic outputs."""
        return self.forward_actor(features), self.forward_critic(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    """Custom Actor-Critic policy with action transformations."""
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """Create the policy and value networks."""
        self.mlp_extractor = CustomNetwork(self.features_dim)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> th.distributions.Distribution:
        """Get action distribution from latent representation."""
        # Split the actor output into mean and log_std
        mean, log_std = th.chunk(latent_pi, 2, dim=-1)
        std = log_std.exp()
        
        # Create normal distribution
        return th.distributions.Normal(mean, std)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass to predict actions."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Sample actions
        if deterministic:
            actions = distribution.mean
        else:
            actions = distribution.sample()
        
        # Apply transformations
        actions = self._transform_actions(actions)
        
        # Get log probabilities (need to account for transformation)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        
        # Get values
        values = self.value_net(latent_vf)
        
        return actions, values, log_prob

    def _transform_actions(self, raw_actions: th.Tensor) -> th.Tensor:
        """Transform raw actions using softplus and tanh."""
        variances = th.nn.functional.softplus(raw_actions[..., :3]) + 1e-10
        correlations = th.tanh(raw_actions[..., 3:])
        return th.cat([variances, correlations], dim=-1)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Evaluate actions according to the current policy."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Inverse transform actions to get raw actions
        raw_actions = self._inverse_transform_actions(actions)
        
        # Get log probabilities
        log_prob = distribution.log_prob(raw_actions).sum(dim=-1)
        
        # Get values
        values = self.value_net(latent_vf)
        
        # Get entropy
        entropy = distribution.entropy().sum(dim=-1)
        
        return values, log_prob, entropy

    def _inverse_transform_actions(self, actions: th.Tensor) -> th.Tensor:
        """Inverse transform actions."""
        variances = actions[..., :3]
        correlations = actions[..., 3:]
        
        # Inverse transformations
        raw_variances = th.log(th.clamp(variances - 1e-10, min=1e-10))
        raw_correlations = th.atanh(th.clamp(correlations, min=-0.999, max=0.999))
        
        return th.cat([raw_variances, raw_correlations], dim=-1)

class SerowEnv(gym.Env):
    """SEROW Gymnasium environment for reinforcement learning."""
    
    def __init__(self, robot, initial_state, contact_frame, state_dim, action_dim, measurements, action_bounds=None):
        super().__init__()
        
        # Initialize SEROW framework
        self.serow_framework = serow.Serow()
        self.robot = robot
        self.serow_framework.initialize(f"{self.robot}_rl.json")

        # Set up initial state
        self.initial_state = self.serow_framework.get_state(allow_invalid=True)
        self.initial_state.set_joint_state(initial_state['joint_state'])
        self.initial_state.set_base_state(initial_state['base_state'])  
        self.initial_state.set_contact_state(initial_state['contact_state'])
        self.serow_framework.set_state(self.initial_state)

        # Contact frames
        self.contact_frames = self.initial_state.get_contacts_frame()
        self.cf = contact_frame if contact_frame in self.contact_frames else self.contact_frames[0]

        # Dimensions
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Measurements
        self.imu_measurements = measurements['imu']
        self.joint_measurements = measurements['joints']
        self.force_torque_measurements = measurements['ft']
        self.base_pose_ground_truth = measurements['base_pose_ground_truth']
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = len(self.imu_measurements) - 1
        
        # Define action and observation spaces
        if action_bounds is None:
            action_bounds = gym.spaces.Box(
                low=np.array([1e-10, 1e-10, 1e-10, -1.0, -1.0, -1.0]), 
                high=np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]), 
                shape=(action_dim,), 
                dtype=np.float32
            )
        
        self.action_space = action_bounds
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=np.float32
        )
        
        # State tracking
        self.kin = None
        self.state = None

    def _compute_reward(self, state, gt, step, max_steps):
        """Compute reward based on state estimation accuracy."""
        try:
            # Position error
            position_error = state.get_base_position() - gt.position
            # Orientation error
            orientation_error = logMap(
                quaternion_to_rotation_matrix(gt.orientation).transpose() 
                @ quaternion_to_rotation_matrix(state.get_base_orientation())
            )
                
            # Check for failure conditions
            if (np.linalg.norm(position_error) > 0.5 or 
                np.linalg.norm(orientation_error) > 0.2):
                return -1.0, True, False
            
            # Base survival reward
            reward = (step + 1) / max_steps
            
            # Position accuracy reward
            position_error_cov = state.get_base_position_cov() + np.eye(3) * 1e-8
            position_error_weighted = position_error.T @ np.linalg.inv(position_error_cov) @ position_error
            position_reward = np.exp(-800.0 * position_error_weighted)
            
            # Orientation accuracy reward
            orientation_error_cov = state.get_base_orientation_cov() + np.eye(3) * 1e-8
            orientation_error_weighted = orientation_error.T @ np.linalg.inv(orientation_error_cov) @ orientation_error
            orientation_reward = np.exp(-600.0 * orientation_error_weighted)
            
            # Combine rewards
            total_reward = (reward + position_reward + orientation_reward) / 3.0
            
            # Check for episode termination
            done = False
            truncated = (step + 1 >= int(0.85 * max_steps))
            
            return float(total_reward), done, truncated
            
        except Exception as e:
            print(f"Warning: Reward computation failed: {e}")
            return -1.0, True, False

    def _compute_state(self, state, kin):
        """Compute the observation state."""
        try:
            R_base = quaternion_to_rotation_matrix(state.get_base_orientation()).transpose()
            local_pos = R_base @ (state.get_contact_position(self.cf) - state.get_base_position())   
            local_kin_pos = kin.contacts_position[self.cf]
            local_vel = state.get_base_linear_velocity()
            contact_prob = np.array([kin.contacts_probability[self.cf]])
            
            return np.concatenate([local_pos - local_kin_pos, local_vel, contact_prob], axis=0)
        except Exception as e:
            print(f"Warning: State computation failed: {e}")
            return np.zeros(self.state_dim)

    def _predict_step(self, imu, joint, ft):
        """Predict the environment state."""
        # Process the measurements
        imu, kin, ft = self.serow_framework.process_measurements(imu, joint, ft, None)

        # Predict the base state
        self.serow_framework.base_estimator_predict_step(imu, kin)

        # Get the state
        state = self.serow_framework.get_state(allow_invalid=True)
        return kin, state

    def _update_step(self, cf, kin, action, gt, step, max_steps):
        """Update the environment state with the action."""
        self.serow_framework.set_action(cf, action)

        # Run the update step with the contact position
        self.serow_framework.base_estimator_update_with_contact_position(cf, kin)

        # Get the post state
        state = self.serow_framework.get_state(allow_invalid=True)

        # Compute the reward
        reward, done, truncated = self._compute_reward(state, gt, step, max_steps)
        return state, reward, done, truncated

    def _finish_update(self, imu, kin):
        """Finish the update step."""
        self.serow_framework.base_estimator_finish_update(imu, kin)
    
    def _get_measurements(self, step):
        """Get the measurements for the current step."""
        return (self.imu_measurements[step], 
                self.joint_measurements[step], 
                self.force_torque_measurements[step], 
                self.base_pose_ground_truth[step])
    
    def pre_step(self):
        """Prepare the environment state before action execution."""
        # Get measurements
        imu, joint, ft, gt = self._get_measurements(self.step_count)
        
        # Predict state
        kin, prior_state = self._predict_step(imu, joint, ft)
        
        # Update other contact frames
        post_state = prior_state
        for cf in self.contact_frames:
            if (cf != self.cf and 
                post_state.get_contact_position(cf) is not None and 
                kin.contacts_status[cf]):
                try:
                    post_state, _, _, _ = self._update_step(
                        cf, kin, np.zeros(self.action_dim), gt, self.step_count, self.max_steps
                    )
                except Exception as e:
                    print(f"Warning: Update failed for {cf}: {e}")
        
        # Compute state
        if (post_state.get_contact_position(self.cf) is not None and 
            kin.contacts_status[self.cf]):
            state = self._compute_state(post_state, kin)
        else:
            state = np.zeros(self.state_dim, dtype=np.float64)
        
        self.kin = kin
        return state

    def step(self, action):
        # Ensure action is the right type and shape
        action = np.array(action, dtype=np.float64)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Action dimension mismatch. Expected {self.action_dim}, got {action.shape[0]}")
        
        # Get the current measurements
        imu, _, _, gt = self._get_measurements(self.step_count)
        kin = self.kin
        post_state = self.serow_framework.get_state(allow_invalid=True)
        
        # Initialize tracking variables
        reward = 0.0
        state = np.zeros(self.state_dim, dtype=np.float64)
        done = False
        truncated = False
        valid_step = True
        invalid_reason = "valid"
        
        # Execute the action
        if (post_state.get_contact_position(self.cf) is not None and 
            kin.contacts_status[self.cf]):
            try:
                post_state, reward, done, truncated = self._update_step(
                    self.cf, kin, action, gt, self.step_count, self.max_steps)
                state = self._compute_state(post_state, kin)
            except Exception as e:
                print(f"Warning: Update step failed: {e}")
                valid_step = False
                invalid_reason = f"update_failed: {str(e)}"
        else:
            valid_step = False
            invalid_reason = "no_contact"
        
        # Finish the update step
        self._finish_update(imu, kin)

        # Ensure all return values are valid
        info = {
            "step": self.step_count, 
            "contact_active": kin.contacts_status[self.cf],
            "valid_step": valid_step,
            "invalid_reason": invalid_reason
        }

        # Update the step count
        self.step_count += 1
        
        # Check if we've reached the end of the episode
        if self.step_count >= 0.85 * self.max_steps and reward is not None:
            truncated = True
            done = False
            
        return state, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reinitialize SEROW 
        self.serow_framework = serow.Serow()
        self.serow_framework.initialize(f"{self.robot}_rl.json")
        self.serow_framework.set_state(self.initial_state)
        self.step_count = 0

        # Take steps till the state is valid
        reward = None
        while reward is None:
           self.pre_step()
           state, reward, _, _, _ = self.step(np.zeros(self.action_dim))
        return state, {}
    
    def render(self, mode='human'):
        """Render the environment (placeholder)."""
        pass
    
    def close(self):
       """Close the environment."""
       pass

class SerowVecEnv(DummyVecEnv):
    """Vectorized environment wrapper for SEROW environments."""

    def __init__(self, env_fns):
        super().__init__(env_fns)

    def pre_step(self):
        observations = []
        for env in self.envs:
            obs = env.pre_step()
            observations.append(obs)
        return np.array(observations)

if __name__ == "__main__":
    # Load and preprocess the data
    print("Loading measurements...")
    imu_measurements  = read_imu_measurements("/tmp/serow_measurements.mcap")
    joint_measurements = read_joint_measurements("/tmp/serow_measurements.mcap")
    force_torque_measurements = read_force_torque_measurements("/tmp/serow_measurements.mcap")
    base_pose_ground_truth = read_base_pose_ground_truth("/tmp/serow_measurements.mcap")
    base_states = read_base_states("/tmp/serow_proprioception.mcap")
    contact_states = read_contact_states("/tmp/serow_proprioception.mcap")
    joint_states = read_joint_states("/tmp/serow_proprioception.mcap")

    measurements = {
        'imu': imu_measurements,
        'joints': joint_measurements,
        'ft': force_torque_measurements,
        'base_pose_ground_truth': base_pose_ground_truth
    }

    # Define the dimensions of your state and action spaces
    state_dim = 7  
    action_dim = 6  # Based on the action vector used in ContactEKF.setAction()
    min_action = 1e-10
    robot = "go2"

    # Create the initial state
    initial_state = {
        'joint_state': joint_states[0],
        'base_state': base_states[0],
        'contact_state': contact_states[0]
    }

    print("Creating environments...")
    # Create environments
    env_fns = [
        lambda cf=cf: SerowEnv(
            robot=robot, 
            initial_state=initial_state, 
            contact_frame=cf, 
            state_dim=state_dim, 
            action_dim=action_dim, 
            measurements=measurements
        ) for cf in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    ]
    vec_env = SerowVecEnv(env_fns)
    
    # Create custom callbacks
    filter_callback = InvalidSampleRemover(verbose=1)
    
    # Combine callbacks
    callbacks = [filter_callback]

    # Instantiate the custom PPO with custom rollout collection
    model = CustomPPO(
        CustomActorCriticPolicy, 
        vec_env, 
        device='cpu', 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    # Train the model with custom callbacks
    model.learn(total_timesteps=100000, callback=callbacks)
    print("Training completed!")
    model.save("serow_ppo_model")
