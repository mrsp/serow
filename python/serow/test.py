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
    This version maintains separate valid/invalid buffers and only trains on valid data.
    More aggressive but ensures no invalid data is used for training.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.valid_buffer_data = []
        self.filtered_count = 0
        self.total_count = 0
        
    def _on_step(self) -> bool:
        """Collect only valid samples for training."""
        try:
            infos = self.locals.get('infos', [])
            obs = self.locals.get('new_obs', [])
            actions = self.locals.get('actions', [])
            rewards = self.locals.get('rewards', [])
            dones = self.locals.get('dones', [])
            
            # Only store valid transitions
            for env_idx, info in enumerate(infos):
                self.total_count += 1
                
                if info.get('valid_step', True):
                    # This is a valid sample, keep it
                    valid_data = {
                        'obs': obs[env_idx] if env_idx < len(obs) else None,
                        'action': actions[env_idx] if env_idx < len(actions) else None,
                        'reward': rewards[env_idx] if env_idx < len(rewards) else None,
                        'done': dones[env_idx] if env_idx < len(dones) else None,
                        'info': info
                    }
                    self.valid_buffer_data.append(valid_data)
                else:
                    # Invalid sample, skip it
                    self.filtered_count += 1
                    if self.verbose >= 1:
                        reason = info.get('invalid_reason', 'unknown')
                        print(f"Skipping invalid sample: env {env_idx}, reason: {reason}")
            
        except Exception as e:
            if self.verbose >= 1:
                warnings.warn(f"Error in sample collection: {e}")
        
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout to filter out invalid samples from PPO's buffer."""
        try:
            # Get the rollout buffer from the model
            rollout_buffer = self.model.rollout_buffer
            
            # Create masks for valid samples
            valid_mask = np.array([info.get('valid_step', True) for info in self.locals.get('infos', [])])
            
            # Filter out invalid samples from the buffer
            if hasattr(rollout_buffer, 'observations'):
                rollout_buffer.observations = rollout_buffer.observations[valid_mask]
            if hasattr(rollout_buffer, 'actions'):
                rollout_buffer.actions = rollout_buffer.actions[valid_mask]
            if hasattr(rollout_buffer, 'rewards'):
                rollout_buffer.rewards = rollout_buffer.rewards[valid_mask]
            if hasattr(rollout_buffer, 'dones'):
                rollout_buffer.dones = rollout_buffer.dones[valid_mask]
            if hasattr(rollout_buffer, 'values'):
                rollout_buffer.values = rollout_buffer.values[valid_mask]
            if hasattr(rollout_buffer, 'log_probs'):
                rollout_buffer.log_probs = rollout_buffer.log_probs[valid_mask]
            if hasattr(rollout_buffer, 'advantages'):
                rollout_buffer.advantages = rollout_buffer.advantages[valid_mask]
            
            if self.verbose >= 1:
                print(f"Filtered {len(valid_mask) - np.sum(valid_mask)} invalid samples from rollout buffer")
                
        except Exception as e:
            if self.verbose >= 1:
                warnings.warn(f"Error in filtering rollout buffer: {e}")

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function that replicates the Actor and Critic from train_ppo.py.
    It receives as input the features extracted by the features extractor.
    """
    def __init__(self, feature_dim: int, last_layer_dim_pi: int = 6, last_layer_dim_vf: int = 1):
        super().__init__()
        
        # Required attributes for Stable Baselines3
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Action transformation parameters (from train_ppo.py)
        self.n_variances = 3
        self.n_correlations = 3
        self.min_var = 1e-10  # Default min_action from your code
        
        # Actor (Policy) network - replicating from train_ppo.py
        self.actor_layer1 = nn.Linear(feature_dim, 64)
        self.actor_layer2 = nn.Linear(64, 64)
        self.actor_layer3 = nn.Linear(64, 32)
        self.actor_mean_layer = nn.Linear(32, last_layer_dim_pi)
        self.actor_log_std_layer = nn.Linear(32, last_layer_dim_pi)
        
        # Critic (Value) network - replicating from train_ppo.py
        self.critic_layer1 = nn.Linear(feature_dim, 128)
        self.critic_layer2 = nn.Linear(128, 128)
        self.critic_value_layer = nn.Linear(128, last_layer_dim_vf)
        
        # Initialize weights like in train_ppo.py
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for PPO like in train_ppo.py"""
        # Actor weights
        for layer in [self.actor_layer1, self.actor_layer2, self.actor_layer3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        
        # Actor output layers
        nn.init.orthogonal_(self.actor_mean_layer.weight, gain=0.01)
        nn.init.constant_(self.actor_mean_layer.bias, 0.0)
        nn.init.orthogonal_(self.actor_log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.actor_log_std_layer.bias, -1.0)  # Start with small std
        
        # Critic weights
        nn.init.orthogonal_(self.critic_layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.critic_layer2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.critic_layer1.bias, 0.0)
        nn.init.constant_(self.critic_layer2.bias, 0.0)
        
        nn.init.orthogonal_(self.critic_value_layer.weight, gain=1.0)
        nn.init.constant_(self.critic_value_layer.bias, 0.0)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """Forward pass for actor network with tanh activation like in train_ppo.py"""
        x = th.tanh(self.actor_layer1(features))
        x = th.tanh(self.actor_layer2(x))
        x = th.tanh(self.actor_layer3(x))
        
        mean = self.actor_mean_layer(x)
        log_std = self.actor_log_std_layer(x).clamp(-20, 2)  # Constrain log_std
        
        return th.cat([mean, log_std], dim=-1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """Forward pass for critic network with ReLU activation like in train_ppo.py"""
        x = self.critic_layer1(features)
        x = th.relu(x)
        x = self.critic_layer2(x)
        x = th.relu(x)
        return self.critic_value_layer(x)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass returning both actor and critic outputs"""
        return self.forward_actor(features), self.forward_critic(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None, # Not used in this custom example
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch, # Pass the default net_arch, but we'll override mlp_extractor
            activation_fn,
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the `_build` method.
        """
        self.mlp_extractor = CustomNetwork(self.features_dim)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in the neural network to predict the next action.
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Get actor and critic outputs
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get the value function output
        values = self.value_net(latent_vf)
        
        # Get the distribution and sample actions
        distribution = self.get_distribution(obs)
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob

    def get_distribution(self, obs: th.Tensor) -> th.distributions.Distribution:
        """
        Get the current policy distribution given the observations.
        Override to handle action transformations like in train_ppo.py.
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Get actor output
        latent_pi, _ = self.mlp_extractor(features)
        
        # Split the actor output into mean and log_std
        mean, log_std = th.chunk(latent_pi, 2, dim=-1)
        
        # Create the base distribution
        std = log_std.exp()
        base_distribution = th.distributions.Normal(mean, std)
        
        # Create a custom distribution that applies the transformations
        class TransformedDistribution(th.distributions.Distribution):
            def __init__(self, base_dist, n_variances, n_correlations, min_var):
                super().__init__()
                self.base_dist = base_dist
                self.n_variances = n_variances
                self.n_correlations = n_correlations
                self.min_var = min_var
            
            def sample(self, sample_shape=th.Size()):
                # Sample from base distribution
                raw_samples = self.base_dist.sample(sample_shape)
                
                # Apply transformations
                variances = th.nn.functional.softplus(raw_samples[..., :self.n_variances]) + self.min_var
                correlations = th.tanh(raw_samples[..., self.n_variances:])
                transformed_samples = th.cat((variances, correlations), dim=-1)
                
                return transformed_samples
            
            def log_prob(self, value):
                # Transform back to raw space
                action_variances = value[..., :self.n_variances]
                action_correlations = value[..., self.n_variances:]
                
                raw_variances = th.log(action_variances - self.min_var)
                raw_correlations = th.atanh(action_correlations)
                raw_value = th.cat([raw_variances, raw_correlations], dim=-1)
                
                # Get base log probability
                log_prob = self.base_dist.log_prob(raw_value).sum(dim=-1)
                
                # Apply jacobian correction
                var_jacobian = th.sigmoid(raw_value[..., :self.n_variances])
                corr_jacobian = 1 - th.tanh(raw_value[..., self.n_variances:])**2
                jacobian = th.cat([var_jacobian, corr_jacobian], dim=-1)
                
                log_prob = log_prob - th.log(jacobian).sum(dim=-1)
                return log_prob
            
            def entropy(self):
                return self.base_dist.entropy().sum(dim=-1)
        
        return TransformedDistribution(
            base_distribution, 
            self.mlp_extractor.n_variances, 
            self.mlp_extractor.n_correlations, 
            self.mlp_extractor.min_var
        )

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        Override to handle action transformations like in train_ppo.py.
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Get actor and critic outputs
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Split the actor output into mean and log_std
        mean, log_std = th.chunk(latent_pi, 2, dim=-1)
        
        # Transform actions back to raw space (inverse of the transformation)
        # Split actions into variances and correlations
        action_variances = actions[..., :self.mlp_extractor.n_variances]
        action_correlations = actions[..., self.mlp_extractor.n_variances:]
        
        # Inverse transformations
        raw_variances = th.log(action_variances - self.mlp_extractor.min_var)  # inverse of softplus
        raw_correlations = th.atanh(action_correlations)  # inverse of tanh
        raw_actions = th.cat([raw_variances, raw_correlations], dim=-1)
        
        # Create distribution with raw parameters
        std = log_std.exp()
        distribution = th.distributions.Normal(mean, std)
        
        # Compute log probabilities with change of variables
        log_probs = distribution.log_prob(raw_actions).sum(dim=-1)
        
        # Compute jacobian for the transformation
        var_jacobian = th.sigmoid(raw_actions[..., :self.mlp_extractor.n_variances])
        corr_jacobian = 1 - th.tanh(raw_actions[..., self.mlp_extractor.n_variances:])**2
        jacobian = th.cat([var_jacobian, corr_jacobian], dim=-1)
        
        # Apply jacobian correction
        log_probs = log_probs - th.log(jacobian).sum(dim=-1)
        
        # Get values
        values = self.value_net(latent_vf)
        
        # Get entropy
        entropy = distribution.entropy().sum(dim=-1)
        
        return values, log_probs, entropy

class SerowEnv(gym.Env):
    def __init__(self, robot, initial_state, contact_frame, state_dim, action_dim, measurements, action_bounds=None):
        super().__init__()
        self.serow_framework = serow.Serow()
        self.robot = robot
        self.serow_framework.initialize(f"{self.robot}_rl.json")

        # Initial state
        self.initial_state = self.serow_framework.get_state(allow_invalid=True)
        self.initial_state.set_joint_state(initial_state['joint_state'])
        self.initial_state.set_base_state(initial_state['base_state'])  
        self.initial_state.set_contact_state(initial_state['contact_state'])
        self.serow_framework.set_state(self.initial_state)

        # Contact frames
        self.contact_frames = self.initial_state.get_contacts_frame()

        # Action and state dimensions
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Contact frame to control 
        self.cf = contact_frame if contact_frame in self.contact_frames else self.contact_frames[0]

        # Measurements
        self.imu_measurements = measurements['imu']
        self.joint_measurements = measurements['joints']
        self.force_torque_measurements = measurements['ft']
        self.base_pose_ground_truth = measurements['base_pose_ground_truth']
        self.step_count = 0
        self.max_steps = len(self.imu_measurements) - 1
        
        # Define action and observation spaces (REQUIRED for SB3)
        if action_bounds is None:
            # Default action bounds - using reasonable finite values
            action_bounds = gym.spaces.Box(low=-np.ones(action_dim) * 10.0, 
                                       high=np.ones(action_dim) * 10.0, shape=(action_dim,), 
                                       dtype=np.float64)
        
        self.action_space = gym.spaces.Box(
            low=action_bounds.low, 
            high=action_bounds.high, 
            shape=(action_dim,), 
            dtype=np.float64
        )
        
        # Observation space - adjust bounds based on your state space
        # Using conservative bounds - you should adjust these based on your actual state ranges
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=np.float64
        )
        
        self.render_mode = None
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

    def _compute_reward(self, state, gt, step, max_steps):
        reward = 0.0
        done = False
        truncated = False

        try:
            # Position error
            position_error = state.get_base_position() - gt.position
            # Orientation error
            orientation_error = logMap(quaternion_to_rotation_matrix(gt.orientation).transpose() 
                                       @ quaternion_to_rotation_matrix(state.get_base_orientation()))
                
            if (np.linalg.norm(position_error) > 0.5 or np.linalg.norm(orientation_error) > 0.2):
                done = True  
                reward = -1.0  # Small negative reward instead of 0 for failed episodes
            else:
                done = False
                reward = (step + 1) / max_steps
                    
                position_error_cov = state.get_base_position_cov() + np.eye(3) * 1e-8
                position_error = position_error.dot(np.linalg.inv(position_error_cov).dot(position_error))
                alpha_pos = 800.0
                position_reward = np.exp(-alpha_pos * position_error)
                reward += position_reward

                orientation_error_cov = state.get_base_orientation_cov() + np.eye(3) * 1e-8
                orientation_error = orientation_error.dot(np.linalg.inv(orientation_error_cov).dot(orientation_error))
                alpha_ori = 600.0
                orientation_reward = np.exp(-alpha_ori * orientation_error)
                reward += orientation_reward

                # Normalize the reward
                reward /= 3.0
                
        except Exception as e:
            # If reward computation fails, return safe defaults
            print(f"Warning: Reward computation failed: {e}")
            done = True

        if step + 1 == max_steps:
            truncated = True
            done = False  # Don't set done=True when truncated=True

        return float(reward), done, truncated    

    def _compute_state(self, state, kin):
        R_base = quaternion_to_rotation_matrix(state.get_base_orientation()).transpose()
        local_pos = R_base @ (state.get_contact_position(self.cf) - state.get_base_position())   
        local_kin_pos = kin.contacts_position[self.cf]
        local_vel = state.get_base_linear_velocity()  
        return np.concatenate((local_pos - local_kin_pos, local_vel, 
                               np.array([kin.contacts_probability[self.cf]])), axis=0)

    def _predict_step(self, imu, joint, ft):
        # Process the measurements
        imu, kin, ft = self.serow_framework.process_measurements(imu, joint, ft, None)

        # Predict the base state
        self.serow_framework.base_estimator_predict_step(imu, kin)

        # Get the state
        state = self.serow_framework.get_state(allow_invalid=True)
        return kin, state

    def _update_step(self, cf, kin, action, gt, step, max_steps):
        self.serow_framework.set_action(cf, action)

        # Run the update step with the contact position
        self.serow_framework.base_estimator_update_with_contact_position(cf, kin)

        # Get the post state
        state = self.serow_framework.get_state(allow_invalid=True)

        # Compute the reward
        reward, done, truncated = self._compute_reward(state, gt, step, max_steps)
        return state, reward, done, truncated

    def _finish_update(self, imu, kin):
        self.serow_framework.base_estimator_finish_update(imu, kin)
    
    def _get_measurements(self, step):
        return (self.imu_measurements[step], 
                self.joint_measurements[step], 
                self.force_torque_measurements[step], 
                self.base_pose_ground_truth[step])
    
    def pre_step(self):
        # Get the current measurements
        imu, joint, ft, gt = self._get_measurements(self.step_count)

        # Predict the SEROW state
        kin, prior_state = self._predict_step(imu, joint, ft)

        # Update the state for the rest of the contact frames
        post_state = prior_state
        state = np.zeros(self.state_dim)
        for cf in self.contact_frames:
            if (cf != self.cf and 
                post_state.get_contact_position(cf) is not None and 
                kin.contacts_status[cf]):
                try:
                    post_state, _, _, _ = self._update_step(
                        cf, kin, np.zeros(self.action_dim), gt, self.step_count, self.max_steps)
                    if post_state.get_contact_position(self.cf) is not None and kin.contacts_status[self.cf]:
                        state = self._compute_state(post_state, kin)
                except Exception as e:
                    print(f"Warning: Update step failed for contact frame {cf}: {e}")

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
        state = np.zeros(self.state_dim)
        done = False
        truncated = False
        valid_step = True
        invalid_reason = "valid"
        
        # Update the state for the actionable contact frame
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
            "invalid_reason": invalid_reason,
            "reward": reward,  # Store original reward value
            "state_available": state is not None,
            "fallback_used": not valid_step
        }

        # Update the step count
        self.step_count += 1
        
        # Check if we've reached the end of the episode
        if self.step_count >= 0.85 * self.max_steps and reward is not None:
            truncated = True
            done = False
            
        return state, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.serow_framework = serow.Serow()
        self.serow_framework.initialize(f"{self.robot}_rl.json")
        self.serow_framework.set_state(self.initial_state)
        self.step_count = 0
        reward = None
        while reward is None:
           self.pre_step()
           state, reward, _, _, _ = self.step(np.zeros(self.action_dim))
        return state, {}
    
    def render(self, mode='human'):
        # Implement rendering if needed
        pass
    
    def close(self):
        if self.screen is not None:
            self.screen = None
        self.isopen = False

class SerowVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)

    def pre_step(self):
        observations = []
        for env in self.envs:
            obs = env.pre_step()
            observations.append(obs)
        return observations

class CustomPPO(PPO):
    """
    Custom PPO that handles invalid states and custom rollout collection.
    """
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Custom rollout collection that handles invalid states.
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
                # Convert to pytorch tensor or to TensorDict
                last_obs = env.pre_step()
                last_obs = np.array(last_obs)
                action, value, log_prob = self.policy(obs_as_tensor(last_obs, self.device))
                action = action.detach().cpu().numpy()
                value = value.detach().cpu().numpy()
                log_prob = log_prob.detach().cpu().numpy()
                    

            # Convert to numpy arrays
            new_obs, rewards, dones, infos = env.step(action)
            rollout_buffer.add(
                new_obs,
                action,
                rewards,
                dones,
                value,
                log_prob,
            )

            # Update callback
            callback.update_locals(locals(), verbose=self.verbose)
            if callback.on_step() is False:
                return False
            
            n_steps += len(new_obs)  
            
            # Early termination if all environments are done
            if dones.any():
                # Get indices of done environments
                done_indices = np.where(dones)[0]
                # Reset those environments
                for done_idx in done_indices:
                    self._last_obs[done_idx] = env.reset()[0][done_idx]
        
        # Compute value for the last timestep
        with th.no_grad():
            obs_tensor = obs_as_tensor(new_obs, self.device)
            _, values, _ = self.policy(obs_tensor)
        
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        
        callback.on_rollout_end()
        
        return True

if __name__ == "__main__":
    # Load and preprocess the data
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

    # Create environments
    env_fns = [
        lambda: SerowEnv(robot=robot, initial_state=initial_state, contact_frame="FL_foot", state_dim=state_dim, action_dim=action_dim, measurements=measurements),
        lambda: SerowEnv(robot=robot, initial_state=initial_state, contact_frame="FR_foot", state_dim=state_dim, action_dim=action_dim, measurements=measurements),
        lambda: SerowEnv(robot=robot, initial_state=initial_state, contact_frame="RL_foot", state_dim=state_dim, action_dim=action_dim, measurements=measurements),
        lambda: SerowEnv(robot=robot, initial_state=initial_state, contact_frame="RR_foot", state_dim=state_dim, action_dim=action_dim, measurements=measurements)
    ]
    vec_env = SerowVecEnv(env_fns)
    
    # Create custom callbacks
    filter_callback = InvalidSampleRemover(verbose=1)
    
    # Combine callbacks
    callbacks = [filter_callback]

    # Instantiate the custom PPO with custom rollout collection
    model = CustomPPO(CustomActorCriticPolicy, vec_env, device='cpu', verbose=1)

    # Train the model with custom callbacks
    model.learn(total_timesteps=100000, callback=callbacks)
