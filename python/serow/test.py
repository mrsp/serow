import gymnasium as gym
from typing import Callable, Optional, List, Union, Dict, Type, Tuple
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import warnings



from read_mcap import(
    read_base_states, 
    read_contact_states, 
    read_force_torque_measurements, 
    read_joint_measurements, 
    read_imu_measurements, 
    read_base_pose_ground_truth,
    read_joint_states
)

import numpy as np
import serow
from utils import quaternion_to_rotation_matrix, logMap, sync_and_align_data, plot_trajectories

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
    
    def get_valid_data(self):
        """Get the collected valid data."""
        return self.valid_buffer_data
    
    def clear_buffer(self):
        """Clear the valid data buffer."""
        self.valid_buffer_data.clear()

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.
    """
    def __init__(self, feature_dim: int, last_layer_dim_pi: int = 6, last_layer_dim_vf: int = 1):
        super().__init__()
        
        # Required attributes for Stable Baselines3
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi),
            nn.ReLU(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
            nn.ReLU(),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf),
            nn.ReLU(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf),
            nn.ReLU(),
        )

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

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
        # Disable orthogonal initialization for now to see if it helps with NaN
        # for param in self.parameters():
        #     if len(param.shape) > 1:
        #         nn.init.orthogonal_(param)


    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the `_build` method.
        """
        # Here, self.features_extractor is the default one (e.g., FlattenExtractor for MlpPolicy)
        # It's output dimension is self.features_dim

        # Instead of using Stable Baselines' default MlpExtractor,
        # we directly use our CustomNetwork here.
        # This CustomNetwork acts as the 'mlp_extractor' in the SB3 policy structure.
        self.mlp_extractor = CustomNetwork(self.features_dim, last_layer_dim_pi=64, last_layer_dim_vf=64)


import gym
import numpy as np
from gym import spaces
import serow

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
            action_bounds = spaces.Box(low=-np.ones(action_dim) * 10.0, 
                                       high=np.ones(action_dim) * 10.0, shape=(action_dim,), 
                                       dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=action_bounds.low, 
            high=action_bounds.high, 
            shape=(action_dim,), 
            dtype=np.float32
        )
        
        # Observation space - adjust bounds based on your state space
        # Using conservative bounds - you should adjust these based on your actual state ranges
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=np.float32
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
            reward = -0.1  # Small penalty for invalid states
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
        action = np.array(action, dtype=np.float64).reshape(-1, 1)
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
    
    def step(self, action):
        # Ensure action is the right type and shape
        action = np.array(action, dtype=np.float32)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Action dimension mismatch. Expected {self.action_dim}, got {action.shape[0]}")
        
        # Get the current measurements
        imu, joint, ft, gt = self._get_measurements(self.step_count)

        # Predict the SEROW state
        kin, prior_state = self._predict_step(imu, joint, ft)
        
        # Initialize tracking variables
        reward = None
        state = None
        done = False
        truncated = False
        valid_step = True
        invalid_reason = "valid"
        
        # Update the state for the actionable contact frame
        if (prior_state.get_contact_position(self.cf) is not None and 
            kin.contacts_status[self.cf]):
            try:
                post_state, reward, done, truncated = self._update_step(
                    self.cf, kin, action, gt, self.step_count, self.max_steps)
                state = self._compute_state(post_state, kin)
            except Exception as e:
                print(f"Warning: Update step failed: {e}")
                post_state = prior_state
                valid_step = False
                invalid_reason = f"update_failed: {str(e)}"
        else:
            post_state = prior_state
            valid_step = False
            invalid_reason = "no_contact"
        
        # Update the state for the rest of the contact frames
        for cf in self.contact_frames:
            if (cf != self.cf and 
                post_state.get_contact_position(cf) is not None and 
                kin.contacts_status[cf]):
                try:
                    post_state, _, _, _ = self._update_step(
                        cf, kin, np.zeros(self.action_dim), gt, self.step_count, self.max_steps)
                except Exception as e:
                    print(f"Warning: Update step failed for contact frame {cf}: {e}")
        
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
        
        # Return default reward of 0.0 if reward is None
        if reward is None:
            reward = 0.0
            
        return state, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.serow_framework = serow.Serow()
        self.serow_framework.initialize(f"{self.robot}_rl.json")
        self.serow_framework.set_state(self.initial_state)
        self.step_count = 0
        reward = None
        while reward is None:
           state, reward, _, _, _ = self.step(np.zeros(self.action_dim))
        return state, {}
    
    def render(self, mode='human'):
        # Implement rendering if needed
        pass
    
    def close(self):
        if self.screen is not None:
            self.screen = None
        self.isopen = False

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

    # Create environment
    contact_frame = "FL_foot"
    env = SerowEnv(robot, initial_state, contact_frame, state_dim, action_dim, measurements)

    env.reset()
    filter_callback = InvalidSampleRemover()

    # Instantiate the custom policy and pass it to PPO
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=100000, callback=filter_callback)
    env.close()
