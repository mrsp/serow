import gymnasium as gym
import torch as th
import numpy as np
import warnings
import serow
from utils import plot_trajectories
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from typing import Callable, Optional, List, Union, Dict, Type, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
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

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
import gym
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import warnings

class InvalidSampleRemover(BaseCallback):
    """
    Callback that tracks invalid samples and provides filtered data during training.
    Does not modify the rollout buffer - instead provides filtered data access.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.valid_mask = None
        self.filtered_count = 0
        self.total_count = 0
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        if self.verbose >= 1:
            print("InvalidSampleRemover callback initialized")
    
    def _on_rollout_start(self) -> None:
        """Called at the start of rollout collection."""
        # Initialize tracking for new rollout
        rollout_buffer = self.model.rollout_buffer
        buffer_size = rollout_buffer.buffer_size
        self.valid_mask = np.ones(buffer_size, dtype=bool)
        self.filtered_count = 0
        self.total_count = 0
        
        if self.verbose >= 2:
            print(f"Starting new rollout collection with buffer size: {buffer_size}")
    
    def _on_step(self) -> bool:
        """Called after each environment step during rollout collection."""
        try:
            # Get the current step information
            infos = self.locals.get('infos', [])
            rollout_buffer = self.model.rollout_buffer
            
            # Get the current buffer position before this step was added
            current_pos = rollout_buffer.pos
            num_envs = len(infos)
            
            # Calculate the buffer positions for this step's samples
            start_pos = (current_pos - num_envs) % rollout_buffer.buffer_size
            
            # Process each environment's info
            for env_idx, info in enumerate(infos):
                self.total_count += 1
                
                # Calculate the actual buffer position for this sample
                buffer_pos = (start_pos + env_idx) % rollout_buffer.buffer_size
                
                is_valid = info.get('valid_step', True)
                
                if not is_valid:
                    self.filtered_count += 1
                    # Mark this position as invalid
                    if buffer_pos < len(self.valid_mask):
                        self.valid_mask[buffer_pos] = False
                    
                    # if self.verbose >= 1:
                    #     reason = info.get('invalid_reason', 'unknown')
                    #     print(f"Invalid sample detected: buffer_pos {buffer_pos}, reason: {reason}")
            
            if self.verbose >= 2 and self.total_count % 100 == 0:
                valid_count = np.sum(self.valid_mask)
                print(f"Step tracking: total={self.total_count}, filtered={self.filtered_count}, valid={valid_count}")
            
        except Exception as e:
            if self.verbose >= 1:
                warnings.warn(f"Error in sample tracking: {e}")
        
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of rollout collection to set up filtering."""
        try:
            rollout_buffer = self.model.rollout_buffer
            actual_buffer_size = rollout_buffer.buffer_size
            
            valid_count = np.sum(self.valid_mask)
            
            if self.verbose >= 1:
                print(f"Rollout ended. Buffer samples: {actual_buffer_size}, Valid samples: {valid_count}, Filtered: {actual_buffer_size - valid_count}")
            
            # Check if we have enough valid samples for training
            min_valid_samples = max(32, self.model.batch_size)
            if valid_count < min_valid_samples:
                if self.verbose >= 1:
                    print(f"Not enough valid samples ({valid_count}) for training (minimum {min_valid_samples})")
                    print("Training will be skipped for this iteration")
                # Set all samples as invalid to skip training
                self.valid_mask[:] = False
                return
            
            if self.verbose >= 1:
                print(f"Ready for training with {valid_count} valid samples")
                
        except Exception as e:
            if self.verbose >= 1:
                warnings.warn(f"Error in rollout end processing: {e}")

    def get_valid_mask(self):
        """Get the current valid sample mask."""
        return self.valid_mask
    
    def has_valid_samples(self):
        """Check if there are any valid samples."""
        return self.valid_mask is not None and np.any(self.valid_mask)


class CustomPPO(PPO):
    """
    Custom PPO that handles invalid states by filtering data during training.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invalid_sample_remover = None

    def set_invalid_sample_remover(self, callback):
        """Set the InvalidSampleRemover callback reference."""
        self.invalid_sample_remover = callback

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
                
                # IMPORTANT: Check for NaNs/Infs in observations before passing to policy
                if isinstance(obs, np.ndarray):
                    if np.isnan(obs).any() or np.isinf(obs).any():
                        if self.verbose >= 1:
                            warnings.warn("NaN or Inf detected in observations during rollout collection.")
                        # You might want to handle this more gracefully, e.g., skip step, reset env, etc.
                        # For now, let it propagate to see if other mechanisms catch it.
                
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
            
            # Add to rollout buffer (this keeps ALL samples, valid and invalid)
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

        # Compute returns and advantages on ALL data (including invalid)
        with th.no_grad():
            obs_tensor = obs_as_tensor(new_obs, self.device) 
            _, values, _ = self.policy(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        
        # This will set up the filtering mask
        callback.on_rollout_end()
        
        # Check if we have any valid data for training
        if (hasattr(callback, 'has_valid_samples') and 
            not callback.has_valid_samples()):
            if self.verbose >= 1:
                print("Warning: No valid samples for training. Skipping this training iteration.")
            return False
        
        return True

    def train(self) -> None:
        """
        Override train method to use filtered data.
        """
        # Check if we have a filter callback
        if (self.invalid_sample_remover is None or 
            not hasattr(self.invalid_sample_remover, 'get_valid_mask')):
            # No filtering - use parent train method
            super().train()
            return
        
        # Get the valid mask
        valid_mask = self.invalid_sample_remover.get_valid_mask()
        if valid_mask is None or not np.any(valid_mask):
            if self.verbose >= 1:
                print("Warning: No valid samples for training, skipping training step")
            return
        
        # Check if we have enough data to train
        valid_count = np.sum(valid_mask)
        if valid_count < self.batch_size:
            if self.verbose >= 1:
                print(f"Warning: Not enough valid samples ({valid_count}) for batch size ({self.batch_size}), skipping training step")
            return
        
        # Train using only valid samples
        self._train_with_mask(valid_mask)

    def _train_with_mask(self, valid_mask: np.ndarray) -> None:
        """
        Train the policy using only valid samples identified by the mask.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # Get valid data from rollout buffer
        rollout_buffer = self.rollout_buffer

        # Extract valid samples
        valid_indices = np.where(valid_mask)[0]
        
        # Create filtered data
        observations = rollout_buffer.observations[valid_indices]
        actions = rollout_buffer.actions[valid_indices]
        old_values = rollout_buffer.values[valid_indices]
        old_log_prob = rollout_buffer.log_probs[valid_indices]
        advantages = rollout_buffer.advantages[valid_indices]
        returns = rollout_buffer.returns[valid_indices]

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            indices = np.random.permutation(len(valid_indices))

            # Train on mini-batches
            for start_idx in range(0, len(indices), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]

                if len(batch_indices) < self.batch_size:
                    continue  # Skip incomplete batches

                # Get batch data
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_old_log_prob = old_log_prob[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Convert to tensors
                batch_obs = obs_as_tensor(batch_obs, self.device)
                batch_actions = th.as_tensor(batch_actions, device=self.device, dtype=th.float32)
                batch_old_values = th.as_tensor(batch_old_values, device=self.device, dtype=th.float32)
                batch_old_log_prob = th.as_tensor(batch_old_log_prob, device=self.device, dtype=th.float32)
                batch_advantages = th.as_tensor(batch_advantages, device=self.device, dtype=th.float32)
                batch_returns = th.as_tensor(batch_returns, device=self.device, dtype=th.float32)
                
                # Check for NaNs/Infs in batch data before policy evaluation
                if (th.isnan(batch_obs).any() or th.isinf(batch_obs).any() or
                    th.isnan(batch_actions).any() or th.isinf(batch_actions).any() or
                    th.isnan(batch_old_values).any() or th.isinf(batch_old_values).any() or
                    th.isnan(batch_old_log_prob).any() or th.isinf(batch_old_log_prob).any() or
                    th.isnan(batch_advantages).any() or th.isinf(batch_advantages).any() or
                    th.isnan(batch_returns).any() or th.isinf(batch_returns).any()):
                    if self.verbose >= 1:
                        warnings.warn("NaN or Inf detected in batch data during training. Skipping batch.")
                    continue


                values, log_prob, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)
                values = values.flatten()
                
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(batch_advantages) > 1:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - batch_old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = batch_advantages * ratio
                policy_loss_2 = batch_advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    batch_old_values_flat = batch_old_values.flatten()
                    values_pred = batch_old_values_flat + th.clamp(
                        values - batch_old_values_flat, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(batch_returns.flatten(), values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - batch_old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # Log some statistics
        if self.verbose >= 1:
            print(f"Training completed with {len(valid_indices)} valid samples")

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
        # Use this min_var consistently for transformations
        self.min_var_transform = 1e-10 
        
        # Debug flag
        self.verbose = 0
        
        # Actor (Policy) network
        self.actor_layer1 = nn.Linear(feature_dim, 64)
        self.actor_layer2 = nn.Linear(64, 64)
        self.actor_layer3 = nn.Linear(64, 32)

        # Initialize actor layers
        nn.init.orthogonal_(self.actor_layer1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.actor_layer1.bias, 0.0)
        nn.init.orthogonal_(self.actor_layer2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.actor_layer2.bias, 0.0)
        nn.init.orthogonal_(self.actor_layer3.weight, gain=np.sqrt(2))
        nn.init.constant_(self.actor_layer3.bias, 0.0)

        # Output layers for mean and log_std of each parameter
        self.actor_mean_layer = nn.Linear(32, last_layer_dim_pi)
        self.actor_log_std_layer = nn.Linear(32, last_layer_dim_pi)
        
        # Initialize output layers
        nn.init.orthogonal_(self.actor_mean_layer.weight, gain=0.01)
        nn.init.constant_(self.actor_mean_layer.bias, 0.0)
        nn.init.orthogonal_(self.actor_log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.actor_log_std_layer.bias, -1.0)  # Start with small std


        # Critic (Value) network
        self.critic_layer1 = nn.Linear(feature_dim, 128)
        self.critic_layer2 = nn.Linear(128, 128)
        nn.init.orthogonal_(self.critic_layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.critic_layer2.weight, gain=np.sqrt(2))
        th.nn.init.constant_(self.critic_layer1.bias, 0.0)
        th.nn.init.constant_(self.critic_layer2.bias, 0.0)

        self.critic_value_layer = nn.Linear(128, last_layer_dim_vf)
        nn.init.orthogonal_(self.critic_value_layer.weight, gain=1.0)
        th.nn.init.constant_(self.critic_value_layer.bias, 0.0)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """Forward pass for actor network."""
        if hasattr(self, 'verbose') and self.verbose >= 2:
            print(f"Actor forward - features shape: {features.shape}")
        
        x = F.tanh(self.actor_layer1(features))
        x = F.tanh(self.actor_layer2(x))
        x = F.tanh(self.actor_layer3(x))
        
        mean = self.actor_mean_layer(x)
        # Ensure log_std is finite after clamping
        log_std = self.actor_log_std_layer(x).clamp(-20, 2) 
        
        # Add checks here to debug NaN propagation
        # if th.isnan(mean).any():
        #     print("NaN detected in actor mean output!")
        #     import pdb; pdb.set_trace()
        # if th.isnan(log_std).any():
        #     print("NaN detected in actor log_std output!")
        #     import pdb; pdb.set_trace()

        return th.cat([mean, log_std], dim=-1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """Forward pass for critic network."""
        x = self.critic_layer1(features)
        x = F.relu(x)
        x = self.critic_layer2(x)
        x = F.relu(x)
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
        # Set verbose before calling super().__init__
        self.verbose = kwargs.get('verbose', 0)
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        
        # Ensure features_dim is set correctly
        if hasattr(observation_space, 'shape'):
            self.features_dim = observation_space.shape[0]
        else:
            # Fallback for other space types
            self.features_dim = np.prod(observation_space.shape)
        
        print(f"CustomActorCriticPolicy initialized with features_dim: {self.features_dim}")
        print(f"Observation space shape: {observation_space.shape}")
        print(f"Action space shape: {action_space.shape}")

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """Extract features from observations."""
        # Convert to float32 to match network expectations
        return obs.float()

    def _build_mlp_extractor(self) -> None:
        """Create the policy and value networks."""
        print(f"Building MLP extractor with features_dim: {self.features_dim}")
        self.mlp_extractor = CustomNetwork(self.features_dim)
        if hasattr(self, 'verbose'):
            self.mlp_extractor.verbose = self.verbose

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> th.distributions.Distribution:
        """Get action distribution from latent representation."""
        # Split the actor output into mean and log_std
        mean, log_std = th.chunk(latent_pi, 2, dim=-1)
        
        # Ensure std is strictly positive
        std = log_std.exp()
        # Add a small epsilon to std to prevent zero std which can lead to NaNs in log_prob
        # This is also good for exploration.
        std = std + 1e-8 
        
        # if th.isnan(mean).any() or th.isinf(mean).any():
        #     print("NaN/Inf detected in mean before Normal distribution!")
        #     import pdb; pdb.set_trace()
        # if th.isnan(std).any() or th.isinf(std).any():
        #     print("NaN/Inf detected in std before Normal distribution!")
        #     import pdb; pdb.set_trace()

        # Create normal distribution
        return th.distributions.Normal(mean, std)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass to predict actions."""
        if hasattr(self, 'verbose') and self.verbose >= 2:
            print(f"Forward pass - obs shape: {obs.shape}")
        
        features = self.extract_features(obs)
        
        if hasattr(self, 'verbose') and self.verbose >= 2:
            print(f"Forward pass - features shape: {features.shape}")
        
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Sample actions
        if deterministic:
            raw_actions = distribution.mean
        else:
            raw_actions = distribution.sample()
        
        # Apply transformations
        actions = self._transform_actions(raw_actions)
        
        # Compute log probability with change of variables
        # For softplus: d/dx softplus(x) = sigmoid(x)
        # For tanh: d/dx tanh(x) = 1 - tanh^2(x)
        var_jacobian = th.sigmoid(raw_actions[..., :self.mlp_extractor.n_variances])
        corr_jacobian = 1 - th.tanh(raw_actions[..., self.mlp_extractor.n_variances:])**2
        
        # Add a small epsilon to jacobians before taking log to prevent log(0)
        jacobian = th.cat([var_jacobian, corr_jacobian], dim=-1) + 1e-8
        
        # Log probability with change of variables
        # log p(y) = log p(x) - log |det(J)|
        # where J is the jacobian of the transformation
        log_prob = distribution.log_prob(raw_actions).sum(dim=-1) - th.log(jacobian).sum(dim=-1)
        
        # Get values
        values = self.value_net(latent_vf)
        
        return actions, values, log_prob

    def _transform_actions(self, raw_actions: th.Tensor) -> th.Tensor:
        """Transform raw actions using softplus and tanh."""
        # Ensure min_var_transform is used consistently
        variances = F.softplus(raw_actions[..., :self.mlp_extractor.n_variances]) + self.mlp_extractor.min_var_transform
        correlations = th.tanh(raw_actions[..., self.mlp_extractor.n_variances:])
        return th.cat([variances, correlations], dim=-1)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Evaluate actions according to the current policy."""
        if hasattr(self, 'verbose') and self.verbose >= 2:
            print(f"Evaluate actions - obs shape: {obs.shape}, actions shape: {actions.shape}")
        
        features = self.extract_features(obs)
        
        if hasattr(self, 'verbose') and self.verbose >= 2:
            print(f"Evaluate actions - features shape: {features.shape}")
        
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Inverse transform actions to get raw actions
        raw_actions = self._inverse_transform_actions(actions)
        
        # Compute jacobian for the transformation
        var_jacobian = th.sigmoid(raw_actions[..., :self.mlp_extractor.n_variances])
        corr_jacobian = 1 - th.tanh(raw_actions[..., self.mlp_extractor.n_variances:])**2
        
        # Add a small epsilon to jacobians before taking log to prevent log(0)
        jacobian = th.cat([var_jacobian, corr_jacobian], dim=-1) + 1e-8
        
        # Log probability with change of variables
        log_prob = distribution.log_prob(raw_actions).sum(dim=-1) - th.log(jacobian).sum(dim=-1)
        
        # Get values
        values = self.value_net(latent_vf)
        
        # Get entropy
        # Ensure entropy calculation is robust to potential NaNs/Infs
        entropy = distribution.entropy().sum(dim=-1)
        
        # if th.isnan(entropy).any() or th.isinf(entropy).any():
        #     print("NaN/Inf detected in entropy!")
        #     import pdb; pdb.set_trace()
        # if th.isnan(log_prob).any() or th.isinf(log_prob).any():
        #     print("NaN/Inf detected in log_prob during evaluate_actions!")
        #     import pdb; pdb.set_trace()

        return values, log_prob, entropy

    def _inverse_transform_actions(self, actions: th.Tensor) -> th.Tensor:
        """Inverse transform actions with numerical stability."""
        # For variances: ensure they're > min_var_transform before inverse softplus
        variances_clamped = th.clamp(actions[..., :self.mlp_extractor.n_variances], min=self.mlp_extractor.min_var_transform)
        
        # Inverse of softplus(x) + min_var_transform is log(exp(y - min_var_transform) - 1)
        # Use torch.expm1 for numerical stability when x is close to 0
        raw_variances = th.log(th.expm1(variances_clamped - self.mlp_extractor.min_var_transform) + 1e-8) # Added 1e-8 for stability of log
        
        # For correlations: clamp to prevent atanh explosion
        correlations_clamped = th.clamp(actions[..., self.mlp_extractor.n_variances:], min=-0.9999999, max=0.9999999)
        raw_correlations = th.atanh(correlations_clamped)
        
        raw_actions = th.cat([raw_variances, raw_correlations], dim=-1)
        return raw_actions

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
                low=np.array([1e-6, 1e-6, 1e-6, -1.0, -1.0, -1.0]),  # Tighter bounds
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # Tighter bounds
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
            
            # Base survival reward (smaller to reduce sparsity)
            reward = 0.01 * (step + 1) / max_steps  # Even smaller survival reward
            
            # Position accuracy reward (more stable scaling)
            position_error_norm = np.linalg.norm(position_error)
            position_reward = np.exp(-5.0 * position_error_norm)  # Simpler reward
            
            # Orientation accuracy reward (more stable scaling)
            orientation_error_norm = np.linalg.norm(orientation_error)
            orientation_reward = np.exp(-5.0 * orientation_error_norm)  # Simpler reward
            
            # Combine rewards with better weighting
            total_reward = 0.1 * reward + 0.45 * position_reward + 0.45 * orientation_reward
            
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
            contact_prob = np.array([kin.contacts_probability[self.cf]], dtype=np.float32)
            
            return np.concatenate([local_pos - local_kin_pos, local_vel, contact_prob], axis=0).astype(np.float32)
        except Exception as e:
            print(f"Warning: State computation failed: {e}")
            return np.zeros(self.state_dim, dtype=np.float32)

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
            state = np.zeros(self.state_dim, dtype=np.float32)
        
        self.kin = kin
        return state

    def step(self, action):
        # Ensure action is the right shape
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Action dimension mismatch. Expected {self.action_dim}, got {action.shape[0]}")
        
        # Get the current measurements
        imu, _, _, gt = self._get_measurements(self.step_count)
        kin = self.kin
        post_state = self.serow_framework.get_state(allow_invalid=True)
        
        # Initialize tracking variables
        reward = 0.0
        state = np.zeros(self.state_dim, dtype=np.float32)
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

        # Fill in the info
        info = {
            "step": self.step_count, 
            "contact_active": kin.contacts_status[self.cf],
            "valid_step": valid_step,
            "invalid_reason": invalid_reason
        }

        # Update the step count
        self.step_count += 1
        
        # Check if we've reached the end of the episode
        if self.step_count >= 0.85 * self.max_steps and valid_step:
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
        while True:
           self.pre_step()
           state, _, _, _, info = self.step(np.zeros(self.action_dim))
           if info['valid_step']:
               break
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

    # Run the baseline policy
    eval_env = SerowEnv(
        robot=robot, 
        initial_state=initial_state, 
        contact_frame="RL_foot", 
        state_dim=state_dim, 
        action_dim=action_dim, 
        measurements=measurements)

    done = False
    truncated = False
    last_observations = []
    observations = []

    timestamps = []
    base_positions = []
    base_orientations = []
    gt_base_positions = []
    gt_base_orientations = []
    rewards = []
    infos = []
    while not done and not truncated:
        obs = eval_env.pre_step()
        last_observations.append(obs)
        imu, _, _, gt = eval_env._get_measurements(eval_env.step_count)
        action = np.zeros(action_dim)

        # Step the environment
        state, reward, done, truncated, info = eval_env.step(action)
        
        # Get the serow base state
        serow_state = eval_env.serow_framework.get_state(allow_invalid=True)
        base_positions.append(serow_state.get_base_position())
        base_orientations.append(serow_state.get_base_orientation())
        # Get the ground truth base state
        gt_base_positions.append(gt.position)
        gt_base_orientations.append(gt.orientation)
        timestamps.append(imu.timestamp)

        # Save the agent data
        observations.append(state)
        rewards.append(reward)
        infos.append(info)

    # Convert to numpy arrays
    base_positions = np.array(base_positions)
    base_orientations = np.array(base_orientations)
    gt_base_positions = np.array(gt_base_positions)
    gt_base_orientations = np.array(gt_base_orientations)
    timestamps = np.array(timestamps)

    # Based on info remove invalid steps
    last_observations = np.array(last_observations)
    observations = np.array(observations)
    rewards = np.array(rewards)
    valid_indices = [info['valid_step'] for info in infos]
    last_observations = last_observations[valid_indices]
    observations = observations[valid_indices]
    rewards = rewards[valid_indices]

    # Plot the trajectories
    plot_trajectories(timestamps, base_positions, base_orientations, gt_base_positions, gt_base_orientations)

    # plot the cummulative reward
    plt.figure(figsize=(15, 12))
    
    # First subplot: Instantaneous rewards
    plt.subplot(4, 1, 1)
    plt.plot(rewards, 'b-', linewidth=1)
    plt.xlabel('Step')
    plt.ylabel('Instantaneous Reward')
    plt.title('Instantaneous Rewards')
    plt.grid(True, alpha=0.3)
    
    # Second subplot: Cumulative rewards
    plt.subplot(4, 1, 2)
    plt.plot(np.cumsum(rewards), 'r-', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward')
    plt.grid(True, alpha=0.3)
    
    # Third subplot: Last observation components
    plt.subplot(4, 1, 3)
    # Plot relative position error (first 3 components)
    plt.plot(last_observations[:, 0], 'g-', label='rel_pos_x', linewidth=1)
    plt.plot(last_observations[:, 1], 'g--', label='rel_pos_y', linewidth=1)
    plt.plot(last_observations[:, 2], 'g:', label='rel_pos_z', linewidth=1)
    
    # Plot base velocity (next 3 components)
    plt.plot(last_observations[:, 3], 'b-', label='base_vel_x', linewidth=1)
    plt.plot(last_observations[:, 4], 'b--', label='base_vel_y', linewidth=1)
    plt.plot(last_observations[:, 5], 'b:', label='base_vel_z', linewidth=1)
    
    # Plot contact probability (last component)
    plt.plot(last_observations[:, 6], 'm-', label='contact_prob', linewidth=2)
    
    plt.xlabel('Step')
    plt.ylabel('Last Observation Values')
    plt.title('Last Observation Components: Relative Position Error, Base Velocity, Contact Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Third subplot: Observation components
    plt.subplot(4, 1, 4)
    # Plot relative position error (first 3 components)
    plt.plot(observations[:, 0], 'g-', label='rel_pos_x', linewidth=1)
    plt.plot(observations[:, 1], 'g--', label='rel_pos_y', linewidth=1)
    plt.plot(observations[:, 2], 'g:', label='rel_pos_z', linewidth=1)
    
    # Plot base velocity (next 3 components)
    plt.plot(observations[:, 3], 'b-', label='base_vel_x', linewidth=1)
    plt.plot(observations[:, 4], 'b--', label='base_vel_y', linewidth=1)
    plt.plot(observations[:, 5], 'b:', label='base_vel_z', linewidth=1)
    
    # Plot contact probability (last component)
    plt.plot(observations[:, 6], 'm-', label='contact_prob', linewidth=2)
    
    plt.xlabel('Step')
    plt.ylabel('Observation Values')
    plt.title('Observation Components: Relative Position Error, Base Velocity, Contact Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # print("Creating environments...")
    # # Create environments
    # env_fns = [
    #     lambda cf=cf: SerowEnv(
    #         robot=robot, 
    #         initial_state=initial_state, 
    #         contact_frame=cf, 
    #         state_dim=state_dim, 
    #         action_dim=action_dim, 
    #         measurements=measurements
    #     ) for cf in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    # ]
    # vec_env = SerowVecEnv(env_fns)
    
    # # Create the callback
    # invalid_sample_remover = InvalidSampleRemover(verbose=1)

    # # Instantiate the custom PPO with custom rollout collection
    # model = CustomPPO(
    #     CustomActorCriticPolicy, 
    #     vec_env, 
    #     device='cpu', 
    #     verbose=1,
    #     learning_rate=1e-4,
    #     n_steps=512,
    #     batch_size=64,
    #     n_epochs=4,
    #     gamma=1.0,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.0,
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     target_kl=0.01,
    #     normalize_advantage=True,
    #     clip_range_vf=0.02
    # )

    # model.set_invalid_sample_remover(invalid_sample_remover)

    # # Train the model with custom callbacks
    # callbacks = [invalid_sample_remover]
    # model.learn(total_timesteps=100000, callback=callbacks)
    # print("Training completed!")
    # model.save("serow_ppo_model")
