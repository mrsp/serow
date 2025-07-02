import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import random
from logger import Logger

from collections import deque

class PPO:
    def __init__(self, actor, critic, params, device='cpu'):
        self.name = "PPO"
        self.robot = params['robot']
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        
        # Learning rate parameters
        self.initial_actor_lr = params['actor_lr']
        self.initial_critic_lr = params['critic_lr']
        self.initial_entropy_coef = params['entropy_coef']
        self.final_lr_ratio = params.get('final_lr_ratio', 1.0)
        self.total_steps = params.get('total_steps', 1000000)
        self.total_training_steps = params.get('total_training_steps', 1000000)
        self.target_kl = params.get('target_kl', None)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.initial_actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.initial_critic_lr)
      
        self.buffer_size = params['buffer_size']
        self.buffer = deque(maxlen=self.buffer_size)
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.gae_lambda = params['gae_lambda']
        self.clip_param = params['clip_param']
        self.entropy_coef = self.initial_entropy_coef
        self.value_loss_coef = params['value_loss_coef']
        self.max_grad_norm = params['max_grad_norm']
        self.ppo_epochs = params['ppo_epochs']
        self.min_action = params.get('min_action', -1e2 * torch.ones(self.action_dim))
        self.max_action = params.get('max_action', 1e2 * torch.ones(self.action_dim))
        self.num_updates = 0
        self.training_step = 0
        self.samples = 0
        
        # Early stopping parameters
        self.value_loss_window_size = params.get('value_loss_window_size', 10) 
        self.returns_window_size = params.get('returns_window_size', 20) 
        self.best_return = float('-inf')
        self.convergence_threshold = params.get('convergence_threshold', 0.1)
        self.critic_convergence_threshold = params.get('critic_convergence_threshold', 0.05)
        self.best_model_state = None
        
        self.logger = Logger(smoothing_window=1000)

        # Checkpoint parameters
        self.checkpoint_dir = params['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Define a separate clipping parameter for the value function
        self.value_clip_param = params.get('value_clip_param', None)  
        self.check_value_loss = params.get('check_value_loss', False)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def add_to_buffer(self, state, action, reward, next_state, done, value, log_prob):
        experience = (state, action, reward, next_state, done, value, log_prob) 
        self.buffer.append(experience)
    
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, deterministic)
            state_tensor = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            value = self.critic(state_tensor).squeeze(0).detach().cpu().item()
        return action, value, log_prob
    

    def compute_gae(self, rewards, values, dones, next_values):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: rewards for each step [T]
            values: value estimates for each step [T] 
            dones: done flags for each step [T] (dones[i] == 1 if next_state[i] is terminal)
            next_values: value estimates for next states [T]
        """
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        last_gae_lam = 0
        
        for step in reversed(range(len(rewards))):
            # dones[step] indicates if next_states[step] (which is S_{t+1} for current step t) is terminal
            next_non_terminal = 1.0 - dones[step]
            # If the episode is done, next state is terminal, so V(s_{t+1}) = 0
            # Otherwise, use the value estimate of the next state
            next_value = next_values[step] * next_non_terminal
            
            # TD error: r_t + γ * V(s_{t+1}) * (1-done_{t+1}) - V(s_t)
            # done_{t+1} is dones[step]
            delta = rewards[step] + self.gamma * next_value - values[step]
            
            # GAE: A_t = δ_t + γλ * (1-done_{t+1}) * A_{t+1}
            advantages[step] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values
        return advantages, returns

    def update_learning_params(self):
        """Update learning rates and entropy coefficient using linear decay."""
        progress = min(1.0, self.training_step / self.total_training_steps)

        # Linear decay from initial_lr to final_lr
        actor_lr = self.initial_actor_lr - (self.initial_actor_lr - self.initial_actor_lr * self.final_lr_ratio) * progress
        critic_lr = self.initial_critic_lr - (self.initial_critic_lr - self.initial_critic_lr * self.final_lr_ratio) * progress
        # Decay to 10% of initial value at the end
        self.entropy_coef = self.initial_entropy_coef * (1.0 - 0.9 * progress)
        self.entropy_coef = max(self.entropy_coef, 0.001)  # Keep some minimum entropy

        # Update optimizer learning rates
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr

    def save_checkpoint(self, avg_return):
        # If this is the best model so far, save it separately
        """Save a checkpoint of the current model state"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'num_updates': self.num_updates,
            'best_return': self.best_return,
            'samples': self.samples,
            'training_step': self.training_step
        }
        if avg_return > self.best_return:
            self.best_return = avg_return
            self.best_model_state = checkpoint
            best_model_path = os.path.join(self.checkpoint_dir, f'trained_policy_{self.robot}.pth')
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved with averag return: {avg_return}")
        checkpoint_path = os.path.join(self.checkpoint_dir, f'policy_checkpoint_{self.robot}.pth')
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.num_updates = checkpoint['num_updates']
        self.best_return = checkpoint['best_return']
        self.samples = checkpoint['samples']
        self.training_step = checkpoint['training_step']
        print(f"Loaded checkpoint with return: {self.best_return}")

    def check_early_stopping(self):
        # Save checkpoint with current average return
        if len(self.logger.returns) > 0: 
            self.save_checkpoint(self.logger.returns[-1])

        converged = False
        if len(self.logger.returns) < self.returns_window_size or \
            len(self.logger.value_losses) < self.value_loss_window_size:
            return converged

        # Check if recent returns have converged to the best return gathered so far
        recent_returns = np.array(self.logger.returns[-self.returns_window_size:])
        
        # Check if all recent returns are within convergence threshold of best_return
        threshold = max(abs(self.best_return) * self.convergence_threshold, 0.01)  # minimum 0.01
        recent_return_converged = np.all(np.abs(recent_returns - self.best_return) <= threshold)

        # Check if value function has converged
        value_loss_converged = False
        if self.check_value_loss:
            recent_value_losses = np.array(self.logger.value_losses[-self.value_loss_window_size:])
            value_loss_std = np.std(recent_value_losses)
            value_loss_mean = np.mean(recent_value_losses)
            value_loss_converged = value_loss_std <= \
                (value_loss_mean * self.critic_convergence_threshold)
        else:
            value_loss_converged = True

        # Only consider early stopping if value function has converged and returns have converged
        if value_loss_converged and recent_return_converged:
           converged = True
                
        return converged
    
    def compute_kl_divergence(self, states, actions, old_log_probs):
        """Compute KL divergence between old and new policy"""
        with torch.no_grad():
            new_log_probs, _ = self.actor.evaluate_actions(states, actions)
            kl_div = (old_log_probs - new_log_probs).mean()
        return kl_div.item()

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0, 0.0, False

        # Update learning parameters before training
        self.update_learning_params()  

        # Convert buffer to tensors
        states, actions, rewards, next_states, dones, values, log_probs = zip(*self.buffer)
        
        # Convert to numpy arrays with consistent shapes
        states = np.array([np.array(s).flatten() for s in states])
        next_states = np.array([np.array(s).flatten() for s in next_states])
        actions = np.array([np.array(a).flatten() for a in actions])
        rewards = np.array(rewards).flatten()
        dones = np.array(dones).flatten()
        values = np.array(values).flatten()
        old_log_probs = np.array(log_probs).flatten()

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        # Compute next state values
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(-1)  # Keep all values for GAE
        
        # Compute GAE and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)
        
        # Log data
        for i in range(len(rewards)):
            self.logger.log_step(self.samples, rewards[i], values[i], advantages[i])
            self.samples += 1

        # Normalize advantages using running statistics
        # Use a small epsilon for numerical stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        dataset_size = len(states)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        early_stopped = False
        for epoch in range(self.ppo_epochs):
            if early_stopped:
                break

            # Shuffle data
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                if (batch_indices.shape[0] != self.batch_size):
                    continue

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_values = values[batch_indices]

                # Get current policy outputs using the new evaluate_actions method
                current_log_probs, entropy = self.actor.evaluate_actions(batch_states, batch_actions)

                # Calculate policy loss with clipping
                log_ratio = current_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = -ratio * batch_advantages
                surr2 = -torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = torch.max(surr1, surr2).mean()

                # Calculate value loss with normalized returns and value clipping
                current_values = self.critic(batch_states)

                # Value clipping in normalized space
                if self.value_clip_param is not None:
                    value_pred_clipped = batch_values + torch.clamp(
                            current_values - batch_values,
                            -self.value_clip_param, self.value_clip_param
                        )
                    value_loss_unclipped = (current_values - batch_returns) ** 2
                    value_loss_clipped = (value_pred_clipped - batch_returns) ** 2
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                else:
                    value_loss = (current_values - batch_returns) ** 2
                value_loss = 0.5 * value_loss.mean()

                # Calculate total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                # Optimization step
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

            # Check KL divergence after each epoch for early stopping
            if self.target_kl is not None:
                kl_div = self.compute_kl_divergence(states, actions, old_log_probs)
                
                if kl_div > self.target_kl:
                    print(f"Early stopping at epoch {epoch + 1}/{self.ppo_epochs}, KL divergence: {kl_div} > target: {self.target_kl}")
                    early_stopped = True

        # Clear buffer after training
        self.buffer.clear()
        
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        # Log log_std mean if available
        log_std_mean = None
        if hasattr(self.actor, 'log_std'):
            log_std_mean = self.actor.log_std.data.mean().item()
        self.logger.log_training_step(self.training_step, avg_policy_loss, avg_value_loss, avg_entropy, log_std=log_std_mean)
        self.training_step += 1

        converged = self.check_early_stopping()

        return avg_policy_loss, avg_value_loss, avg_entropy, converged

    def save_models(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, f"{path}_ppo.pth")
    
    def load_models(self, path):
        checkpoint = torch.load(f"{path}_ppo.pth")
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
