import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import random
from logger import Logger

from collections import deque
from utils import normalize_vector

class PPO:
    def __init__(self, actor, critic, params, device='cpu', normalize_state=False):
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
        self.update_lr = params.get('update_lr', False)
        self.final_lr_ratio = params.get('final_lr_ratio', 0.1)
        self.total_steps = params.get('total_steps', 1000000)
        self.current_step = 0
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.initial_actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.initial_critic_lr)
      
        self.buffer_size = params['buffer_size']
        self.buffer = deque(maxlen=self.buffer_size)
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.gae_lambda = params['gae_lambda']
        self.clip_param = params['clip_param']
        self.entropy_coef = params['entropy_coef']
        self.value_loss_coef = params['value_loss_coef']
        self.max_grad_norm = params['max_grad_norm']
        self.ppo_epochs = params['ppo_epochs']
        self.min_action = params['min_action']
        self.max_action = params['max_action']
        self.num_updates = 0
        self.training_step = 0
        self.timestep = 0

        self.normalize_state = normalize_state
        self.max_state_value = params['max_state_value']
        self.min_state_value = params['min_state_value']
        
        # Early stopping parameters
        self.value_loss_window_size = params.get('value_loss_window_size', 10) 
        self.reward_window_size = params.get('reward_window_size', 10000) 
        self.best_reward = float('-inf')
        self.convergence_threshold = params.get('convergence_threshold', 0.1)
        self.critic_convergence_threshold = params.get('critic_convergence_threshold', 0.05)
        self.best_model_state = None
        
        self.logger = Logger(smoothing_window=1000)

        # Checkpoint parameters
        self.checkpoint_dir = params['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def add_to_buffer(self, state, action, reward, next_state, done, value, log_prob):
        if self.normalize_state:
            state = normalize_vector(state.copy(), self.min_state_value, self.max_state_value)
            next_state = normalize_vector(next_state.copy(), self.min_state_value, self.max_state_value)

        # Convert to numpy and store in buffer
        state = np.array(state.flatten())
        action = np.array(action.flatten())
        next_state = np.array(next_state.flatten())

        experience = (state, action, reward, next_state, done, value, log_prob) 
        self.buffer.append(experience)
    
    def get_action(self, state, deterministic=False):
        if self.normalize_state:
            state = normalize_vector(state.copy(), self.min_state_value, self.max_state_value)

        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, deterministic)
            state_tensor = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            value = self.critic(state_tensor).item()
        return action, value, log_prob
    
    def compute_gae(self, rewards, values, dones):
        """
        Computes Generalized Advantage Estimation (GAE).
        Args:
            rewards: List of rewards collected in the trajectory.
                    Shape: (T,)
            values: List of value estimates for each state in the trajectory.
                    The last value estimate should be for the next_state
                    after the last reward. So, if rewards has length T,
                    values should have length T+1.
                    Shape: (T+1,)
            dones: List of flags indicating if a state was terminal.
                   (1.0 for done, 0.0 for not done).
                   Shape: (T,)
        Returns:
            tuple:
                - advantages: Computed GAE advantages for each state. Shape: (T,)
                - returns: Computed discounted returns (targets for the critic). Shape: (T,)
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        # Iterate backwards through the trajectory
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]

            # Calculate TD error
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # Calculate GAE
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]

        # Calculate returns as sum of advantages and values
        returns = advantages + values[:-1]

        return advantages, returns
    
    def update_learning_rates(self):
        """Update learning rates using linear decay."""
        progress = min(1.0, self.current_step / self.total_steps)
        
        # Linear decay from initial_lr to final_lr
        actor_lr = self.initial_actor_lr - (self.initial_actor_lr - self.initial_actor_lr * self.final_lr_ratio) * progress
        critic_lr = self.initial_critic_lr - (self.initial_critic_lr - self.initial_critic_lr * self.final_lr_ratio) * progress
        
        # Update optimizer learning rates
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr
        self.current_step += 1
        print(f"Actor LR: {actor_lr}, Critic LR: {critic_lr}")

    def save_checkpoint(self, avg_reward):
        # If this is the best model so far, save it separately
        """Save a checkpoint of the current model state"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'num_updates': self.num_updates,
            'best_reward': self.best_reward,
            'timestep': self.timestep,
            'training_step': self.training_step
        }
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_model_state = checkpoint
            best_model_path = os.path.join(self.checkpoint_dir, f'trained_policy_{self.robot}.pth')
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved with averag reward: {avg_reward:.2f}")
        checkpoint_path = os.path.join(self.checkpoint_dir, f'policy_checkpoint_{self.robot}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint with reward: {avg_reward:.2f}")

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.num_updates = checkpoint['num_updates']
        self.best_reward = checkpoint['best_reward']
        self.timestep = checkpoint['timestep']
        self.training_step = checkpoint['training_step']
        print(f"Loaded checkpoint with reward: {self.best_reward:.2f}")

    def check_early_stopping(self):
        converged = False
        if len(self.logger.rewards) < self.reward_window_size or \
            len(self.logger.value_losses) < self.value_loss_window_size:
            return False
        
        # Get recent rewards and calculate statistics
        recent_rewards = np.array(self.logger.rewards[-self.reward_window_size:])
        avg_reward = np.mean(recent_rewards)
        
        # Save checkpoint with current average reward
        self.save_checkpoint(avg_reward)

        # Check if value function has converged
        recent_value_losses = np.array(self.logger.value_losses[-self.value_loss_window_size:])
        value_loss_std = np.std(recent_value_losses)
        value_loss_mean = np.mean(recent_value_losses)
        value_loss_converged = value_loss_std <= \
            (value_loss_mean * self.critic_convergence_threshold)
        
        # Only consider early stopping if value function has converged
        # and we've seen enough samples
        if value_loss_converged and len(self.logger.rewards) >= self.reward_window_size:
            # Calculate reward trend using linear regression
            x = np.arange(len(recent_rewards))
            slope, _ = np.polyfit(x, recent_rewards, 1)
            
            # Check if rewards have plateaued
            # We use a more lenient threshold for the slope
            rewards_plateaued = abs(float(slope)) < self.convergence_threshold * abs(avg_reward)
            
            if rewards_plateaued:
                print(f"Training may have converged:")
                print(f"  - Average reward: {avg_reward:.2f}")
                print(f"  - Reward trend slope: {float(slope):.4f}")
                print(f"  - Value loss std/mean ratio: {value_loss_std/value_loss_mean:.4f}")
                print(f"  - Consider stopping if performance is satisfactory")
                converged = True
                
        return converged
    
    def train(self):
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0, 0.0

        # Update learning rates before training
        if self.update_lr:
            self.update_learning_rates()  

        self.training_step += 1

        # Convert buffer to tensors
        states, actions, rewards, next_states, dones, values, log_probs = zip(*self.buffer)
        
        # Convert to numpy arrays with consistent shapes
        states = np.array([np.array(s).flatten() for s in states])
        next_states = np.array([np.array(s).flatten() for s in next_states])
        actions = np.array([np.array(a).flatten() for a in actions])
        rewards = np.array(rewards).reshape(-1, 1)
        dones = np.array(dones).reshape(-1, 1)
        values = np.array(values).reshape(-1, 1)
        old_log_probs = np.array(log_probs).reshape(-1, 1)

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
            next_values = self.critic(next_states[-1].unsqueeze(0))
        
        # Compute GAE and returns
        advantages, returns = self.compute_gae(rewards, torch.cat([values, next_values]), dones)

        for i in range(len(rewards)):
            self.logger.log_step(self.timestep, rewards[i], values[i], advantages[i])
            self.timestep += 1

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        dataset_size = len(states)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Get current policy outputs using the new evaluate_actions method
                current_log_probs, entropy = self.actor.evaluate_actions(batch_states, batch_actions)
                
                # Calculate policy loss with clipping
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                current_values = self.critic(batch_states)
                value_loss = F.mse_loss(current_values, batch_returns)
                
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
        
        # Clear buffer after training
        self.buffer.clear()
        
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        # Log log_std mean if available
        log_std_mean = None
        if hasattr(self.actor, 'log_std'):
            log_std_mean = self.actor.log_std.data.mean().item()
        self.logger.log_training_step(self.timestep, avg_policy_loss, avg_value_loss, avg_entropy, log_std=log_std_mean)

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
