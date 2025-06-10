import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import random

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
        self.batch_size = params['batch_size']
      
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

        self.normalize_state = normalize_state
        self.max_state_value = params['max_state_value']
        self.min_state_value = params['min_state_value']
        
        # Early stopping parameters
        self.window_size = params.get('window_size', 10) 
        self.best_reward = float('-inf')
        self.convergence_threshold = params.get('convergence_threshold', 0.1)
        self.critic_convergence_threshold = params.get('critic_convergence_threshold', 0.05)
        self.reward_history = []
        self.critic_loss_history = []
        self.best_model_state = None
        
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
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Calculate Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns
    
    def update_learning_rates(self):
        """Update learning rates using cosine annealing."""
        progress = min(1.0, self.current_step / self.total_steps)
        
        # Cosine annealing
        actor_lr = self.initial_actor_lr * self.final_lr_ratio + \
                  0.5 * (self.initial_actor_lr - self.initial_actor_lr * self.final_lr_ratio) * \
                  (1 + np.cos(np.pi * progress))
        
        critic_lr = self.initial_critic_lr * self.final_lr_ratio + \
                   0.5 * (self.initial_critic_lr - self.initial_critic_lr * self.final_lr_ratio) * \
                   (1 + np.cos(np.pi * progress))
        
        # Update optimizer learning rates
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr
            
        print(f"Actor LR: {actor_lr}, Critic LR: {critic_lr}")
        self.current_step += 1

    def save_checkpoint(self, episode_reward):
        # If this is the best model so far, save it separately
        """Save a checkpoint of the current model state"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode_reward': episode_reward,
            'num_updates': self.num_updates
        }
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_model_state = checkpoint
            best_model_path = os.path.join(self.checkpoint_dir, f'trained_policy_{self.robot}.pth')
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved with reward: {episode_reward:.2f}")
        checkpoint_path = os.path.join(self.checkpoint_dir, f'policy_checkpoint_{self.robot}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint with reward: {episode_reward:.2f}")

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.num_updates = checkpoint['num_updates']
        self.best_reward = checkpoint['episode_reward']
        print(f"Loaded checkpoint with reward: {self.best_reward:.2f}")

    def check_early_stopping(self, episode_reward, critic_loss):
        converged = False
        self.reward_history.append(episode_reward)
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)

        recent_rewards = np.array(self.reward_history)
        # Calculate how close recent rewards are to best reward
        rewards_ratios = recent_rewards / (abs(self.best_reward) + 1e-6)  # Avoid division by zero
        
        # Check if rewards have converged
        rewards_converged = np.all(rewards_ratios >= (1.0 - self.convergence_threshold))

        self.critic_loss_history.append(critic_loss)
        if len(self.critic_loss_history) > self.window_size:
            self.critic_loss_history.pop(0)

        # Check if critic loss has converged
        critic_loss_converged = False
        if len(self.critic_loss_history) >= self.window_size:
            recent_critic_losses = np.array(self.critic_loss_history)
            critic_loss_std = np.std(recent_critic_losses)
            critic_loss_mean = np.mean(recent_critic_losses)
            critic_loss_converged = critic_loss_std <= \
                (critic_loss_mean * self.critic_convergence_threshold)
                
        # Both rewards and critic loss must be converged
        if rewards_converged and critic_loss_converged:
            print(f"Training converged!")
            print(f"Recent rewards are within {self.convergence_threshold*100}% of best episode "  
                  f"reward {self.best_reward:.2f}")
            print(f"Critic loss has stabilized with std/mean ratio: "
                  f"{critic_loss_std/critic_loss_mean:.4f}")
            converged = True
        return converged
    
    def train(self):
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0, 0.0

        # Update learning rates before training
        if self.update_lr:
            self.update_learning_rates()  

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
            next_values = self.critic(next_states)
        
        # Compute GAE and returns
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

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
        
        return (total_policy_loss / num_updates, 
                total_value_loss / num_updates, 
                total_entropy / num_updates)

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
