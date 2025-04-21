import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

class PPO:
    def __init__(self, actor, critic, state_dim, action_dim, max_action, min_action, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.buffer = deque(maxlen=1000000)
        self.batch_size = 128
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_param = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.ppo_epochs = 5
        self.min_action = min_action
        self.max_action = max_action
        
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def add_to_buffer(self, state, action, reward, next_state, done, value, log_prob):
        self.buffer.append((state, action, reward, next_state, done, value, log_prob))
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        with torch.no_grad():
            # Assuming actor outputs mean and log_std
            action_mean, action_log_std = self.actor(state)
            action_std = torch.exp(action_log_std)
            
            if deterministic:
                action = action_mean
                log_prob = torch.tensor(0.0).to(self.device)
            else:
                # Create a normal distribution and sample
                normal = torch.distributions.Normal(action_mean, action_std)
                action = normal.sample()
                # Clip action to ensure it stays within bounds
                action = torch.clamp(action, self.min_action, self.max_action)
                log_prob = self.calculate_log_probs(action, action_mean, action_std)
            
            # Get value estimate
            value = self.critic(state)

        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0]
    
    def calculate_log_probs(self, action, mean, std):
        # Calculate log probability of action under the Gaussian policy
        normal = torch.distributions.Normal(mean, std)
        log_probs = normal.log_prob(action).sum(dim=-1, keepdim=True)
        return log_probs
    
    def compute_gae(self, rewards, values, next_values, dones):
        # Calculate Generalized Advantage Estimation
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + values
        return advantages, returns
    
    def train(self):
        if len(self.buffer) == 0:
            return
        
        # Convert buffer to tensors
        states, actions, rewards, next_states, dones, values, old_log_probs = zip(*self.buffer)
        
        # Convert states and next_states to numpy arrays with consistent shapes
        states = np.array([np.array(s).flatten() for s in states])
        next_states = np.array([np.array(s).flatten() for s in next_states])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        old_values = torch.FloatTensor(np.array(values)).unsqueeze(1).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).unsqueeze(1).to(self.device)
        
        # Compute next state values
        with torch.no_grad():
            next_values = self.critic(next_states)
        
        # Compute GAE and returns
        advantages, returns = self.compute_gae(rewards, old_values, next_values, dones)
        if advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create mini-batches
        indices = np.arange(len(self.buffer))
        np.random.shuffle(indices)
        for start in range(0, len(self.buffer), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            
            # PPO update loop
            for _ in range(self.ppo_epochs):
                # Actor update
                action_means, action_log_stds = self.actor(batch_states)
                action_stds = torch.exp(action_log_stds)
                current_log_probs = self.calculate_log_probs(batch_actions, action_means, action_stds)
                
                # Calculate entropy for exploration
                entropy = torch.distributions.Normal(action_means, action_stds).entropy().mean()
                
                # Compute policy loss with clipping
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Actor loss
                actor_loss = policy_loss - self.entropy_coef * entropy
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Critic update
                current_values = self.critic(batch_states)
                value_loss = self.value_loss_coef * F.mse_loss(current_values, batch_returns)
                
                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
        
        # Clear buffer after processing all data
        self.buffer.clear()
    
    def save_models(self, path):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")
    
    def load_models(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))
