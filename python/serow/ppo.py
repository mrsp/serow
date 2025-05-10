import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

class PPO:
    def __init__(self, actor, critic, params, device='cpu'):
        self.buffer_size = params['buffer_size']
        self.buffer = deque(maxlen=self.buffer_size)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params['critic_lr'])
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
    
    def add_to_buffer(self, state, action, reward, next_state, done, value, log_prob):
        # Check if the buffer is full
        if len(self.buffer) == self.buffer_size:
            print(f"Buffer is full. Removing oldest sample.")
            self.buffer.popleft()
        self.buffer.append((state, action, reward, next_state, done, value, log_prob))
    
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, deterministic)
            value = self.critic(state)
        return (
            action.squeeze(0).detach().cpu().numpy(),  # Flatten single-batch action
            value.item(),                              # Extract scalar value
            log_prob.item()                            # Extract scalar log_prob
        )

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
                
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns
    
    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        # Convert buffer to tensors
        states, actions, rewards, next_states, dones, values, old_log_probs = zip(*self.buffer)
        
        # Convert states and next_states to numpy arrays with consistent shapes
        states = np.array([np.array(s).flatten() for s in states])
        next_states = np.array([np.array(s).flatten() for s in next_states])
        actions = np.array([np.array(a).flatten() for a in actions])
        rewards = np.array(rewards)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        old_values = torch.FloatTensor(np.array(values)).unsqueeze(1).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).unsqueeze(1).to(self.device)
        
        # Compute next state values
        with torch.no_grad():
            next_values = self.critic(next_states)
        
        # Compute GAE and returns
        advantages, returns = self.compute_gae(rewards, old_values, next_values, dones)

        # Standardize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Loop structure: outer = epochs, inner = batches
        dataset_size = len(states)
        for _ in range(self.ppo_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Get current policy outputs
                action_means, action_log_stds = self.actor(batch_states)
                action_stds = torch.exp(action_log_stds)
                current_log_probs = self.calculate_log_probs(batch_actions, action_means, action_stds)
                
                # Calculate entropy for exploration
                if (self.entropy_coef > 0):
                    entropy = torch.distributions.Normal(action_means, action_stds).entropy().mean()
                else:
                    entropy = 0 

                # Calculate policy loss with clipping
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss with clipping
                current_values = self.critic(batch_states)
                value_pred_clipped = old_values[batch_indices] + torch.clamp(
                    current_values - old_values[batch_indices],
                    -self.clip_param,
                    self.clip_param
                )
                value_loss = torch.max(
                    F.mse_loss(current_values, batch_returns),
                    F.mse_loss(value_pred_clipped, batch_returns)
                )

                # Combine losses
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Single backward pass and optimization step
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients for both networks
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                # Single optimization step
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.buffer.clear()
    
    def save_models(self, path):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")
    
    def load_models(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))
