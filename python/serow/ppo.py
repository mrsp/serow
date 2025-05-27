import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import deque
from utils import normalize_vector

class PPO:
    def __init__(self, actor, critic, params, device='cpu', normalize_state=False):
        self.buffer_size = params['buffer_size']
        self.buffer = deque(maxlen=self.buffer_size)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.name = "PPO"
        
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

        self.normalize_state = normalize_state
        self.max_state_value = params['max_state_value']
        self.min_state_value = params['min_state_value']


    def add_to_buffer(self, state, action, reward, next_state, done, value, log_prob):
        if self.normalize_state:
            state = normalize_vector(state.copy(), self.min_state_value, self.max_state_value)
            next_state = normalize_vector(next_state.copy(), self.min_state_value, self.max_state_value)

        # Convert to numpy and store in buffer
        state = np.array(state.flatten())
        action = np.array(action.flatten())
        reward = float(reward)
        next_state = np.array(next_state.flatten())
        done = float(done)
        value = float(value)
        log_prob = float(log_prob)

        # Store experience
        experience = (state, action, reward, next_state, done, value, log_prob)
        self.buffer.append(experience)
    
    def get_action(self, state, deterministic=False):
        if self.normalize_state:
            state = normalize_vector(state.copy(), self.min_state_value, self.max_state_value)

        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, deterministic)
            value = self.critic(state)
        return action, value, log_prob
    
    def calculate_log_probs(self, action, mean, std):
        # Calculate log probability of action under the Gaussian policy
        normal = torch.distributions.Normal(mean, std)
        log_probs = normal.log_prob(action)
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
        states, actions, rewards, next_states, dones, values, log_probs = zip(*self.buffer)
        
        # Convert states and next_states to numpy arrays with consistent shapes
        states = np.array([np.array(s).flatten() for s in states])
        next_states = np.array([np.array(s).flatten() for s in next_states])
        actions = np.array([np.array(a).flatten() for a in actions])
        rewards = np.array(rewards).reshape(-1, 1)
        dones = np.array(dones).reshape(-1, 1)
        values = np.array(values).reshape(-1, 1)
        log_probs = np.array(log_probs).reshape(-1, 1)


        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)

        # Compute next state values
        with torch.no_grad():
            next_values = self.critic(next_states)
        
        # Compute GAE and returns
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

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
                batch_log_probs = log_probs[batch_indices]
                
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
                ratio = torch.exp(current_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss with clipping
                current_values = self.critic(batch_states)
                value_pred_clipped = values[batch_indices] + torch.clamp(
                    current_values - values[batch_indices],
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
        return float(policy_loss.item()), float(value_loss.item())

    def save_models(self, path):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")
    
    def load_models(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))
