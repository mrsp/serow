import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import copy

from collections import deque
from utils import normalize_vector

class DDPG:
    def __init__(self, actor, critic, params, device='cpu', normalize_state=False):
        self.name = "DDPG"
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.actor_target = copy.deepcopy(actor).to(self.device)
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), 
            lr=params['actor_lr'],
            weight_decay=1e-5,  # Small weight decay for regularization
            eps=1e-5  # Improved numerical stability
        )

        self.critic = critic.to(self.device)
        self.critic_target = copy.deepcopy(critic).to(self.device)
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(), 
            lr=params['critic_lr'],
            weight_decay=1e-5,
            eps=1e-5
        )

        # Prioritized experience replay
        self.buffer = deque(maxlen=params['buffer_size'])
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch_size']
        self.train_for_batches = params['train_for_batches']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.min_action = params['min_action']
        self.max_action = params['max_action']

        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.normalize_state = normalize_state
        self.max_state_value = params['max_state_value']
        self.min_state_value = params['min_state_value']
        
        # Adaptive parameters
        self.grad_clip_value = 1.0

    def add_to_buffer(self, state, action, reward, next_state, done):
        if self.normalize_state:
            state = normalize_vector(state.copy(), self.min_state_value, self.max_state_value)
            next_state = normalize_vector(next_state.copy(), self.min_state_value, self.max_state_value)
        
        # Convert to numpy and store in buffer
        state = np.array(state.flatten())
        action = np.array(action.flatten())
        reward = float(reward)
        next_state = np.array(next_state.flatten())
        done = float(done)
        
        # Store experience
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def get_action(self, state, deterministic=False):
        if self.normalize_state:
            state = normalize_vector(state.copy(), self.min_state_value, self.max_state_value)
        
        return self.actor.get_action(state, deterministic)
            
    def train(self):
        # Warm-up phase
        if len(self.buffer) < 5 * self.batch_size: 
            return None, None
        
        # Sample multiple batches and stack them
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        
        for _ in range(self.train_for_batches):
            batch = random.sample(self.buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            all_states.append(np.vstack(states))
            all_actions.append(np.vstack(actions))
            all_rewards.append(np.array(rewards).reshape(-1, 1))
            all_next_states.append(np.vstack(next_states))
            all_dones.append(np.array(dones).reshape(-1, 1))
        
        # Stack all batches
        states = torch.FloatTensor(np.vstack(all_states)).to(self.device)
        actions = torch.FloatTensor(np.vstack(all_actions)).to(self.device)
        rewards = torch.FloatTensor(np.vstack(all_rewards)).to(self.device)
        next_states = torch.FloatTensor(np.vstack(all_next_states)).to(self.device)
        dones = torch.FloatTensor(np.vstack(all_dones)).to(self.device)
        
        # Critic update with target network
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * next_Q
            
        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_value)
        self.critic_optimizer.step()
        
        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()  
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_value)
        self.actor_optimizer.step()
        
        # Soft update target networks using in-place operations
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
                    
        return float(critic_loss.item()), float(actor_loss.item())
