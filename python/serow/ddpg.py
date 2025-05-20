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
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params['actor_lr'])
        self.name = "DDPG"

        self.critic = critic.to(self.device)
        self.critic_target = copy.deepcopy(critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params['critic_lr'])

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.buffer = deque(maxlen=params['buffer_size'])
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.min_action = params['min_action']
        self.max_action = params['max_action']

        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.normalize_state = normalize_state
        self.max_state_value = params['max_state_value']
        self.min_state_value = params['min_state_value']

    def add_to_buffer(self, state, action, reward, next_state, done):
        # Check if the buffer is full
        if len(self.buffer) == self.buffer_size:
            print(f"Buffer is full. Removing oldest sample.")
            self.buffer.popleft()
        
        if self.normalize_state:
            state = state.copy()
            next_state = next_state.copy()
            state = normalize_vector(state, self.min_state_value, self.max_state_value)
            next_state = normalize_vector(next_state, self.min_state_value, self.max_state_value)
        
        # Convert to tensors and store in buffer
        state = torch.FloatTensor(state.flatten()).to(self.device)
        action = torch.FloatTensor(action.flatten()).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        next_state = torch.FloatTensor(next_state.flatten()).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        self.buffer.append((state, action, reward, next_state, done))

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            if self.normalize_state:
                state = state.copy()
                state = normalize_vector(state, self.min_state_value, self.max_state_value)
            
            action = self.actor.get_action(state, deterministic) 
            return action
            
    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample from buffer - data is already in tensor format
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Stack tensors along batch dimension
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1.0 - dones) * self.gamma * target_Q
            
        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks - using in-place operations
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
