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

        # Pre-allocate buffer tensors for better memory efficiency
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

        # Pre-allocate tensors for batch processing
        self.states = torch.zeros((self.batch_size, self.state_dim), device=self.device)
        self.actions = torch.zeros((self.batch_size, self.action_dim), device=self.device)
        self.rewards = torch.zeros((self.batch_size, 1), device=self.device)
        self.next_states = torch.zeros((self.batch_size, self.state_dim), device=self.device)
        self.dones = torch.zeros((self.batch_size, 1), device=self.device)

    def add_to_buffer(self, state, action, reward, next_state, done):
        if self.normalize_state:
            state = normalize_vector(state.copy(), self.min_state_value, self.max_state_value)
            next_state = normalize_vector(next_state.copy(), self.min_state_value, self.max_state_value)
        
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
                state = normalize_vector(state.copy(), self.min_state_value, self.max_state_value)
            
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.get_action(state, deterministic) 
            return action
            
    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample from buffer and stack tensors efficiently
        batch = random.sample(self.buffer, self.batch_size)
        
        # Efficiently stack tensors using pre-allocated memory
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            self.states[i] = state
            self.actions[i] = action
            self.rewards[i] = reward
            self.next_states[i] = next_state
            self.dones[i] = done
        
        # Critic update with target network
        with torch.no_grad():
            next_actions = self.actor_target(self.next_states)
            target_Q = self.critic_target(self.next_states, next_actions)
            target_Q = self.rewards + (1.0 - self.dones) * self.gamma * target_Q
            
        current_Q = self.critic(self.states, self.actions)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # Optimize critic with gradient clipping
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(self.states, self.actor(self.states)).mean()
        
        # Optimize actor with gradient clipping
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks using in-place operations
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
