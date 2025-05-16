import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import copy

class DDPG:
    def __init__(self, actor, critic, params, device='cpu'):
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

    def add_to_buffer(self, state, action, reward, next_state, done):
        # Check if the buffer is full
        if len(self.buffer) == self.buffer_size:
            print(f"Buffer is full. Removing oldest sample.")
            self.buffer.popleft()
        self.buffer.append((state, action, reward, next_state, done))

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            return self.actor.get_action(state, deterministic)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert states and next_states to numpy arrays with consistent shapes
        states = np.array([np.array(s).flatten() for s in states])
        next_states = np.array([np.array(s).flatten() for s in next_states])
        actions = np.array([np.array(a).flatten() for a in actions])

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # Ensure dones are scalar values
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1.0 - dones) * self.gamma * target_Q
        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
