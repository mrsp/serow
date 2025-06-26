import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
import os
from logger import Logger

from collections import deque
from utils import normalize_vector

class DDPG:
    def __init__(self, actor, critic, params, device='cpu', normalize_state=False):
        self.name = "DDPG"
        self.robot = params['robot']
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
    
        # Learning rate parameters
        self.initial_actor_lr = params['actor_lr']
        self.initial_critic_lr = params['critic_lr']
        self.final_lr_ratio = params.get('final_lr_ratio', 0.1)
        self.total_steps = params.get('total_steps', 1000000)
        self.total_training_steps = params.get('total_training_steps', 1000000)
        self.samples = 0
        self.training_step = 0

        self.critic_target = copy.deepcopy(critic).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params['critic_lr'])
        self.actor_target = copy.deepcopy(actor).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params['actor_lr'])


        # Prioritized experience replay
        self.buffer = deque(maxlen=params['buffer_size'])
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch_size']
        self.train_for_batches = params['train_for_batches']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.min_action = params['min_action']
        self.max_action = params['max_action']
        self.num_updates = 0
        self.max_grad_norm = params['max_grad_norm']

        self.normalize_state = normalize_state
        self.max_state_value = params['max_state_value']
        self.min_state_value = params['min_state_value']
        
        # Early stopping parameters
        self.check_value_loss = params.get('check_value_loss', False)
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
    
    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

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
    
    def update_learning_params(self):
        """Update learning rates using linear decay."""
        progress = min(1.0, self.training_step / self.total_training_steps)

        # Linear decay from initial_lr to final_lr
        actor_lr = self.initial_actor_lr - (self.initial_actor_lr - self.initial_actor_lr * self.final_lr_ratio) * progress
        critic_lr = self.initial_critic_lr - (self.initial_critic_lr - self.initial_critic_lr * self.final_lr_ratio) * progress

        # Update optimizer learning rates
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr

    def add_to_buffer(self, state, action, reward, next_state, done):
        if self.normalize_state:
           state = normalize_vector(state.copy(), self.min_state_value, self.max_state_value)
           next_state = normalize_vector(next_state.copy(), self.min_state_value, self.max_state_value)

        # Store experience
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.logger.log_step(self.samples, reward, 
                             self.critic(torch.FloatTensor(state).to(self.device), 
                                         torch.FloatTensor(action).to(self.device)).item(), None)
        self.samples += 1

    def get_action(self, state, deterministic=False):
        if self.normalize_state:
            state = normalize_vector(state.copy(), self.min_state_value, self.max_state_value)
        
        return self.actor.get_action(state, deterministic)
            
    def learn(self):
        # Warm-up phase
        if len(self.buffer) < self.train_for_batches * self.batch_size: 
            return 0.0, 0.0, False
        
        # Update learning rates before training
        self.update_learning_params()  

        # OPTION 1: Single large batch (recommended)
        total_batch_size = self.batch_size * self.train_for_batches
        batch = random.sample(self.buffer, total_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays with proper flattening
        states = np.array([np.array(s).flatten() for s in states])
        actions = np.array([np.array(a).flatten() for a in actions])
        rewards = np.array(rewards)
        next_states = np.array([np.array(s).flatten() for s in next_states])
        dones = np.array(dones)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Critic update with target network
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_actions = torch.clamp(next_actions, self.min_action, self.max_action)
            next_Q = self.critic_target(next_states, next_actions).squeeze(-1)
            target_Q = rewards + (1 - dones) * self.gamma * next_Q
            
        current_Q = self.critic(states, actions).squeeze(-1)
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Optimize actor
        actor_loss = -self.critic(states, self.actor(states)).mean()  
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Soft update target networks using in-place operations
        with torch.no_grad():
            # Update parameters
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            
            # Update batch normalization statistics
            for target_module, module in zip(self.actor_target.modules(), self.actor.modules()):
                if isinstance(target_module, torch.nn.BatchNorm1d):
                    target_module.running_mean.mul_(1.0 - self.tau).add_(self.tau * module.running_mean)
                    target_module.running_var.mul_(1.0 - self.tau).add_(self.tau * module.running_var)
            
            for target_module, module in zip(self.critic_target.modules(), self.critic.modules()):
                if isinstance(target_module, torch.nn.BatchNorm1d):
                    target_module.running_mean.mul_(1.0 - self.tau).add_(self.tau * module.running_mean)
                    target_module.running_var.mul_(1.0 - self.tau).add_(self.tau * module.running_var)

        self.logger.log_training_step(self.training_step, actor_loss.item(), critic_loss.item(), None, None)
        self.training_step += 1
        converged = self.check_early_stopping()

        return actor_loss.item(), critic_loss.item(), converged

    def save_models(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, f"{path}_ddpg.pth")
    
    def load_models(self, path):
        checkpoint = torch.load(f"{path}_ddpg.pth")
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
