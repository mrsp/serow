import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
import os

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
        self.update_lr = params.get('update_lr', False)
        self.final_lr_ratio = params.get('final_lr_ratio', 0.1)
        self.total_steps = params.get('total_steps', 1000000)
        self.current_step = 0

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
            return 0.0, 0.0
        
        # Update learning rates before training
        if self.update_lr:
            self.update_learning_rates()  

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
                    
        return actor_loss.item(), critic_loss.item()
