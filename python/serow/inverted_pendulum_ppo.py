import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import unittest
import os

from ppo import PPO

params = {
    'state_dim': 3,
    'action_dim': 1,
    'max_action': 2.0,
    'min_action': -2.0,
    'clip_param': 0.2,
    'value_loss_coef': 0.5,  
    'entropy_coef': 0.01,    
    'gamma': 0.995,
    'gae_lambda': 0.98,
    'ppo_epochs': 20,         
    'batch_size': 256,      
    'max_grad_norm': 0.3,
    'actor_lr': 5e-4,       
    'critic_lr': 5e-4,       
    'buffer_size': 10000,
    'max_state_value': 1e4,
    'min_state_value': -1e4,
    'n_steps': 8192,
    'update_lr': True,
    'convergence_threshold': 0.1,
    'critic_convergence_threshold': 1.0,
    'window_size': 20,
    'checkpoint_dir': 'policy/inverted_pendulum',
    'total_steps': 100000,  # Reduced from default 1,000,000 to 100,000 for faster decay
    'final_lr_ratio': 0.1,  # Learning rate will decay to 10% of initial value
}

class SharedNetwork(nn.Module):
    def __init__(self, state_dim):
        super(SharedNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        nn.init.orthogonal_(self.layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer2.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.layer1.bias, 0.0)
        torch.nn.init.constant_(self.layer2.bias, 0.0)

    def forward(self, state):
        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return x

class Actor(nn.Module):
    def __init__(self, params, shared_network):
        super(Actor, self).__init__()
        self.shared_network = shared_network
        
        # Policy network
        self.mean_layer = nn.Linear(64, params['action_dim'])
        self.log_std = nn.Parameter(torch.zeros(params['action_dim']))
        
        # Initialize weights
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)  # Smaller gain for policy
        torch.nn.init.constant_(self.mean_layer.bias, 0.0)
        
        self.max_action = params['max_action']
        self.min_action = params['min_action']
        self.action_dim = params['action_dim']
        
        # Action scaling
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

    def forward(self, state):
        x = self.shared_network(state)
        x = F.relu(x)
        mean = self.mean_layer(x)
        # Clamp log_std for numerical stability
        log_std = self.log_std.clamp(-20, 2)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Ensure state has shape (batch_size, state_dim)
        if state.dim() == 1:
            state = state.reshape(1, -1)  # Reshape to (1, state_dim)
        elif state.dim() == 2:
            if state.shape[0] == 1:
                state = state.reshape(1, -1)  # Ensure shape is (1, state_dim)
            else:
                state = state.T  # Transpose if shape is (state_dim, 1)
        
        state = state.to(next(self.parameters()).device)
        
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
        
        # Scale to action bounds using tanh
        action_scaled = torch.tanh(action) * self.action_scale + self.action_bias
        
        # Calculate log probability
        if not deterministic:
            # Log prob with tanh correction
            log_prob = normal.log_prob(action).sum(dim=-1)
            # Apply tanh correction
            log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-8).sum(dim=-1)
        else:
            log_prob = torch.zeros(1)
        
        return action_scaled.detach().cpu().numpy()[0], log_prob.detach().cpu().item()
    
    def evaluate_actions(self, states, actions):
        """Evaluate log probabilities and entropy for given state-action pairs"""
        mean, log_std = self.forward(states)
        std = log_std.exp()
        
        # Convert actions back to raw space (inverse of tanh scaling)
        actions_normalized = (actions - self.action_bias) / self.action_scale
        actions_normalized = torch.clamp(actions_normalized, -1 + 1e-7, 1 - 1e-7)
        actions_raw = torch.atanh(actions_normalized)
        
        # Calculate log probabilities
        normal = torch.distributions.Normal(mean, std)
        log_probs = normal.log_prob(actions_raw).sum(dim=-1, keepdim=True)
        
        # Apply tanh correction
        log_probs -= torch.log(1 - actions_normalized.pow(2) + 1e-8).sum(dim=-1, keepdim=True)
        
        # Calculate entropy
        entropy = normal.entropy().sum(dim=-1)
        return log_probs, entropy
    
class Critic(nn.Module):
    def __init__(self, params, shared_network):
        super(Critic, self).__init__()
        self.shared_network = shared_network
        self.value_layer = nn.Linear(64, 1)
        nn.init.orthogonal_(self.value_layer.weight, gain=1.0)
        torch.nn.init.constant_(self.value_layer.bias, 0.0)

    def forward(self, state):
        x = self.shared_network(state)
        x = F.relu(x)
        return self.value_layer(x)

# Inverted Pendulum Environment
class InvertedPendulum:
    def __init__(self):
        self.g = 9.81
        self.l = 1.0
        self.m = 1.0
        self.max_angle = np.pi  # Allow full rotation
        self.max_angular_vel = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.state = None
        self.reset()

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reset(self):
        # Start with a small random angle near upright
        theta = np.random.uniform(-np.pi/4, np.pi/4)
        theta_dot = np.random.uniform(-1.0, 1.0)
        # Convert to [cos(theta), sin(theta), theta_dot] representation
        self.state = np.array([np.cos(theta), np.sin(theta), theta_dot])
        return self.state

    def step(self, action):
        action = np.clip(action, -self.max_torque, self.max_torque)
        done = 0.0

        # Recover theta from cos(theta) and sin(theta)
        cos_theta, sin_theta, theta_dot = self.state
        theta = np.arctan2(sin_theta, cos_theta)
        
        # Calculate next state
        theta_ddot = (self.g / self.l) * np.sin(theta) + (action / (self.m * self.l**2))
        theta_dot = theta_dot + theta_ddot * self.dt
        theta = self.angle_normalize(theta + theta_dot * self.dt)
        
        # Convert back to [cos(theta), sin(theta), theta_dot] representation
        self.state = np.array([np.cos(theta), np.sin(theta), theta_dot])
        
        # Primary reward: exponential decay based on angle from upright
        angle_from_upright = abs(theta)
        angle_reward = np.exp(-2.0 * angle_from_upright)  # High reward when upright
        
        # Stability bonus: reward for low angular velocity when near upright
        if angle_from_upright < 0.5:  # Only when reasonably upright
            stability_bonus = 0.5 * np.exp(-abs(theta_dot))
        else:
            stability_bonus = 0.0
        
        # Control penalty
        control_penalty = 0.001 * action**2
        
        reward = float(angle_reward) + float(stability_bonus) - float(control_penalty)
        
        # Termination condition - only terminate for extreme angular velocities
        done = 1.0 if abs(theta_dot) > self.max_angular_vel else 0.0

        return self.state, reward, done

# Unit tests for PPO with Inverted Pendulum
class TestPPOInvertedPendulum(unittest.TestCase):
    def setUp(self):
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.max_action = params['max_action']
        self.min_action = params['min_action']
        
        # Create device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create shared network
        self.shared_network = SharedNetwork(self.state_dim)
        
        # Create actor and critic with shared network
        self.actor = Actor(params, self.shared_network)
        self.critic = Critic(params, self.shared_network)
        self.agent = PPO(self.actor, self.critic, params, device=self.device, normalize_state=False)
        self.env = InvertedPendulum()

    def test_initialization(self):
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertEqual(self.agent.max_action, self.max_action)
        self.assertEqual(self.agent.min_action, self.min_action)
        self.assertIsInstance(self.agent.actor, Actor)
        self.assertIsInstance(self.agent.critic, Critic)

    def test_get_action(self):
        state = self.env.reset()
        action, _ = self.agent.actor.get_action(state, deterministic=False)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(action >= self.min_action) and np.all(action <= self.max_action))

    def test_add_to_buffer(self):
        state = self.env.reset()
        action, log_prob = self.agent.actor.get_action(state)
        value = self.agent.critic(torch.FloatTensor(state).reshape(1, -1).to(self.device)).item()
        next_state, reward, done = self.env.step(action)
        self.agent.add_to_buffer(state, action, reward, next_state, done, value, log_prob)
        self.assertEqual(len(self.agent.buffer), 1)
        stored = self.agent.buffer[0]
        
        # Use np.allclose for floating-point comparisons with a small tolerance
        state_flat = state.flatten()
        self.assertTrue(np.allclose(stored[0], state_flat.flatten(), rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[1], action, rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[2], reward, rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[3], next_state.flatten(), rtol=1e-5, atol=1e-5))
        self.assertEqual(stored[4], done)
        self.assertEqual(stored[5], value)
        self.assertEqual(stored[6], log_prob)

    def test_train_and_evaluate(self):   
        max_steps_per_episode = 2048
        episode_rewards = []

        best_reward = float('-inf')
        max_episodes = 250
        collected_steps = 0
        
        # Create checkpoint directory
        checkpoint_dir = 'policy/inverted_pendulum'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Update params with checkpoint and early stopping parameters
        params.update({
            'checkpoint_dir': checkpoint_dir,
            'window_size': 20, 
        })
        
        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_critic_losses = []
            episode_policy_losses = []
            for step in range(max_steps_per_episode):
                action, log_prob = self.agent.actor.get_action(state, deterministic=False)
                value = self.agent.critic(torch.FloatTensor(state).reshape(1, -1).to(self.device)).item()
                next_state, reward, done = self.env.step(action)
                self.agent.add_to_buffer(state, action, reward, next_state, done, value, log_prob)
                
                episode_reward += reward
                collected_steps += 1
                
                if collected_steps >= params['n_steps']:
                    policy_loss, critic_loss, entropy = self.agent.train()
                    collected_steps = 0
                    if policy_loss != 0.0 or critic_loss != 0.0:  # Only print if actual training occurred
                        print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")
                        episode_critic_losses.append(critic_loss)
                        episode_policy_losses.append(policy_loss)

                if done > 0.0 or step == max_steps_per_episode - 1:
                    state = self.env.reset()
                    break
                else:
                    state = next_state
            
            episode_rewards.append(episode_reward)
            self.agent.save_checkpoint(episode_reward)
            
            # Check for early stopping
            if self.agent.check_early_stopping(episode_reward, np.mean(episode_critic_losses)):
                print("Early stopping triggered. Loading best model...")
                self.agent.load_checkpoint(os.path.join(checkpoint_dir, 'best_model.pth'))
                break
                
            print(f"Episode {episode}/{max_episodes}, Reward: {episode_reward:.2f}, Best Reward: " 
                  f"{self.agent.best_reward:.2f}")

            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode}, Avg Reward (last 10): {avg_reward:.2f}")
        
        # Convert episode_rewards to numpy arrays and compute a smoothed reward curve using a low pass filter
        episode_rewards = np.array(episode_rewards)
        smoothed_episode_rewards = []
        smoothed_episode_reward = episode_rewards[0]
        alpha = 0.95
        for i in range(len(episode_rewards)):
            smoothed_episode_reward = alpha * smoothed_episode_reward + (1.0 - alpha) * episode_rewards[i]
            smoothed_episode_rewards.append(smoothed_episode_reward)

        # Plot results
        plt.figure(figsize=(15, 5))
        plt.plot(episode_rewards, label='Episode Rewards')
        plt.plot(smoothed_episode_rewards, label='Smoothed Rewards')
        plt.axhline(y=best_reward, color='r', linestyle='--', label='Best Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Over Time')
        plt.legend()
        plt.show()

        # Evaluate the policy and collect data for plotting
        state = self.env.reset()
        total_reward = 0.0  
        states = []
        actions = []
        rewards = []
        for step in range(max_steps_per_episode):
            action, _ = self.agent.actor.get_action(state, deterministic=True)
            next_state, reward, done = self.env.step(action)
            state_flat = np.array(state).reshape(-1)
            states.append(np.array([np.arctan2(state_flat[1], state_flat[0]),  state_flat[2]]))
            actions.append(action)
            rewards.append(reward) 
            total_reward += reward
            state = next_state
            if done:
                break
        
        # Convert to numpy arrays with consistent shapes
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Plot results
        plt.figure(figsize=(15, 5))

        # State plot (angle and angular velocity)
        plt.subplot(1, 3, 1)
        plt.plot(states[:, 0], label='Angle (rad)')
        plt.plot(states[:, 1], label='Angular Velocity (rad/s)')
        plt.xlabel('Time Step')
        plt.ylabel('State')
        plt.title('Pendulum State')
        plt.legend()
        plt.grid(True)

        # Action plot (torque)
        plt.subplot(1, 3, 2)
        plt.plot(actions, label='Torque')
        plt.xlabel('Time Step')
        plt.ylabel('Action')
        plt.title('Applied Torque (Nm)')
        plt.legend()
        plt.grid(True)

        # Reward plot
        plt.subplot(1, 3, 3)
        plt.plot(rewards, label='Reward')
        plt.xlabel('Time Step')
        plt.ylabel('Reward')
        plt.title('Reward Over Time')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    unittest.main()
