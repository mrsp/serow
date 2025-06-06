import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import unittest
import os

from ddpg import DDPG

params = {
    'robot': 'inverted_pendulum',
    'state_dim': 3,
    'action_dim': 1,
    'max_action': 2.0,
    'min_action': -2.0,
    'gamma': 0.99,
    'tau': 0.01,
    'batch_size': 256,
    'actor_lr': 1e-4, 
    'critic_lr': 1e-4,
    'noise_scale': 0.5,
    'noise_decay': 0.9995,
    'max_grad_norm': 0.3,
    'buffer_size': 1000000,
    'max_state_value': 1e8,
    'n_steps': 2000,
    'min_state_value': -1e8,
    'train_for_batches': 5,
    'update_lr': True,
    'convergence_threshold': 0.1,
    'critic_convergence_threshold': 1.0,
    'window_size': 20,
    'checkpoint_dir': 'policy/inverted_pendulum/ddpg',
    'total_steps': 100000, 
    'final_lr_ratio': 0.1,  # Learning rate will decay to 10% of initial value
}

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.3):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class Actor(nn.Module):
    def __init__(self, params):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(params['state_dim'], 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, params['action_dim'])

        # Initialize with better weights for improved exploration
        nn.init.xavier_uniform_(self.layer1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.layer2.weight, gain=1.0)
        nn.init.uniform_(self.layer3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.layer3.bias, -3e-3, 3e-3)

        self.max_action = params['max_action']
        self.min_action = params['min_action']
        self.action_dim = params['action_dim']

        self.noise = OUNoise(params['action_dim'], sigma=0.2)
        self.noise_scale = params['noise_scale']
        self.noise_decay = params['noise_decay']

    def forward(self, state):
        x = F.tanh(self.layer1(state))
        x = F.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(next(self.parameters()).device)
            action = self.forward(state).cpu().numpy()[0]
            if not deterministic:
                noise = self.noise.sample() * self.noise_scale
                action = action + noise
                self.noise_scale *= self.noise_decay
                action = np.clip(action, self.min_action, self.max_action)
            return action

class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.state_layer = nn.Linear(params['state_dim'], 256)
        self.action_layer = nn.Linear(params['action_dim'], 256)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Better initialization for critic
        nn.init.xavier_uniform_(self.state_layer.weight, gain=1.0)
        nn.init.xavier_uniform_(self.action_layer.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        s = F.relu(self.state_layer(state))
        a = F.relu(self.action_layer(action))
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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

# Unit tests for DDPG with Inverted Pendulum
class TestDDPGInvertedPendulum(unittest.TestCase):
    def setUp(self):
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.max_action = params['max_action']
        self.min_action = params['min_action']

        # Create device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create actor and critic networks
        self.actor = Actor(params)
        self.critic = Critic(params)
        self.agent = DDPG(self.actor, self.critic, params, device=self.device)
        self.env = InvertedPendulum()

    def test_initialization(self):
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertEqual(self.agent.max_action, self.max_action)
        self.assertEqual(self.agent.min_action, self.min_action)
        self.assertIsInstance(self.agent.actor, Actor)
        self.assertIsInstance(self.agent.critic, Critic)
        self.assertIsInstance(self.agent.actor_target, Actor)
        self.assertIsInstance(self.agent.critic_target, Critic)

    def test_get_action(self):
        state = self.env.reset()
        action = self.agent.actor.get_action(state, deterministic=False)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(action >= self.min_action) and np.all(action <= self.max_action))

    def test_add_to_buffer(self):
        state = self.env.reset()
        action = self.agent.actor.get_action(state, deterministic=False)
        next_state, reward, done = self.env.step(action)
        self.agent.add_to_buffer(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.buffer), 1)
        stored = self.agent.buffer[0]
        
        # Use np.allclose for floating-point comparisons with a small tolerance
        state_flat = state.flatten()
        self.assertTrue(np.allclose(stored[0], state_flat.flatten(), rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[1], action, rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[2], reward, rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[3], next_state.flatten(), rtol=1e-5, atol=1e-5))
        self.assertEqual(stored[4], done)

    def test_train_and_evaluate(self):
        max_steps_per_episode = 2048
        episode_rewards = []
        max_episodes = 250
        best_reward = float('-inf')
        max_episodes = 250
        collected_steps = 0

        # Create checkpoint directory
        checkpoint_dir = 'policy/inverted_pendulum/ddpg'
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

            # Exploration phases - more exploration early on
            noise_scale = max(0.1, 0.3 * (0.99 ** episode))  # Decay noise over episodes
            self.agent.actor.noise_scale = noise_scale
            
            for step in range(max_steps_per_episode):
                action = self.agent.actor.get_action(state, deterministic=False)
                next_state, reward, done = self.env.step(action)
                self.agent.add_to_buffer(state, action, reward, next_state, done)
                
                episode_reward += reward
                collected_steps += 1

                if collected_steps >= params['n_steps']:
                    actor_loss, critic_loss = self.agent.train()
                    if actor_loss != 0.0 or critic_loss != 0.0:  # Only print if actual training occurred
                        print(f"Policy Loss: {actor_loss:.4f}, Value Loss: {critic_loss:.4f}")
                        episode_critic_losses.append(critic_loss)
                        episode_policy_losses.append(actor_loss)

                    collected_steps = 0

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
            action = self.agent.actor.get_action(state, deterministic=True)
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
