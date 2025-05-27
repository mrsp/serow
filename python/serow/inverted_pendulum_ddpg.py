import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import unittest

from ddpg import DDPG

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
        theta = np.random.uniform(-np.pi, np.pi)
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
        
        # Improved reward function using cos(theta) and sin(theta)
        # The upright position is when cos(theta) = 1, sin(theta) = 0
        angledot_cost = 0.1 * theta_dot**2
        torque_cost = 0.001 * action**2
        reward = float(np.cos(theta)) - float(angledot_cost) - float(torque_cost)
        done = 1.0 if (abs(theta_dot) > self.max_angular_vel) else 0.0
        if done:
            reward -= 20  # Stronger penalty for falling
        return self.state, reward, done

# Unit tests for DDPG with Inverted Pendulum
class TestDDPGInvertedPendulum(unittest.TestCase):
    def setUp(self):
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # torque
        self.max_action = 2.0
        self.min_action = -2.0

        # Create device
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'

        params = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'max_action': self.max_action,
            'min_action': self.min_action,
            'gamma': 0.99,
            'tau': 0.01,
            'batch_size': 128,
            'actor_lr': 1e-4, 
            'critic_lr': 1e-4,
            'noise_scale': 0.5,
            'noise_decay': 0.9995,
            'buffer_size': 1000000,
            'max_state_value': 1e8,
            'min_state_value': -1e8,
            'train_for_batches': 5,
        }
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

    def test_train(self):
        # Fill buffer with some transitions
        state = self.env.reset()
        for _ in range(100):
            action = self.agent.actor.get_action(state, deterministic=False)
            next_state, reward, done = self.env.step(action)
            self.agent.add_to_buffer(state, action, reward, next_state, done)
            state = next_state if not done else self.env.reset()
        # Train and check if it runs without errors
        try:
            self.agent.train()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Training failed with error: {e}")

    def test_policy_evaluation(self):
        max_steps_per_episode = 2048
        episode_rewards = []
        best_reward = float('-inf')
        max_episodes = 1000
        reward_history = []
        window_size = 10
        convergence_threshold = 0.5  # How close to best reward we need to be (as a fraction)

        update_steps = 1  # Train after collecting this many timesteps

        episode_rewards = []
        best_reward = float('-inf')
        collected_steps = 0

        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0.0  

            # Exploration phases - more exploration early on
            noise_scale = max(0.1, 0.3 * (0.99 ** episode))  # Decay noise over episodes
            self.agent.actor.noise_scale = noise_scale
            
            for step in range(max_steps_per_episode):
                action = self.agent.actor.get_action(state, deterministic=False)
                next_state, reward, done = self.env.step(action)
                self.agent.add_to_buffer(state, action, reward, next_state, done)
                
                episode_reward += reward
                collected_steps += 1

                if done > 0.0:
                    state = self.env.reset()
                else:
                    state = next_state
                
                if collected_steps >= update_steps:
                    self.agent.train()
                    collected_steps = 0

                if done or step == max_steps_per_episode - 1:
                    break

            episode_rewards.append(episode_reward)
            if episode_reward > best_reward:
                best_reward = episode_reward
            print(f"Episode {episode}/{max_episodes}, Reward: {episode_reward:.2f}, Best Reward: {best_reward:.2f}")

            # Update reward history
            reward_history.append(episode_reward)
            if len(reward_history) > window_size:
                reward_history.pop(0)
                
            # Check convergence by comparing recent rewards to best reward
            if len(reward_history) >= window_size:
                recent_rewards = np.array(reward_history)
                # Calculate how close recent rewards are to best reward
                reward_ratios = recent_rewards / (abs(best_reward) + 1e-6)  # Avoid division by zero
                # Check if all recent rewards are within threshold of best reward
                if np.all(reward_ratios >= (1.0 - convergence_threshold)):
                    print(f"Training converged! Recent rewards are within {convergence_threshold*100}% of best reward {best_reward:.2f}")
                    break

            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode}, Avg Reward (last 10): {avg_reward:.2f}")
        
        # Convert episode_rewards to numpy arrays and compute a smoothed reward curve using a low pass filter
        episode_rewards = np.array(episode_rewards)
        smoothed_episode_rewards = []
        smoothed_episode_reward = episode_rewards[0]
        alpha = 0.8
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
        total_reward = 0.0  # Initialize as float
        states = []
        actions = []
        rewards = []
        for step in range(max_steps_per_episode):
            action = self.agent.actor.get_action(state, deterministic=True)
            next_state, reward, done = self.env.step(action)
            state_flat = np.array(state).reshape(-1)
            states.append(np.array([np.arctan2(state_flat[1], state_flat[0]),  state_flat[2]]))
            actions.append(action)
            rewards.append(float(reward))  # Ensure reward is float
            total_reward += float(reward)  # Ensure reward is float
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
