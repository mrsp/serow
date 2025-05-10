import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import unittest

from ddpg import DDPG

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=1.0):
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
        self.layer1 = nn.Linear(params['state_dim'], 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, params['action_dim'])
        # Initialize the output layer with a much wider range
        nn.init.xavier_uniform_(self.layer3.weight, gain=1.4141) # sqrt(2)
        nn.init.uniform_(self.layer3.bias, -0.1, 0.1)  # Non-zero bias

        self.max_action = params['max_action']
        self.min_action = params['min_action']
        self.action_dim = params['action_dim']

        self.noise = OUNoise(params['action_dim'], sigma=1.0)
        self.noise_scale = params['noise_scale']
        self.noise_decay = params['noise_decay']

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

    # epsilon-greedy policy
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
        self.layer1 = nn.Linear(params['state_dim'] + params['action_dim'], 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

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
        angle_cost = theta**2
        angledot_cost = 0.1 * theta_dot**2
        torque_cost = 0.001 * action**2
        reward = - float(angle_cost) - float(angledot_cost) - float(torque_cost)
        
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
            'batch_size': 64,  
            'actor_lr': 5e-4, 
            'critic_lr': 1e-4,
            'noise_scale': 2.0,
            'noise_decay': 0.995,
            'buffer_size': 1000000,
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
        self.assertTrue(np.array_equal(stored[0], state))
        self.assertTrue(np.array_equal(stored[1], action))
        self.assertEqual(stored[2], reward)
        self.assertTrue(np.array_equal(stored[3], next_state))
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
        # Train the agent for longer
        state = self.env.reset()
        total_steps = 0
        max_steps = 200 
        episode_reward = 0.0  # Initialize as float
        best_reward = float('-inf')
        
        for episode in range(100):
            for step in range(max_steps):
                total_steps += 1
                action = self.agent.actor.get_action(state, deterministic=False)
                next_state, reward, done = self.env.step(action)
                self.agent.add_to_buffer(state, action, reward, next_state, done)
                self.agent.train()
            
                episode_reward += float(reward)  # Ensure reward is float
                state = next_state
                
                if done or step == max_steps - 1:
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                    print(f"Step {step}, Episode Reward: {episode_reward:.2f}, Best Reward: {best_reward:.2f}")
                    state = self.env.reset()
                    episode_reward = 0.0  # Reset as float
        
        # Evaluate the policy and collect data for plotting
        state = self.env.reset()
        total_reward = 0.0  # Initialize as float
        states = []
        actions = []
        rewards = []
        for step in range(max_steps):
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
