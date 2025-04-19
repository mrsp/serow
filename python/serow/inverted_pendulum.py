import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest
import matplotlib.pyplot as plt

from ddpg import DDPG

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 128)
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
        self.max_angle = np.pi / 4
        self.max_angular_vel = 8.0
        self.max_torque = 2.0
        self.dt = 0.02
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.array([0.0, 0.0])  # [theta, theta_dot]
        return self.state

    def step(self, action):
        action = np.clip(action, -self.max_torque, self.max_torque)
        theta, theta_dot = self.state
        theta_ddot = (self.g / self.l) * np.sin(theta) + (action / (self.m * self.l**2))
        theta_dot = theta_dot + theta_ddot * self.dt
        theta = theta + theta_dot * self.dt
        theta_dot = np.clip(theta_dot, -self.max_angular_vel, self.max_angular_vel)
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        self.state = np.array([theta, theta_dot])
        reward = 1.0 - abs(theta) / self.max_angle
        done = abs(theta) > self.max_angle
        return self.state, reward, done

# Unit tests for DDPG with Inverted Pendulum
class TestDDPGInvertedPendulum(unittest.TestCase):
    def setUp(self):
        self.state_dim = 2  # [theta, theta_dot]
        self.action_dim = 1  # torque
        self.max_action = 2.0
        self.min_action = -2.0
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.agent = DDPG(self.actor, self.critic, self.state_dim, self.action_dim, self.max_action, self.min_action)
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
        action = self.agent.get_action(state)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(action >= -self.max_action) and np.all(action <= self.max_action))

    def test_add_to_buffer(self):
        state = self.env.reset()
        action = self.agent.get_action(state)
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
            action = self.agent.get_action(state)
            next_state, reward, done = self.env.step(action)
            self.agent.add_to_buffer(state, action, reward, next_state, done)
            state = next_state if not done else self.env.reset()
        # Train and check if it runs without errors
        try:
            self.agent.train()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Training failed with error: {e}")

    # def test_policy_learning(self):
    #     # Train the agent for a few episodes
    #     episodes = 50
    #     max_steps = 200
    #     rewards = []
    #     for episode in range(episodes):
    #         state = self.env.reset()
    #         episode_reward = 0
    #         for step in range(max_steps):
    #             action = self.agent.get_action(state, add_noise=episode < episodes // 2)
    #             next_state, reward, done = self.env.step(action)
    #             self.agent.add_to_buffer(state, action, reward, next_state, done)
    #             self.agent.train()
    #             episode_reward += reward
    #             state = next_state
    #             if done:
    #                 break
    #         rewards.append(episode_reward)
    #     # Check if the average reward in the last 10 episodes is reasonable
    #     avg_reward = np.mean(rewards[-10:])
    #     self.assertGreater(avg_reward, 50.0, f"Average reward {avg_reward} is too low, expected > 50.0")

    def test_policy_evaluation(self):
        # Train the agent briefly
        state = self.env.reset()
        for _ in range(1000):
            action = self.agent.get_action(state)
            next_state, reward, done = self.env.step(action)
            self.agent.add_to_buffer(state, action, reward, next_state, done)
            self.agent.train()
            state = next_state if not done else self.env.reset()
        # Evaluate the policy and collect data for plotting
        state = self.env.reset()
        total_reward = 0
        max_steps = 200
        states = []
        actions = []
        rewards = []
        for step in range(max_steps):
            action = self.agent.get_action(state, add_noise=False)
            next_state, reward, done = self.env.step(action)
            # Ensure state is a 1D array
            state_flat = np.array(state).reshape(-1)
            states.append(state_flat)
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
        plt.plot(states[:, 0], label='Angle (theta)')
        plt.plot(states[:, 1], label='Angular Velocity (theta_dot)')
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
        plt.title('Applied Torque')
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
        plt.savefig('pendulum_evaluation_plots.png')
        plt.show()

if __name__ == '__main__':
    unittest.main()
