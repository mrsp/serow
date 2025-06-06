import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import unittest
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

class InvertedPendulum:
    def __init__(self):
        self.g = 9.81
        self.l = 1.0
        self.m = 1.0
        self.max_angle = np.pi
        self.max_angular_vel = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.state = None
        self.reset()

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reset(self):
        # Start very close to upright for initial learning
        theta = np.random.uniform(-0.05, 0.05)  # Very small angle near upright
        theta_dot = np.random.uniform(-0.05, 0.05)  # Very small angular velocity
        self.state = np.array([np.cos(theta), np.sin(theta), theta_dot])
        return self.state

    def step(self, action):
        action = np.clip(action, -self.max_torque, self.max_torque)
        
        # Recover theta from cos(theta) and sin(theta)
        cos_theta, sin_theta, theta_dot = self.state
        theta = np.arctan2(sin_theta, cos_theta)
        
        # Physics simulation
        theta_ddot = (self.g / self.l) * np.sin(theta) + (action / (self.m * self.l**2))
        theta_dot = theta_dot + theta_ddot * self.dt
        theta_dot = np.clip(theta_dot, -self.max_angular_vel, self.max_angular_vel)
        theta = self.angle_normalize(theta + theta_dot * self.dt)
        
        # Update state
        self.state = np.array([np.cos(theta), np.sin(theta), theta_dot])
        
        # Improved reward function with better shaping
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
        
        reward = angle_reward + stability_bonus - control_penalty
        
        # Termination condition - only terminate for extreme angular velocities
        done = abs(theta_dot) > self.max_angular_vel
        
        return self.state, reward, done

class InvertedPendulumEnv(gym.Env):
    def __init__(self):
        super(InvertedPendulumEnv, self).__init__()
        self.pendulum = InvertedPendulum()
        
        # Action space: continuous torque
        self.action_space = gym.spaces.Box(
            low=-self.pendulum.max_torque,
            high=self.pendulum.max_torque,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space: [cos(theta), sin(theta), theta_dot]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -self.pendulum.max_angular_vel]),
            high=np.array([1.0, 1.0, self.pendulum.max_angular_vel]),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        state = self.pendulum.reset()
        return state.astype(np.float32), {}

    def step(self, action):
        # Ensure action is scalar
        if isinstance(action, np.ndarray):
            action = action[0]
        
        state, reward, done = self.pendulum.step(action)
        
        # Convert to proper types
        state = state.astype(np.float32)
        reward = float(reward)
        terminated = bool(done)
        truncated = False  # We don't use time limits
        
        return state, reward, terminated, truncated, {}

# Unit tests for PPO with Inverted Pendulum
class TestPPOInvertedPendulum(unittest.TestCase):
    def setUp(self):
        device = "cpu"
        print(f"Using device: {device}")
        
        # Create multiple parallel environments for training
        n_envs = 8  
        self.vec_env = make_vec_env(
            InvertedPendulumEnv,
            n_envs=n_envs,
            seed=0,
            vec_env_cls=DummyVecEnv
        )
        
        # Keep a single environment for evaluation
        self.env = InvertedPendulumEnv()
        
        # Optimized PPO parameters for inverted pendulum with GPU support
        self.params = {
            'learning_rate': 5e-4, # Step size for updating neural network weights
            'n_steps': 8192, # Number of steps to collect before updating the policy
            'batch_size': 256, # Number of samples to use for each policy update (minibatch size)
            'n_epochs': 20, # Number of epochs to train the policy
            'gamma': 0.995, # Discount factor for future rewards
            'gae_lambda': 0.98, # Lambda parameter for Generalized Advantage Estimation
            'clip_range': 0.2, # Clipping range for the policy gradient
            'ent_coef': 0.01, # Entropy coefficient for regularization (exploration bonus)
            'vf_coef': 0.5, # Value function coefficient for regularization (how much to trust the value function - exploration bonus)
            'max_grad_norm': 0.3, # Maximum gradient norm for gradient clipping
            'verbose': 1, # Verbosity level
            'device': device, # Device to use for training
        }
        
        self.model = PPO("MlpPolicy", self.vec_env, **self.params)

    def test_initialization(self):
        self.assertIsNotNone(self.model)
        # Test that spaces match
        self.assertEqual(self.model.action_space.shape, self.env.action_space.shape)
        self.assertEqual(self.model.observation_space.shape, self.env.observation_space.shape)

    def test_policy_evaluation(self):
        print("Training PPO agent...")
        # Train with more timesteps and monitor progress
        self.model.learn(total_timesteps=1000000)  # Increased training time
        
        print("Evaluating trained policy...")
        # Test the trained policy
        state, _ = self.env.reset()
        total_reward = 0.0
        states = []
        actions = []
        rewards = []
        episode_length = 0
        max_steps = 1000
        
        for step in range(max_steps):
            action, _ = self.model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            # Extract angle from state representation
            cos_theta, sin_theta, theta_dot = state
            theta = np.arctan2(sin_theta, cos_theta)
            
            states.append([theta, theta_dot])
            actions.append(action[0] if isinstance(action, np.ndarray) else action)
            rewards.append(reward)
            total_reward += reward
            episode_length += 1
            
            state = next_state
            if terminated or truncated:
                break
        
        # Print episode statistics
        print(f"Episode finished after {episode_length} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward per step: {total_reward/episode_length:.2f}")
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Plot results
        plt.figure(figsize=(15, 5))

        # State plot (angle and angular velocity)
        plt.subplot(1, 3, 1)
        plt.plot(states[:, 0], label='Angle (rad)', linewidth=2)
        plt.plot(states[:, 1], label='Angular Velocity (rad/s)', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('State')
        plt.title('Pendulum State Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Action plot (torque)
        plt.subplot(1, 3, 2)
        plt.plot(actions, label='Torque', color='red', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('Torque (Nm)')
        plt.title('Applied Torque Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Reward plot
        plt.subplot(1, 3, 3)
        plt.plot(rewards, label='Reward', color='green', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('Reward')
        plt.title('Reward Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        
        # Assert that the agent learned something reasonable
        # The agent should achieve high average reward for staying upright
        avg_reward = total_reward / episode_length
        print(f"Final average reward: {avg_reward:.3f}")
        
        # Calculate percentage of time spent near upright (within 30 degrees)
        angles = states[:, 0]
        upright_time = np.sum(np.abs(angles) < np.pi/6) / len(angles) * 100
        print(f"Time spent within 30° of upright: {upright_time:.1f}%")
        
        self.assertGreater(avg_reward, 0.8, "Agent should achieve high average reward > 0.8")
        self.assertGreater(upright_time, 70, "Agent should spend >70% of time near upright")

if __name__ == '__main__':
    # Run a quick test to see if the agent can learn
    test_case = TestPPOInvertedPendulum()
    test_case.setUp()
    
    # Test trained agent
    test_case.test_policy_evaluation()
    
    # You can also run full unittest suite
    # unittest.main()
