import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import unittest
import os

from ddpg import DDPG

params = {
    'device': 'cpu',
    'robot': 'inverted_pendulum',
    'state_dim': 3,
    'action_dim': 1,
    'max_action': 2.0,
    'min_action': -2.0,
    'clip_param': 0.2,
    'value_clip_param': 0.2,
    'value_loss_coef': 0.5,  
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 128,
    'max_grad_norm': 1.0,
    'max_episodes': 200,
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'noise_scale': 0.1,
    'noise_decay': 0.9999,
    'buffer_size': 1000000,
    'max_state_value': 1e4,
    'min_state_value': -1e4,
    'n_steps': 256,
    'train_for_batches': 1,
    'convergence_threshold': 0.25,
    'critic_convergence_threshold': 0.15,
    'returns_window_size': 20,
    'value_loss_window_size': 20,
    'checkpoint_dir': 'policy/inverted_pendulum/ddpg',
    'total_training_steps': 100000,
    'total_steps': 200000,
    'final_lr_ratio': 0.01,
    'check_value_loss': True,
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
        self.layer1 = nn.Linear(params['state_dim'], 128)
        self.layer2 = nn.Linear(128, 128)
        nn.init.orthogonal_(self.layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer2.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.layer1.bias, 0.0)
        torch.nn.init.constant_(self.layer2.bias, 0.0)
        
        # Policy network
        self.mean_layer = nn.Linear(128, params['action_dim'])

        # Initialize weights
        nn.init.orthogonal_(self.mean_layer.weight, gain=np.sqrt(2)) 
        torch.nn.init.constant_(self.mean_layer.bias, 0.0)
        
        self.max_action = params['max_action']
        self.min_action = params['min_action']
        self.action_dim = params['action_dim']
        self.device = params['device']
        # Action scaling
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

        # Exploration noise
        self.noise = OUNoise(params['action_dim'], sigma=0.1)
        self.noise_scale = params['noise_scale']
        self.noise_decay = params['noise_decay']

    def forward(self, state):
        x = self.layer1(state)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        mean = self.mean_layer(x)
        # Apply tanh to bound the output to [-1, 1], then scale to action range
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean

    def get_action(self, state, deterministic=False):
        mean = self.forward(torch.FloatTensor(state).reshape(1, -1).to(self.device))

        if deterministic:
            action = mean
        else:
            action = mean + torch.FloatTensor(self.noise.sample() * self.noise_scale).to(self.device)
            self.noise_scale *= self.noise_decay
            
        # Clamp to ensure actions are within bounds
        action = torch.clamp(action, self.min_action, self.max_action)
        return action.detach().cpu().numpy()[0]
    
class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.state_layer = nn.Linear(params['state_dim'], 128)
        self.action_layer = nn.Linear(params['action_dim'], 128)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)
        nn.init.orthogonal_(self.state_layer.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.action_layer.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer3.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.state_layer.bias, 0.0)
        torch.nn.init.constant_(self.action_layer.bias, 0.0)
        torch.nn.init.constant_(self.layer2.bias, 0.0)
        torch.nn.init.constant_(self.layer3.bias, 0.0)

    def forward(self, state, action):
        s = F.relu(self.state_layer(state))
        a = F.relu(self.action_layer(action))
        x = torch.cat([s, a], dim=-1)
        x = F.relu(self.layer2(x))
        # No activation on final layer to allow negative values
        return self.layer3(x)

# Inverted Pendulum Environment
class InvertedPendulum:
    def __init__(self):
        self.g = 9.81
        self.l = 1.0
        self.m = 1.0
        self.max_angle = np.pi  # Allow full rotation
        self.max_angular_vel = 8.0
        self.dt = 0.05
        self.state = None
        self.upright_steps = 0  # Track consecutive upright steps
        self.reset()

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reset(self):
        # Start with a small random angle near upright
        theta = np.random.uniform(-np.pi/6, np.pi/6)
        theta_dot = np.random.uniform(-0.1, 0.1)
        # Convert to [cos(theta), sin(theta), theta_dot] representation
        self.state = np.array([np.cos(theta), np.sin(theta), theta_dot])
        self.upright_steps = 0  # Reset counter on episode reset
        return self.state.flatten()

    def step(self, action):
        # Recover theta from cos(theta) and sin(theta)
        cos_theta, sin_theta, theta_dot = self.state
        theta = np.arctan2(sin_theta, cos_theta)
        
        # Calculate next state
        theta_ddot = (self.g / self.l) * np.sin(theta) + (action / (self.m * self.l**2))
        theta_dot = theta_dot + theta_ddot * self.dt
        theta = theta + theta_dot * self.dt
        self.state = np.array([np.cos(theta), np.sin(theta), theta_dot])

        # Normalize angle to [-π, π]
        theta = self.angle_normalize(theta)

        # Primary reward: exponential reward for being upright
        angle_reward = np.exp(-3.0 * theta**2)  # Peak at θ=0, smooth decay
        
        # Velocity penalty: encourage low angular velocity
        velocity_penalty = 0.1 * theta_dot**2
        
        # Control penalty: encourage efficient control
        control_penalty = 0.01 * action**2
        
        # Bonus for sustained balancing
        if abs(theta) < 0.1 and abs(theta_dot) < 0.5:
            sustained_bonus = 1.0
        else:
            sustained_bonus = 0.0
        
        reward = angle_reward - velocity_penalty - control_penalty + sustained_bonus
        
        # Termination condition - only terminate for extreme angular velocities
        done = 0.0
        if abs(theta_dot) > self.max_angular_vel:
            done = 1.0
            reward = -10.0
        return self.state.flatten(), float(reward), float(done)

# Unit tests for DDPG with Inverted Pendulum
class TestDDPGInvertedPendulum(unittest.TestCase):
    def setUp(self):
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.max_action = params['max_action']
        self.min_action = params['min_action']
        
        # Create device
        self.device = 'cpu'
        
        # Create actor and critic 
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
        # stored[0] and stored[3] are the original state and next_state (not flattened)
        self.assertTrue(np.allclose(stored[0], state, rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[1], action, rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[2], reward, rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[3], next_state, rtol=1e-5, atol=1e-5))
        self.assertEqual(stored[4], done)

    def test_train_and_evaluate(self):   
        max_steps_per_episode = 2500
        checkpoint_dir = 'policy/inverted_pendulum/ddpg'
        checkpoint_file = 'policy_checkpoint_inverted_pendulum.pth'
        # Update params with checkpoint 
        params.update({
            'checkpoint_dir': checkpoint_dir
        })

        # Check if there is a checkpoint
        if os.path.exists(params['checkpoint_dir']) and os.path.exists(params['checkpoint_dir'] + '/' + checkpoint_file):
            self.agent.load_checkpoint(params['checkpoint_dir'] + '/' + checkpoint_file)
            print("Loaded checkpoint")
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print("No checkpoint found")
            self.agent.train()
            episode_returns = []
            best_return = float('-inf')
            max_episodes = params['max_episodes']
            n_steps = params['n_steps']
            params['total_steps'] = max_episodes * max_steps_per_episode
            params['total_training_steps'] = params['total_steps'] // n_steps
            collected_steps = 0
        
            # Warm-up phase: collect initial experiences without training
            print("Warming up buffer with initial experiences...")
            warmup_episodes = 10
            for warmup_ep in range(warmup_episodes):
                state = self.env.reset()
                for step in range(max_steps_per_episode):
                    # Use random actions during warm-up
                    action = np.random.uniform(params['min_action'], params['max_action'], params['action_dim'])
                    next_state, reward, done = self.env.step(action)
                    self.agent.add_to_buffer(state, action, reward, next_state, done)
                    state = next_state
                    if done > 0.0:
                        break
            print(f"Warm-up complete. Buffer size: {len(self.agent.buffer)}")
        
            converged = False
            for episode in range(max_episodes):
                state = self.env.reset()
                episode_return = 0.0
                # Exploration phases - more exploration early on
                noise_scale = max(0.1, 0.3 * (0.99 ** episode))  # Decay noise over episodes
                self.agent.actor.noise_scale = noise_scale
            
                for step in range(max_steps_per_episode):
                    action = self.agent.actor.get_action(state, deterministic=False)
                    next_state, reward, done = self.env.step(action)
                    self.agent.add_to_buffer(state, action, reward, next_state, done)
                    
                    episode_return += reward
                    collected_steps += 1
                    
                    if collected_steps >= params['n_steps']:
                        policy_loss, critic_loss, converged = self.agent.learn()
                        collected_steps = 0
                        if step % 100 == 0 and (policy_loss != 0.0 or critic_loss != 0.0):  # Only print if actual training occurred
                            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {critic_loss:.4f}")

                    if done > 0.0 or step == max_steps_per_episode - 1:
                        if (done > 0.0 and step < max_steps_per_episode - 1):
                            print("Episode terminated early")
                        break
                    state = next_state
                
                # Episode completed, log the return
                episode_returns.append(episode_return)
                self.agent.logger.log_episode(episode_return, step)
                if episode_return > best_return:
                    best_return = episode_return
                    self.agent.best_return = best_return  # Update agent's best_return

                # Check for early stopping
                if converged:
                    print("Early stopping triggered. Loading best model...")
                    break
                    
                print(f"Episode {episode + 1}/{max_episodes}, Steps: {step + 1}/{max_steps_per_episode}, Return: {episode_return:.2f}, Best Return: " 
                    f"{best_return:.2f}, Noise: {noise_scale:.3f}")

                if episode % 10 == 0:
                    avg_return = np.mean(episode_returns[-10:])
                    print(f"Episode {episode + 1}, Avg Return (last 10): {avg_return:.2f}")
            
            self.agent.logger.plot_training_curves()

        # Evaluate the policy and collect data for plotting
        self.agent.eval()
        state = self.env.reset()
        total_reward = 0.0  
        angles = []
        velocities = []
        actions = []
        rewards = []
        for step in range(max_steps_per_episode):
            action = self.agent.actor.get_action(state, deterministic=True)
            next_state, reward, done = self.env.step(action)
            angles.append(np.arctan2(state[1], state[0]).item())
            velocities.append(state[2].item())
            actions.append(action)
            rewards.append(reward) 
            total_reward += reward
            state = next_state
            if done:
                break
        
        # Convert to numpy arrays with consistent shapes
        angles = np.array(angles)
        velocities = np.array(velocities)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Plot results
        plt.figure(figsize=(15, 5))

        # State plot (angle and angular velocity)
        plt.subplot(1, 3, 1)
        plt.plot(angles, label='Angle (rad)')
        plt.plot(velocities, label='Angular Velocity (rad/s)')
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
