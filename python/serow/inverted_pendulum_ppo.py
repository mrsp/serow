import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import unittest
import os

from ppo import PPO

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
    'entropy_coef': 0.01,    
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'ppo_epochs': 5,         
    'batch_size': 64,      
    'max_grad_norm': 0.5,
    'max_episodes': 200,
    'actor_lr': 3e-4,       
    'critic_lr': 5e-4,       
    'buffer_size': 10000,
    'max_state_value': 1e4,
    'min_state_value': -1e4,
    'n_steps': 256,
    'convergence_threshold': 0.25,
    'critic_convergence_threshold': 0.15,
    'returns_window_size': 20,
    'value_loss_window_size': 20,
    'checkpoint_dir': 'policy/inverted_pendulum/ppo',
    'total_training_steps': 100000,
    'total_steps': 100000, 
    'final_lr_ratio': 0.01,  # Learning rate will decay to 1% of initial value
    'check_value_loss': False,
    'target_kl': 0.03,
}

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
        self.log_std = nn.Parameter(torch.zeros(params['action_dim']))
        
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

    def forward(self, state):
        x = self.layer1(state)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        mean = self.mean_layer(x)
        # Clamp log_std for numerical stability
        log_std = self.log_std.clamp(-20, 2)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        mean, log_std = self.forward(torch.FloatTensor(state).reshape(1, -1).to(self.device))
        std = log_std.exp()

        if deterministic:
            action = mean
            log_prob = torch.zeros(1)
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1)
            
        action = torch.clamp(action, self.min_action, self.max_action)
        return action.detach().cpu().numpy()[0], log_prob.detach().cpu().item()
    
    def evaluate_actions(self, states, actions):
        """Evaluate log probabilities and entropy for given state-action pairs"""
        mean, log_std = self.forward(states)
        std = log_std.exp()
        
        # Calculate log probabilities
        normal = torch.distributions.Normal(mean, std)
        log_probs = normal.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # Calculate entropy
        entropy = normal.entropy().sum(dim=-1)
        return log_probs.squeeze(-1), entropy.squeeze(-1)
    
class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(params['state_dim'], 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)
        nn.init.orthogonal_(self.layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer3.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.layer1.bias, 0.0)
        torch.nn.init.constant_(self.layer2.bias, 0.0)
        torch.nn.init.constant_(self.layer3.bias, 0.0)

    def forward(self, state):
        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
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

# Unit tests for PPO with Inverted Pendulum
class TestPPOInvertedPendulum(unittest.TestCase):
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
        # stored[0] and stored[3] are the original state and next_state (not flattened)
        self.assertTrue(np.allclose(stored[0], state, rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[1], action, rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[2], reward, rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(stored[3], next_state, rtol=1e-5, atol=1e-5))
        self.assertEqual(stored[4], done)
        self.assertEqual(stored[5], value)
        self.assertEqual(stored[6], log_prob)

    def test_train_and_evaluate(self):   
        max_steps_per_episode = 2500
        checkpoint_dir = 'policy/inverted_pendulum/ppo'
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
        
            converged = False
            for episode in range(max_episodes):
                state = self.env.reset()
                episode_return = 0.0
                for step in range(max_steps_per_episode):
                    action, log_prob = self.agent.actor.get_action(state, deterministic=False)
                    value = self.agent.critic(torch.FloatTensor(state).reshape(1, -1).to(self.device)).item()
                    next_state, reward, done = self.env.step(action)
                    self.agent.add_to_buffer(state, action, reward, next_state, done, value, log_prob)
                    
                    episode_return += reward
                    collected_steps += 1
                    
                    if collected_steps >= params['n_steps']:
                        policy_loss, critic_loss, entropy, converged = self.agent.learn()
                        collected_steps = 0
                        if policy_loss != 0.0 or critic_loss != 0.0:  # Only print if actual training occurred
                            print(f"Policy Loss: {policy_loss}, Value Loss: {critic_loss}, Entropy: {entropy}")

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
                    
                print(f"Episode {episode + 1}/{max_episodes}, Steps: {step + 1}/{max_steps_per_episode}, Return: {episode_return}, Best Return: " 
                    f"{best_return}")

                if episode % 10 == 0:
                    avg_return = np.mean(episode_returns[-10:])
                    print(f"Episode {episode + 1}, Avg Return (last 10): {avg_return}")
            
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
            action, _ = self.agent.actor.get_action(state, deterministic=True)
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
