import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import unittest
import os
import multiprocessing
import traceback

from ppo import PPO

params = {
    'robot': 'inverted_pendulum',
    'state_dim': 3,
    'action_dim': 1,
    'max_action': 2.0,
    'min_action': -2.0,
    'clip_param': 0.2,
    'value_clip_param': 0.2,
    'value_loss_coef': 0.5,  
    'entropy_coef': 0.005,    
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'ppo_epochs': 5,         
    'batch_size': 64,      
    'max_grad_norm': 0.5,
    'max_episodes': 10,
    'actor_lr': 3e-4,       
    'critic_lr': 1e-3,       
    'buffer_size': 10000,
    'max_state_value': 1e4,
    'min_state_value': -1e4,
    'n_steps': 256,
    'update_lr': True,
    'convergence_threshold': 0.25,
    'critic_convergence_threshold': 0.1,
    'returns_window_size': 20,
    'value_loss_window_size': 20,
    'checkpoint_dir': 'policy/inverted_pendulum/ppo',
    'total_steps': 100000, 
    'final_lr_ratio': 0.01,  # Learning rate will decay to 1% of initial value
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
        # Scale rewards to reasonable range
        angle_penalty = theta**2
        
        if angle_from_upright < 0.25:
            self.upright_steps += 1
        else:
            self.upright_steps = 0
        
        # Bonus for consecutive upright steps
        consecutive_bonus = self.upright_steps * 1.0  
        
        control_penalty = 0.001 * action**2  
        velocity_penalty = 0.1 * theta_dot**2  

        reward = -angle_penalty - control_penalty - velocity_penalty  + consecutive_bonus
            
        # Termination condition - only terminate for extreme angular velocities
        done = 0.0
        if abs(theta_dot) > self.max_angular_vel:
            done = 1.0

        return self.state, reward, done

    def compute_reward(self, theta, theta_dot, action):
        """Improved reward function for better learning"""
        # Primary reward: cosine of angle (1 when upright, -1 when inverted)
        angle_reward = np.cos(theta)
        
        # Stability bonus when near upright (within 30 degrees)
        if abs(theta) < np.pi/6:
            stability_bonus = 2.0 * np.exp(-abs(theta_dot))
        else:
            stability_bonus = 0.0
        
        # Penalties
        velocity_penalty = 0.01 * theta_dot**2
        control_penalty = 0.001 * action**2
        
        # Total reward
        reward = angle_reward + stability_bonus - velocity_penalty - control_penalty
        
        # Bonus for staying upright
        if abs(theta) < np.pi/12:  # Within 15 degrees
            reward += 1.0
            
        return reward

def collect_experience_worker(
    worker_id, params, shared_actor_state_dict, shared_critic_state_dict,
    shared_buffer, training_event, update_event, device
):
    """
    Worker process to collect experience from an InvertedPendulum environment and add to a shared buffer.
    """
    try:
        shared_network_worker = SharedNetwork(params['state_dim'])
        actor_worker = Actor(params, shared_network_worker).to(device)
        critic_worker = Critic(params, shared_network_worker).to(device)
        agent_worker = PPO(actor_worker, critic_worker, params, device=device, normalize_state=False)
        agent_worker.train()
    except Exception as e:
        print(f"[Worker {worker_id}] Traceback: {traceback.format_exc()}", flush=True)
        raise

    # Create environment
    env = InvertedPendulum()
    max_steps_per_episode = 2048
    max_episodes = params['max_episodes']
    print(f"[Worker {worker_id}] Starting experience collection.")
    
    best_return = float('-inf')
    for episode in range(max_episodes):
        state = env.reset()
        episode_return = 0.0
        
        # Run episode
        for step in range(max_steps_per_episode):
            update_event.wait(timeout=10)
            
            action, log_prob = agent_worker.actor.get_action(state, deterministic=False)
            value = agent_worker.critic(torch.FloatTensor(state).reshape(1, -1).to(device)).item()
            next_state, reward, done = env.step(action)
            
            try:
                # Add experience to the shared buffer
                shared_buffer.append((state, action, reward, next_state, done, value, log_prob))
                episode_return = episode_return * params['gamma'] + reward
            except Exception as e:
                print(f"[Worker {worker_id}] Error appending to shared buffer: {str(e)}")
                print(f"[Worker {worker_id}] Traceback: {traceback.format_exc()}")
                continue

            # Check if enough steps are collected for training
            if len(shared_buffer) >= params['n_steps']:
                update_event.wait(timeout=10)
                update_event.clear()
                training_event.set()
                training_event.wait(timeout=60)
                # Load the updated policy
                actor_worker.load_state_dict(dict(shared_actor_state_dict))
                critic_worker.load_state_dict(dict(shared_critic_state_dict))
                # Set update_event to allow other workers to proceed
                update_event.set()

            if done > 0.0 or step == max_steps_per_episode - 1:
                state = env.reset()
                break
            else:
                state = next_state

            # Print progress
            if step % 1000 == 0 or step == max_steps_per_episode - 1:
                print(f"[Worker {worker_id}] -[{episode}/{max_episodes}] - [{step}/{max_steps_per_episode}] "
                      f"Current return: {float(episode_return):.2f} Best return: {float(best_return):.2f}")
        
        # At the end of an episode, check for new best return
        agent_worker.logger.log_episode(episode_return, step)
        if episode_return > best_return:
            best_return = episode_return

def train_ppo_parallel(agent, params, num_workers=4):
    """
    Parallel training function for PPO with InvertedPendulum environment.
    """
    # Set to train mode
    agent.train()

    # Create a manager for shared data structures
    manager = multiprocessing.Manager()
    shared_buffer = manager.list()  # Use a managed list for the replay buffer
    shared_buffer._timeout = 5.0  # Add a 5-second timeout for operations

    # Use managed dictionaries for shared model state_dicts
    shared_actor_state_dict = manager.dict(agent.actor.state_dict())
    shared_critic_state_dict = manager.dict(agent.critic.state_dict())

    # Events for synchronization
    training_event = manager.Event() # Set by workers when buffer is full, cleared by main after training
    update_event = manager.Event()   # Set by main after model update, cleared by main before training

    # Start workers for parallel data collection
    processes = []
    print(f"[Main process] Starting {num_workers} worker processes...")
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=collect_experience_worker,
            args=(i, params, shared_actor_state_dict, shared_critic_state_dict,
                  shared_buffer, training_event, update_event, agent.device)
        )
        processes.append(p)
        p.start()
    print(f"[Main process] All {num_workers} worker processes started")

    # Initial signal to workers to start collecting
    update_event.set()

    # Check if at least one worker is still running
    while any(p.is_alive() for p in processes):
        try:
            # Main process waits until enough data is collected
            training_event.wait(timeout=5.0) # Wait for a worker to signal that buffer is full
            buffer_size = len(shared_buffer)
            
            if buffer_size < params['n_steps']:
                training_event.clear()
                update_event.set()
                continue

            # Clear training_event to prevent other workers from proceeding
            training_event.clear()

            # Safely get the collected experiences
            try:
                collected_experiences = list(shared_buffer)
                shared_buffer[:] = [] # Clear the shared buffer
            except Exception as e:
                print(f"[Main process] Error accessing shared buffer: {str(e)}")
                print(f"[Main process] Traceback: {traceback.format_exc()}")
                training_event.set()  # Allow workers to continue
                continue

            # Add collected experiences to the agent's internal buffer
            for exp in collected_experiences:
                agent.add_to_buffer(*exp) # Unpack the tuple

            actor_loss, critic_loss, entropy, converged = agent.train()

            if actor_loss is not None and critic_loss is not None:
                print(f"[Main process] Policy Loss: {actor_loss:.4f}, Value Loss: {critic_loss:.4f}, "
                      f"Entropy: {entropy:.4f}")

            # Update shared model parameters
            shared_actor_state_dict.update(agent.actor.state_dict())
            shared_critic_state_dict.update(agent.critic.state_dict())

            if converged:
                break

            # Signal workers to resume collection by setting training_event
            training_event.set()
        except Exception as e:
            print(f"[Main process] Error in training loop: {str(e)}")
            print(f"[Main process] Traceback: {traceback.format_exc()}")
            training_event.set()  # Allow workers to continue
            continue

    # Terminate worker processes
    for p in processes:
        p.terminate()
        p.join()

    # Plot training curves
    agent.logger.plot_training_curves()
    return agent

# Unit tests for PPO with Inverted Pendulum
class TestPPOInvertedPendulum(unittest.TestCase):
    def setUp(self):
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.max_action = params['max_action']
        self.min_action = params['min_action']
        
        # Create device
        self.device = 'cpu'
        
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
        # Create checkpoint directory
        checkpoint_dir = 'policy/inverted_pendulum/ppo'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Update params with checkpoint and early stopping parameters
        params.update({
            'checkpoint_dir': checkpoint_dir
        })
        
        # Train using parallel implementation
        best_agent = train_ppo_parallel(self.agent, params, num_workers=6)
        
        # Evaluate the trained policy
        state = self.env.reset()
        total_reward = 0.0  
        states = []
        actions = []
        rewards = []
        max_steps_per_episode = 2048
        
        for step in range(max_steps_per_episode):
            action, _ = best_agent.actor.get_action(state, deterministic=True)
            next_state, reward, done = self.env.step(action)
            state_flat = np.array(state).reshape(-1)
            states.append(np.array([np.arctan2(state_flat[1], state_flat[0]),  state_flat[2]]))
            actions.append(action)
            rewards.append(reward) 
            total_reward = params['gamma'] * total_reward + reward
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
    multiprocessing.set_start_method('spawn', force=True)
    unittest.main()
