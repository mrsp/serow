#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import serow
import os
import psutil

from env import SerowEnv
from ddpg import DDPG

from utils import(
    Normalizer,
    export_models_to_onnx
)

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
        self.state_dim = params['state_dim']
        self.state_fb_dim = params['state_fb_dim']
        self.state_history_dim = params['state_history_dim']
        self.device = params['device']
        self.action_dim = params['action_dim']
        self.min_action = torch.FloatTensor(params['min_action']).to(self.device)
        self.max_action = torch.FloatTensor(params['max_action']).to(self.device)
        
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
        self.layer1 = nn.Linear(self.state_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        
        # Output layers
        self.mean_layer = nn.Linear(64, self.action_dim) 
        self._init_weights()
        
        # Exploration noise
        self.noise = OUNoise(params['action_dim'], sigma=0.1)
        self.noise_scale = params['noise_scale']
        self.noise_decay = params['noise_decay']

    def _init_weights(self):
        for layer in [self.conv1, self.layer1, self.layer2, self.layer3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)
    
    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)

        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        state_fb = state[:, :self.state_fb_dim]
        state_history = state[:, self.state_fb_dim:]
        state_history = state_history.unsqueeze(1)
        input0 = F.relu(self.conv1(state_history))
        input0 = input0.squeeze(1)
        input1 = torch.cat([state_fb, input0], dim=-1)
        x = F.relu(self.layer1(input1))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))  
        x = self.mean_layer(x)
        
        mean = torch.zeros_like(x, dtype=torch.float32)
        mean[:, :3] = F.softplus(x[:, :3])
        mean[:, 3:] = x[:, 3:]
        return mean
    
    def get_action(self, state, deterministic=False):
        mean = self.forward(state)

        if deterministic:
            action = mean
        else:
            noise = torch.FloatTensor(self.noise.sample() * self.noise_scale).to(self.device)
            action = mean + noise
            self.noise_scale *= self.noise_decay

        action = torch.clamp(action, min=self.min_action, max=self.max_action)
        return action.squeeze(0).detach().cpu().numpy()
    
class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.state_dim = params['state_dim']
        self.state_fb_dim = params['state_fb_dim']
        self.state_history_dim = params['state_history_dim']
        self.device = params['device']
        self.state_layer = nn.Linear(self.state_dim, 256)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)

        self.action_layer = nn.Linear(params['action_dim'], 256)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 1)
        nn.init.orthogonal_(self.state_layer.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.action_layer.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer4.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.state_layer.bias, 0.0)
        torch.nn.init.constant_(self.action_layer.bias, 0.0)
        torch.nn.init.constant_(self.layer2.bias, 0.0)
        torch.nn.init.constant_(self.layer3.bias, 0.0)
        torch.nn.init.constant_(self.layer4.bias, 0.0)

    def forward(self, state, action):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        state_fb = state[:, :self.state_fb_dim]
        state_history = state[:, self.state_fb_dim:]
        state_history = state_history.unsqueeze(1)
        input0 = F.relu(self.conv1(state_history))
        input0 = input0.squeeze(1)
        input1 = torch.cat([state_fb, input0], dim=-1)
        s = F.relu(self.state_layer(input1))
        a = F.relu(self.action_layer(action))
        x = torch.cat([s, a], dim=-1) # Concatenate state and action
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        # No activation on final layer to allow negative values
        return self.layer4(x)

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024 / 1024  # Convert to GB

def train_ddpg(datasets, agent, params):
    # Set to train mode
    agent.train()

    converged = False
    best_return = float('-inf')
    best_return_episode = 0

    # Training loop with evaluation phases
    for i, dataset in enumerate(datasets):
        print(f"Training on dataset {i+1}/{len(datasets)}")

        # Get dataset measurements
        imu_measurements = dataset['imu']
        joint_measurements = dataset['joints']
        force_torque_measurements = dataset['ft']
        base_pose_ground_truth = dataset['base_pose_ground_truth']
        contact_states = dataset['contact_states']
        joint_states = dataset['joint_states']
        base_states = dataset['base_states']
        kinematics = dataset['kinematics']
        # Set proper limits on number of episodes
        max_episodes = params['max_episodes']
        max_steps = len(imu_measurements) - 1 

        # Warm-up phase: collect initial experiences without training
        warmup_episodes = 1
        serow_env = SerowEnv(robot, joint_states[0], base_states[0], contact_states[0],  
                             params['action_dim'], params['state_dim'], 
                             params['history_buffer_size'], params['state_normalizer'])
        for episode in range(max_episodes + warmup_episodes):
            serow_env.reset()
            # Episode tracking variables
            episode_return = 0.0
            collected_steps = 0

            # Run episode
            for time_step, (imu, joints, ft, gt, next_kin) in enumerate(zip(
                imu_measurements[:max_steps], 
                joint_measurements[:max_steps], 
                force_torque_measurements[:max_steps], 
                base_pose_ground_truth[:max_steps],
                kinematics[1:max_steps+1]
            )):
                contact_frames = serow_env.contact_frames

                # Run the predict step
                kin, prior_state = serow_env.predict_step(imu, joints, ft)

                # Run the update step
                done = 0.0
                post_state = prior_state
                for cf in contact_frames:
                    if (post_state.get_contact_position(cf) is not None and kin.contacts_status[cf] and next_kin.contacts_status[cf]):
                        # Compute the state
                        x = serow_env.compute_state(cf, post_state, kin)

                        # Compute the action
                        if episode < warmup_episodes:
                            action = np.zeros((params['action_dim'],))
                        else:
                            action = agent.get_action(x, deterministic=False)

                        # Run the update step
                        post_state, reward, done = serow_env.update_step(cf, kin, action, gt, time_step, max_steps)

                        # Compute the next state
                        next_x = serow_env.compute_state(cf, post_state, next_kin)

                        # Add to buffer
                        agent.add_to_buffer(x, action, reward, next_x, done)
                        # Accumulate rewards
                        collected_steps += 1
                        episode_return += reward
                    else:
                        action = np.zeros((serow_env.action_dim, 1), dtype=np.float64)
                        # Just run the update step
                        post_state, _, _ = serow_env.update_step(cf, kin, action, gt, time_step, max_steps)

                    if done:
                        break

                # Finish the update
                serow_env.finish_update(imu, kin)
                
                # Train policy periodically
                if collected_steps >= params['n_steps']:
                    actor_loss, critic_loss, converged = agent.learn()
                    if actor_loss is not None and critic_loss is not None:
                        print(f"[Episode {episode + 1}/{max_episodes}] Step [{time_step + 1}/{max_steps}] Policy Loss: {actor_loss}, Value Loss: {critic_loss}")
                    collected_steps = 0

                # Check for early termination due to filter divergence
                if done:
                    print(f"Episode {episode + 1} terminated early at step {time_step + 1}/{max_steps} due to " 
                          f"filter divergence")
                    break
                
            memory_usage = get_memory_usage()
            print(f"Episode {episode + 1}/{max_episodes}, Step {time_step + 1}/{max_steps}, " 
                  f"Episode return: {episode_return}, Best: {best_return}, " 
                  f"in episode {best_return_episode}, "
                  f"Memory: {memory_usage:.2f}GB, "
                  f"Normalization stats: {params['state_normalizer'].get_normalization_stats() if params['state_normalizer'] is not None else 'None'}")

            # End of episode processing
            if episode_return > best_return:
                best_return = episode_return
                best_return_episode = episode
                if params['state_normalizer'] is not None:
                    params['state_normalizer'].save_stats(agent.checkpoint_dir, '/state_normalizer')
                export_models_to_onnx(agent, robot, params, agent.checkpoint_dir)

            agent.logger.log_episode(episode_return, time_step)

            # Check for early stopping
            if converged:
                break

        if converged or episode == max_episodes - 1:
            try:
                agent.load_checkpoint(os.path.join(agent.checkpoint_dir, f'trained_policy_{robot}.pth'))
            except FileNotFoundError:
                print(f"No trained policy found for {robot}. Training new policy...")
            export_models_to_onnx(agent, robot, params, agent.checkpoint_dir)
            if params['state_normalizer'] is not None:
                params['state_normalizer'].save_stats(agent.checkpoint_dir, '/state_normalizer')
            break  # Break out of dataset loop
    
    # Plot training curves
    agent.logger.plot_training_curves()
    return agent

if __name__ == "__main__":
    # Load and preprocess the data
    robot = "go2"
    dataset = np.load(f"{robot}_training_dataset.npz", allow_pickle=True)
    test_dataset = dataset
    imu_measurements = dataset['imu']
    contact_states = dataset['contact_states']
    dt = dataset['dt']
    dataset_size = len(imu_measurements) - 1

    # Define the dimensions of your state and action spaces
    normalizer = None
    history_buffer_size = 100
    print(f"History buffer size: {history_buffer_size * dt} seconds")
    state_history_dim = 3 * 3 * history_buffer_size + 3 * history_buffer_size
    state_fb_dim = 3 * 3 + 3 + 2 
    state_dim = state_fb_dim + state_history_dim  
    print(f"State dimension: {state_dim}")
    action_dim = 6  # Based on the action vector used in ContactEKF.setAction()
    min_action = np.array([1e-8, 1e-8, 1e-8, -1e2, -1e2, -1e2])
    max_action = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2])

    # Create the evaluation environment and get the contacts frames
    serow_env = SerowEnv(robot, dataset['joint_states'][0], dataset['base_states'][0], 
                         dataset['contact_states'][0], action_dim, state_dim, 
                         history_buffer_size, normalizer)
    contact_frames = serow_env.contact_frames
    print(f"Contacts frame: {contact_frames}")
    train_datasets = [dataset]
    device = 'cpu'

    max_episodes = 120
    n_steps = 1024
    total_steps = max_episodes * dataset_size * len(contact_frames)
    total_training_steps = total_steps // n_steps
    print(f"Total training steps: {total_training_steps}")

    params = {
        'history_buffer_size': history_buffer_size,
        'device': device,
        'robot': robot,
        'state_dim': state_dim,
        'state_fb_dim': state_fb_dim,
        'state_history_dim': state_history_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'min_action': min_action,
        'gamma': 0.99,
        'batch_size': 256,
        'max_grad_norm': 0.5,
        'tau': 0.005,
        'buffer_size': 100000,
        'max_episodes': max_episodes,
        'actor_lr': 1e-4,
        'critic_lr': 1e-4,
        'noise_scale': 0.5,
        'noise_decay': 0.9999,
        'n_steps': n_steps,
        'train_for_batches': 5,
        'convergence_threshold': 0.25,
        'critic_convergence_threshold': 0.15,
        'return_window_size': 20,
        'value_loss_window_size': 20,
        'checkpoint_dir': 'policy/ddpg',
        'total_steps': total_steps,
        'final_lr_ratio': 0.01,
        'check_value_loss': False,
        'total_training_steps': total_training_steps,
        'state_normalizer': normalizer,
    }

    loaded = False
    print(f"Initializing agent for {robot}")
    actor = Actor(params).to(device)
    critic = Critic(params).to(device)
    normalize_state = False
    agent = DDPG(actor, critic, params, device=device)

    policy_path = params['checkpoint_dir']
    print(f"Policy path: {policy_path}")
    # Try to load a trained policy for this robot if it exists
    try:
        agent.load_checkpoint(f'{policy_path}/trained_policy_{robot}.pth')
        if params['state_normalizer'] is not None:
            params['state_normalizer'].load_stats(policy_path, '/state_normalizer')
            print(f"Stats loaded: {params['state_normalizer'].get_normalization_stats()}")
        print(f"Loaded trained policy for {robot} from '{policy_path}/trained_policy_{robot}.pth'")
        loaded = True
    except FileNotFoundError:
        print(f"No trained policy found for {robot}. Training new policy...")

    if not loaded:
        # Train the policy
        if params['state_normalizer'] is not None:
            params['state_normalizer'].reset_stats()
        train_ddpg(train_datasets, agent, params)
        # Load the best policy
        checkpoint = torch.load(f'{policy_path}/trained_policy_{robot}.pth')
        actor = Actor(params).to(device)
        critic = Critic(params).to(device)
        best_policy = DDPG(actor, critic, params, device=device)
        best_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        best_policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded optimal trained policy for {robot} from " 
              f"'{policy_path}/trained_policy_{robot}.pth'")
        serow_env.evaluate(test_dataset, agent)
    else:
        # Just evaluate the loaded policy
        serow_env.evaluate(test_dataset, agent)
