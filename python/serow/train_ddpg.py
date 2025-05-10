#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import serow
from ddpg import DDPG
from read_mcap import(
    read_base_states, 
    read_contact_states, 
    read_force_torque_measurements, 
    read_joint_measurements, 
    read_joint_states,
    read_imu_measurements, 
    read_base_pose_ground_truth,
    run_step,
    plot_trajectories,
    sync_and_align_data
)

# Actor network per leg end-effector
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
        self.layer1 = nn.Linear(params['state_dim'], 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, params['action_dim'])
        
        # Initialize the output layer with a much wider range
        nn.init.xavier_uniform_(self.output_layer.weight, gain=1.4141) # sqrt(2)
        nn.init.uniform_(self.output_layer.bias, -0.1, 0.1)  # Non-zero bias
        
        self.max_action = params['max_action']
        self.min_action = params['min_action']
        self.action_dim = params['action_dim']

        self.noise = OUNoise(params['action_dim'], sigma=1.0)
        self.noise_scale = params['noise_scale']
        self.noise_decay = params['noise_decay']

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.output_layer(x)
        # Use tanh instead of sigmoid to get a wider range of outputs
        return torch.tanh(x) * (self.max_action - self.min_action) / 2.0 + (self.max_action + self.min_action) / 2.0

    # def get_action(self, state, deterministic=False):
    #     with torch.no_grad():
    #         state = torch.FloatTensor(state).reshape(1, -1).to(next(self.parameters()).device)
    #         action = self.forward(state).cpu().numpy()[0]

    #         if not deterministic:
    #             noise = self.noise.sample() * self.noise_scale
    #             action = action + noise
    #             self.noise_scale *= self.noise_decay
    #             action = np.clip(action, self.min_action, self.max_action)
            
    #         return action
    
    # epsilon-greedy
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(next(self.parameters()).device)
            action = self.forward(state).cpu().numpy()[0]
            if not deterministic and np.random.rand() < 0.1:  # 10% chance of random action
                action = np.random.uniform(self.min_action, self.max_action, self.action_dim)
            elif not deterministic:
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
    
def train_policy(datasets, contacts_frame, agents):
    episode_rewards = {}
    collected_steps = {}
    best_rewards = {}
    no_improvement_count = {}
    patience = 10  # Number of episodes to wait for improvement
    min_delta = 1.0  # Minimum improvement in reward to be considered progress

    for cf in contacts_frame:
        episode_rewards[cf] = []
        collected_steps[cf] = 0
        best_rewards[cf] = float('-inf')
        no_improvement_count[cf] = 0

    for i, dataset in enumerate(datasets):
        # Get the measurements and the ground truth
        imu_measurements = dataset['imu']
        joint_measurements = dataset['joints']
        force_torque_measurements = dataset['ft']
        base_pose_ground_truth = dataset['base_pose_ground_truth']
        
        # Reset to initial state
        initial_base_state = dataset['base_states'][0]
        initial_contact_state = dataset['contact_states'][0]
        initial_joint_state = dataset['joint_states'][0]

        max_episode = 1000
        update_steps = 32  # Train after collecting this many timesteps

        for episode in range(max_episode):
            serow_framework = serow.Serow()
            serow_framework.initialize("go2_rl.json")
            state = serow_framework.get_state(allow_invalid=True)

            # Initialize the state
            # state.set_base_state(initial_base_state)
            # state.set_contact_state(initial_contact_state)
            # state.set_joint_state(initial_joint_state)
            # serow_framework.set_state(state)

            episode_reward = {}
            for cf in contacts_frame:
                episode_reward[cf] = 0.0
            
            for step, (imu, joints, ft, gt) in enumerate(zip(imu_measurements, 
                                                             joint_measurements, 
                                                             force_torque_measurements, 
                                                             base_pose_ground_truth)):
                actions = {}
                x = np.concatenate([
                    state.get_base_position(),
                    state.get_base_orientation()
                ])

                for cf in state.get_contacts_frame():
                    actions[cf] = agents[cf].actor.get_action(x, deterministic=False)

                _, state, rewards = run_step(imu, joints, ft, gt, serow_framework, state, actions)

                 # Compute the next state
                next_x = np.concatenate([
                    state.get_base_position(),
                    state.get_base_orientation()
                ])

                # Add to buffer
                for cf in contacts_frame:
                    if rewards[cf] is not None:
                        episode_reward[cf] += rewards[cf]
                        agents[cf].add_to_buffer(x, actions[cf], rewards[cf], next_x, 0.0)
                        collected_steps[cf] += 1

                # Train policy if we've collected enough steps
                if collected_steps[cf] >= update_steps:
                    agents[cf].train()
                    collected_steps[cf] = 0
                
                if step % 5000 == 0:  # Print progress every 5000 steps
                    for cf in contacts_frame:
                        print(f"Episode {episode}, Step {step}, {cf} has Reward: {episode_reward[cf]:.2f}, Best Reward: {best_rewards[cf]:.2f}")
            
            # Early stopping check for each contact frame
            all_converged = True
            for cf in contacts_frame:   
                print(f"Dataset {i}, Episode {episode}, {cf} has Reward: {episode_reward[cf]:.2f}, Best Reward: {best_rewards[cf]:.2f}")
                episode_rewards[cf].append(episode_reward[cf])
                
                # Check if reward improved
                if episode_reward[cf] > best_rewards[cf] + min_delta:
                    best_rewards[cf] = episode_reward[cf]
                    no_improvement_count[cf] = 0
                else:
                    no_improvement_count[cf] += 1
                
                # Check if this contact frame has converged
                if no_improvement_count[cf] < patience:
                    all_converged = False
            
            # If all contact frames have converged, stop training
            if all_converged:
                print(f"Training converged at episode {episode} for all contact frames")
                break

    # Convert episode_rewards to numpy arrays and compute a smoothed reward curve using a low pass filter
    smoothed_episode_rewards = {}
    for cf in contacts_frame:
        episode_rewards[cf] = np.array(episode_rewards[cf])
        smoothed_episode_rewards[cf] = np.convolve(episode_rewards[cf], np.ones(10)/10, mode='valid')

    # Create a single figure with subplots for each contact frame
    n_cf = len(contacts_frame)
    fig, axes = plt.subplots(n_cf, 1, figsize=(10, 5*n_cf))
    if n_cf == 1:
        axes = [axes]  # Make axes iterable for single subplot case
    
    for ax, cf in zip(axes, contacts_frame):
        ax.plot(episode_rewards[cf], label='Episode Rewards')
        ax.plot(smoothed_episode_rewards[cf], label='Smoothed Rewards')
        ax.fill_between(range(len(episode_rewards[cf])), 
                       episode_rewards[cf] - np.std(episode_rewards[cf]), 
                       episode_rewards[cf] + np.std(episode_rewards[cf]), 
                       alpha=0.2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'Episode Rewards for {cf}')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_policy(dataset, contacts_frame, agents, save_policy=False):
        # After training, evaluate the policy
        print("\nEvaluating trained policy...")
        
        # Get the measurements and the ground truth
        imu_measurements = dataset['imu']
        joint_measurements = dataset['joints']
        force_torque_measurements = dataset['ft']
        base_pose_ground_truth = dataset['base_pose_ground_truth']

        # Reset to initial state
        initial_base_state = dataset['base_states'][0]
        initial_contact_state = dataset['contact_states'][0]
        initial_joint_state = dataset['joint_states'][0]

        # Initialize SEROW
        serow_framework = serow.Serow()
        serow_framework.initialize("go2_rl.json")
        state = serow_framework.get_state(allow_invalid=True)
        # state.set_base_state(initial_base_state)
        # state.set_contact_state(initial_contact_state)
        # state.set_joint_state(initial_joint_state)
        # serow_framework.set_state(state)

        # Run SEROW
        timestamps = []
        base_positions = []
        base_orientations = []
        cumulative_rewards = {}
        for cf in contacts_frame:
            cumulative_rewards[cf] = []

        gt_positions = []
        gt_orientations = []
        gt_timestamps = []

        for imu, joints, ft, gt in zip(imu_measurements, 
                                       joint_measurements, 
                                       force_torque_measurements, 
                                       base_pose_ground_truth):
            x = np.concatenate([
                state.get_base_position(),
                state.get_base_orientation()
            ])

            actions = {}
            print("-------------------------------------------------")
            for cf in contacts_frame:
                actions[cf] = agents[cf].actor.get_action(x, deterministic=True)
                print(f"Action for {cf}: {actions[cf]}")

            timestamp, state, rewards = run_step(imu, joints, ft, gt, serow_framework, state, actions)
            
            timestamps.append(timestamp)
            base_positions.append(state.get_base_position())
            base_orientations.append(state.get_base_orientation())
            gt_positions.append(gt.position)
            gt_orientations.append(gt.orientation)
            gt_timestamps.append(gt.timestamp)
            for cf in contacts_frame:
                if rewards[cf] is not None:
                    cumulative_rewards[cf].append(rewards[cf])

        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        base_positions = np.array(base_positions)
        base_orientations = np.array(base_orientations)
        gt_positions = np.array(gt_positions)
        gt_orientations = np.array(gt_orientations)
        cumulative_rewards = {cf: np.array(cumulative_rewards[cf]) for cf in contacts_frame}

        # Sync and align the data
        timestamps, base_positions, base_orientations, gt_positions, gt_orientations = sync_and_align_data(timestamps, base_positions, base_orientations, gt_timestamps, gt_positions, gt_orientations, align = False)

        # Plot the trajectories
        plot_trajectories(timestamps, base_positions, base_orientations, gt_positions, gt_orientations, cumulative_rewards)

        # Print evaluation metrics
        print("\nPolicy Evaluation Metrics:")
        for cf in contacts_frame:
            print(f"Average Cumulative Reward for {cf}: {np.mean(cumulative_rewards[cf]):.4f}")
            print(f"Max Cumulative Reward for {cf}: {np.max(cumulative_rewards[cf]):.4f}")
            print(f"Min Cumulative Reward for {cf}: {np.min(cumulative_rewards[cf]):.4f}")
            print("-------------------------------------------------")
        # Save the trained policy for each contact frame
        if save_policy:
            import os
            # Create policy directory if it doesn't exist
            os.makedirs('policy/ddpg', exist_ok=True)
            
            for cf in contacts_frame:
                torch.save({
                    'actor_state_dict': agents[cf].actor.state_dict(),
                    'critic_state_dict': agents[cf].critic.state_dict(),
                    'actor_optimizer_state_dict': agents[cf].actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agents[cf].critic_optimizer.state_dict(),
                }, f'policy/ddpg/trained_policy_{cf}.pth')
                print(f"Saved policy for {cf} to 'policy/ddpg/trained_policy_{cf}.pth'")

if __name__ == "__main__":
    # Read the data
    imu_measurements  = read_imu_measurements("/tmp/serow_measurements.mcap")
    joint_measurements = read_joint_measurements("/tmp/serow_measurements.mcap")
    force_torque_measurements = read_force_torque_measurements("/tmp/serow_measurements.mcap")
    base_pose_ground_truth = read_base_pose_ground_truth("/tmp/serow_measurements.mcap")
    base_states = read_base_states("/tmp/serow_proprioception.mcap")
    contact_states = read_contact_states("/tmp/serow_proprioception.mcap")
    joint_states = read_joint_states("/tmp/serow_proprioception.mcap")

    offset = len(imu_measurements) - len(base_states)
    imu_measurements = imu_measurements[offset:]
    joint_measurements = joint_measurements[offset:]
    force_torque_measurements = force_torque_measurements[offset:]
    base_pose_ground_truth = base_pose_ground_truth[offset:]

    # Dataset length
    dataset_length = len(imu_measurements)
    print(f"Dataset length: {dataset_length}")

    # Calculate the size of each dataset
    total_size = len(contact_states)
    print(f"Total size: {total_size}")

    # N = 10  # Number of datasets
    # dataset_size = total_size // N  # Size of each dataset

    # # Create N contiguous datasets
    # train_datasets = []
    # for i in range(N):
    #     start_idx = i * dataset_size
    #     end_idx = start_idx + dataset_size
        
    #     # Create a dataset with measurements and states from start_idx to end_idx
    #     dataset = {
    #         'imu': imu_measurements[start_idx:end_idx],
    #         'joints': joint_measurements[start_idx:end_idx],
    #         'ft': force_torque_measurements[start_idx:end_idx],
    #         'base_states': base_states[start_idx:end_idx],
    #         'contact_states': contact_states[start_idx:end_idx],
    #         'joint_states': joint_states[start_idx:end_idx],
    #         'base_pose_ground_truth': base_pose_ground_truth[start_idx:end_idx]
    #     }
    #     train_datasets.append(dataset)
    test_dataset = {
        'imu': imu_measurements,
        'joints': joint_measurements,
        'ft': force_torque_measurements,
        'base_states': base_states,
        'contact_states': contact_states,
        'joint_states': joint_states,
        'base_pose_ground_truth': base_pose_ground_truth
    }
    train_datasets = [test_dataset]

    # Get the contacts frame
    contacts_frame = set(contact_states[0].contacts_status.keys())
    print(f"Contacts frame: {contacts_frame}")

    # Define the dimensions of your state and action spaces
    state_dim = 7  # 3 position, 4 orientation
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    max_action = 1000
    min_action = 0.001

    params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'min_action': min_action,
        'gamma': 0.99,
        'tau': 0.05,
        'batch_size': 64,  
        'actor_lr': 5e-4, 
        'critic_lr': 1e-4,
        'noise_scale': 5.0,
        'noise_decay': 0.995,
        'buffer_size': 1000000,
    }

    # Initialize the actor and critic
    agents = {}
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    loaded = False
    for cf in contacts_frame:
        print(f"Initializing agent for {cf}")
        actor = Actor(params).to(device)
        critic = Critic(params).to(device)
        agents[cf] = DDPG(actor, critic, params, device=device)

    # Try to load a trained policy for this contact frame if it exists
    try:
        for cf in contacts_frame:
            checkpoint = torch.load(f'policy/ddpg/trained_policy_{cf}.pth', weights_only=True)
            try:
                agents[cf].actor.load_state_dict(checkpoint['actor_state_dict'])
                agents[cf].critic.load_state_dict(checkpoint['critic_state_dict'])
                print(f"Loaded trained policy for {cf} from 'policy/ddpg/trained_policy_{cf}.pth'")
                loaded = True
            except RuntimeError as e:
                print(f"Could not load existing model for {cf} due to architecture mismatch. Starting with new model.")
                print(f"Error details: {str(e)}")
                loaded = False
    except FileNotFoundError:
        print(f"No trained policy found. Training new policy...")
        loaded = False

    if not loaded:
        # Train the policy
        train_policy(train_datasets, contacts_frame, agents)
        evaluate_policy(test_dataset, contacts_frame, agents, save_policy=True)
    else:
        # Just evaluate the loaded policy
        evaluate_policy(test_dataset, contacts_frame, agents, save_policy=False)
