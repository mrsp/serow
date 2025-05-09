#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import serow
from ppo import PPO
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
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_action):
        super(Actor, self).__init__()
        # Initialize layers with smaller weights
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Linear(64, action_dim)
        
        # Initialize weights with smaller values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
        
        self.max_action = max_action
        self.min_action = min_action

    def forward(self, state):
        # Use tanh for more stable gradients
        x = torch.tanh(self.layer1(state))
        x = torch.tanh(self.layer2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # Constrain log_std to a smaller range
        log_std = torch.clamp(log_std, min=-3, max=0.5)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).reshape(1, -1).to(next(self.parameters()).device)
        mean, log_std = self.forward(state)
        
        # Calculate the range once
        action_range = self.max_action - self.min_action

        if deterministic:
            # Scale to [min_action, max_action]
            action = torch.sigmoid(mean) * action_range + self.min_action
            log_prob = torch.tensor(0.0)
        else:
            std = log_std.exp()
            # Add small epsilon to std for numerical stability
            std = std + 1e-6
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
            
            # Transform to bounded range with tanh for more stable gradients
            tanh_z = torch.tanh(z)
            action = (tanh_z + 1.0) * 0.5 * action_range + self.min_action
            
            # Correct log prob adjustment for tanh
            log_prob = normal.log_prob(z) - torch.log(torch.clamp(1 - tanh_z.pow(2), min=1e-8))
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action.detach().cpu().numpy()[0], log_prob.detach().cpu().item()

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def train_policy(datasets, contacts_frame, agents):
    episode_rewards = {}
    collected_steps = {}

    for cf in contacts_frame:
        episode_rewards[cf] = []
        collected_steps[cf] = 0

    for i, dataset in enumerate(datasets):
        best_reward = {}
        for cf in contacts_frame:
            best_reward[cf] = float('-inf')

        # Get the measurements and the ground truth
        imu_measurements = dataset['imu']
        joint_measurements = dataset['joints']
        force_torque_measurements = dataset['ft']
        base_pose_ground_truth = dataset['base_pose_ground_truth']
        
        # Reset to initial state
        initial_base_state = dataset['base_states'][0]
        initial_contact_state = dataset['contact_states'][0]
        initial_joint_state = dataset['joint_states'][0]

        max_episode = 100
        update_steps = 64  # Train after collecting this many timesteps

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
                log_probs = {}
                values = {}
                contact_status = {}
                for cf in contacts_frame:
                    contact_status[cf] = state.get_contact_status(cf)

                x = np.concatenate([
                    state.get_base_position(),
                    state.get_base_orientation()
                ])

                for cf in state.get_contacts_frame():
                    if contact_status[cf]:
                        actions[cf], log_probs[cf] = agents[cf].actor.get_action(x, deterministic=False)
                        values[cf] = agents[cf].critic(torch.FloatTensor(x).reshape(1, -1).to(next(agents[cf].critic.parameters()).device)).item()

                _, state, rewards = run_step(imu, joints, ft, gt, serow_framework, state, actions)

                 # Compute the next state
                next_x = np.concatenate([
                    state.get_base_position(),
                    state.get_base_orientation()
                ])

                # Add to buffer
                for cf in contacts_frame:
                    if contact_status[cf] and rewards[cf] is not None:
                        episode_reward[cf] += rewards[cf]
                        agents[cf].add_to_buffer(x, actions[cf], rewards[cf], next_x, 0.0, values[cf], log_probs[cf])

                        # Train policy if we've collected enough steps
                        if collected_steps[cf] >= update_steps:
                            agents[cf].train()
                            collected_steps[cf] = 0
                
                if step % 5000 == 0:  # Print progress every 100 steps
                    for cf in contacts_frame:
                        print(f"Episode {episode}, Step {step}, {cf} has Reward: {episode_reward[cf]:.2f}, Best Reward: {best_reward[cf]:.2f}")
            
            for cf in contacts_frame:   
                print(f"Dataset {i}, Episode {episode}, {cf} has Reward: {episode_reward[cf]:.2f}, Best Reward: {best_reward[cf]:.2f}")
                episode_rewards[cf].append(episode_reward[cf])
                if episode_reward[cf] > best_reward[cf]:
                    best_reward[cf] = episode_reward[cf]

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
            for cf in contacts_frame:
                actions[cf], _ = agents[cf].actor.get_action(x, deterministic=True)

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
            for cf in contacts_frame:
                torch.save({
                    'actor_state_dict': agents[cf].actor.state_dict(),
                    'critic_state_dict': agents[cf].critic.state_dict(),
                    'actor_optimizer_state_dict': agents[cf].actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agents[cf].critic_optimizer.state_dict(),
                }, f'policy/trained_policy_{cf}.pth')
                print(f"Saved policy for {cf} to 'policy/trained_policy_{cf}.pth'")

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
    action_dim = 2  # Based on the action vector used in ContactEKF.setAction()
    max_action = 100.0 
    min_action = 0.01  

    params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'min_action': min_action,
        'clip_param': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_epochs': 5,
        'batch_size': 64,
        'max_grad_norm': 0.5,
        'actor_lr': 1e-4,
        'critic_lr': 1e-4,
    }

    # Initialize the actor and critic
    agents = {}
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    loaded = False
    for cf in contacts_frame:
        print(f"Initializing agent for {cf}")
        actor = Actor(state_dim, action_dim, max_action, min_action).to(device)
        critic = Critic(state_dim).to(device)
        agents[cf] = PPO(actor, critic, params, device=device)

    # Try to load a trained policy for this contact frame if it exists
    try:
        for cf in contacts_frame:
            checkpoint = torch.load(f'policy/trained_policy_{cf}.pth')
            agents[cf].actor.load_state_dict(checkpoint['actor_state_dict'])
            agents[cf].critic.load_state_dict(checkpoint['critic_state_dict'])
            print(f"Loaded trained policy for {cf} from 'policy/trained_policy_{cf}.pth'")
            loaded = True
    except FileNotFoundError:
        print(f"No trained policy found for {cf}. Training new policy...")
        loaded = False

    if not loaded:
        # Train the policy
        train_policy(train_datasets, contacts_frame, agents)
        evaluate_policy(test_dataset, contacts_frame, agents, save_policy=True)
    else:
        # Just evaluate the loaded policy
        evaluate_policy(test_dataset, contacts_frame, agents, save_policy=False)
