#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import serow
import os

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
    sync_and_align_data,
    quaternion_to_rotation_matrix
)

# Actor network per leg end-effector
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.mean_layer = nn.Linear(32, action_dim)
        self.log_std_layer = nn.Linear(32, action_dim)
        self.min_action = min_action
        
        # Initialize weights with smaller values to prevent large outputs
        nn.init.xavier_uniform_(self.layer1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.layer2.weight, gain=0.1)
        nn.init.xavier_uniform_(self.mean_layer.weight, gain=0.01)
        nn.init.xavier_uniform_(self.log_std_layer.weight, gain=0.01)
        
        # Initialize biases to small positive values
        nn.init.constant_(self.layer1.bias, 0.1)
        nn.init.constant_(self.layer2.bias, 0.1)
        nn.init.constant_(self.mean_layer.bias, 0.1)
        nn.init.constant_(self.log_std_layer.bias, -1.0)  # Start with small std

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        mean = F.softplus(self.mean_layer(x)) 
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=0.5)  # Prevent extreme values
        return mean, log_std

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).reshape(1, -1).to(next(self.parameters()).device)
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = torch.tensor(0.0)
        else:
            std = log_std.exp()
            std = torch.clamp(std, min=1e-6, max=1.0)  # Prevent too small or large std
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
            action = z
            log_prob = normal.log_prob(z)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = torch.where(action < 0.0, torch.tensor(self.min_action, device=action.device), action)
        action = action.detach().cpu().numpy()[0]
        log_prob = log_prob.detach().cpu().item()
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def train_policy(datasets, contacts_frame, agent, robot, save_policy=True):
    episode_rewards = {}
    collected_steps = 0
    converged = {}
    best_rewards = {}
    reward_history = {}  # Track recent rewards for convergence check
    window_size = 10  # Size of window for moving averages
    reward_threshold = 0.01  # Minimum change in reward to be considered improvement
    reward_scale = 0.001  # Scale rewards to prevent extreme values

    for cf in contacts_frame:
        episode_rewards[cf] = []
        best_rewards[cf] = float('-inf')
        reward_history[cf] = []  
        converged[cf] = False
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

        max_episodes = 500
        update_steps = 64  # Train after collecting this many timesteps
        max_steps = len(imu_measurements)

        for episode in range(max_episodes):
            serow_framework = serow.Serow()
            serow_framework.initialize(f"{robot}_rl.json")
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
                x = {}
                for cf in state.get_contacts_frame():
                    contact_status[cf] = state.get_contact_status(cf)
                    if contact_status[cf]:
                        x[cf] = np.concatenate((np.abs(state.get_base_position() - state.get_contact_position(cf)), state.get_base_orientation()))
                        actions[cf], log_probs[cf] = agent.actor.get_action(x[cf], deterministic=False)
                        values[cf] = agent.critic(torch.FloatTensor(x[cf]).reshape(1, -1).to(next(agent.critic.parameters()).device)).item()
                    else:
                        x[cf] = None
                        actions[cf] = None
                        log_probs[cf] = None
                        values[cf] = None

                _, state, rewards, done = run_step(imu, joints, ft, gt, serow_framework, state, actions)

                # Add to buffer
                for cf in contacts_frame:
                    if state.get_contact_status(cf) and rewards[cf] is not None and x[cf] is not None:
                        if done:
                            rewards[cf] = -1e3  # Less extreme penalty
                        episode_reward[cf] += rewards[cf]
                        # Scale rewards before adding to buffer
                        scaled_reward = rewards[cf] * reward_scale
                        # Compute the next state
                        next_x = np.concatenate((np.abs(state.get_base_position() - state.get_contact_position(cf)), state.get_base_orientation()))
                        agent.add_to_buffer(x[cf], actions[cf], scaled_reward, next_x, done, values[cf], log_probs[cf])
                        collected_steps += 1
                        
                    # Train policy if we've collected enough steps
                    if not converged[cf] and collected_steps >= update_steps:
                        agent.train()
                        collected_steps = 0
                
                if done:
                    break
                
                if step % 5000 == 0 or step % (max_steps - 1) == 0:  # Print progress 
                    for cf in contacts_frame:
                        print(f"Episode {episode}/{max_episodes}, Step {step}/{max_steps - 1}, {cf} has Reward: {episode_reward[cf]:.2f}, Best Reward: {best_rewards[cf]:.2f}")
            
            episode_rewards[cf].append(episode_reward[cf])

            # Update reward histories
            for cf in contacts_frame:
                # Add reward to history
                reward_history[cf].append(episode_reward[cf])
                if len(reward_history[cf]) > window_size:
                    reward_history[cf].pop(0)

            # Check for convergence
            all_converged = False
            for cf in contacts_frame:   
                if (episode_reward[cf] > best_rewards[cf]):
                    best_rewards[cf] = episode_reward[cf]
                    # Save the trained policy for each contact frame
                    if save_policy:
                        # Create policy directory if it doesn't exist
                        os.makedirs('policy/ppo', exist_ok=True)
                        
                        torch.save({
                            'actor_state_dict': agent.actor.state_dict(),
                            'critic_state_dict': agent.critic.state_dict(),
                            'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                            'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                        }, f'policy/ppo/trained_policy_{robot}.pth')
                        print(f"Saved policy for {robot} to 'policy/ppo/trained_policy_{robot}.pth'")

                print(f"Dataset {i}, Episode {episode}/{max_episodes}, {cf} has Reward: {episode_reward[cf]:.2f}, Best Reward: {best_rewards[cf]:.2f}")
                episode_rewards[cf].append(episode_reward[cf])

                # Check reward convergence
                if len(reward_history[cf]) >= window_size:
                    recent_rewards = np.array(reward_history[cf])
                    reward_std = np.std(recent_rewards)
                    reward_mean = np.mean(recent_rewards)
                    reward_cv = reward_std / (abs(reward_mean) + 1e-6)  # Coefficient of variation
                    
                    # Check if reward has converged
                    if reward_cv < reward_threshold:
                        print(f"{cf} has converged!")
                        converged[cf] = True
                        all_converged = True
                    else:
                        all_converged = False
            
            # If all contact frames have converged, stop training
            if all_converged:
                print(f"Training converged at episode {episode} for all contact frames")
                break

    # Convert episode_rewards to numpy arrays and compute a smoothed reward curve using a low pass filter
    smoothed_episode_rewards = {}
    for cf in contacts_frame:
        episode_rewards[cf] = np.array(episode_rewards[cf])
        smoothed_episode_rewards[cf] = []
        smoothed_episode_reward = episode_rewards[cf][0]
        alpha = 0.7
        for i in range(len(episode_rewards[cf])):
            smoothed_episode_reward = alpha * smoothed_episode_reward + (1.0 - alpha) * episode_rewards[cf][i]
            smoothed_episode_rewards[cf].append(smoothed_episode_reward)

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

def evaluate_policy(dataset, contacts_frame, agent, robot):
        # After training, evaluate the policy
        print(f"\nEvaluating trained policy for {robot}...")
        
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
        serow_framework.initialize(f"{robot}_rl.json")
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
            actions = {}
            contact_status = {}
            print("-------------------------------------------------")
            for cf in contacts_frame:
                contact_status[cf] = state.get_contact_status(cf)
                if contact_status[cf]:
                    x = np.concatenate((np.abs(state.get_base_position() - state.get_contact_position(cf)), state.get_base_orientation()))
                    actions[cf], _ = agent.actor.get_action(x, deterministic=True)
                    print(f"Action for {cf}: {actions[cf]}")
                else:
                    actions[cf] = None
                    
            timestamp, state, rewards, _ = run_step(imu, joints, ft, gt, serow_framework, state, actions)
            
            timestamps.append(timestamp)
            base_positions.append(state.get_base_position())
            base_orientations.append(state.get_base_orientation())
            gt_positions.append(gt.position)
            gt_orientations.append(gt.orientation)
            gt_timestamps.append(gt.timestamp)
            for cf in contacts_frame:
                if contact_status[cf] and rewards[cf] is not None:
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
        print(f"\nPolicy Evaluation Metrics for {robot}:")
        for cf in contacts_frame:
            print(f"Average Cumulative Reward for {cf}: {np.mean(cumulative_rewards[cf]):.4f}")
            print(f"Max Cumulative Reward for {cf}: {np.max(cumulative_rewards[cf]):.4f}")
            print(f"Min Cumulative Reward for {cf}: {np.min(cumulative_rewards[cf]):.4f}")
            print("-------------------------------------------------")

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
    state_dim = 7 # 3 for position, 4 for orientation
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    min_action = 0.0001

    params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': None,
        'min_action': min_action,
        'clip_param': 0.1,  # More conservative clipping
        'value_loss_coef': 0.5,  # Reduce value loss coefficient
        'entropy_coef': 0.01,  # Reduce entropy coefficient
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_epochs': 5,  # Reduce number of epochs
        'batch_size': 32,  # Smaller batch size
        'max_grad_norm': 0.1,  # More conservative gradient clipping
        'actor_lr': 1e-4,  # Lower learning rate
        'critic_lr': 1e-4,  # Lower learning rate
        'buffer_size': 100000
    }

    # Initialize the actor and critic
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    loaded = False
    robot = "go2"
    print(f"Initializing agent for {robot}")
    actor = Actor(state_dim, action_dim, min_action).to(device)
    critic = Critic(state_dim).to(device)
    agent = PPO(actor, critic, params, device=device)

    # Try to load a trained policy for this contact frame if it exists
    try:
        checkpoint = torch.load(f'policy/ppo/trained_policy_{robot}.pth')
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded trained policy for {robot} from 'policy/ppo/trained_policy_{robot}.pth'")
        loaded = True
    except FileNotFoundError:
        print(f"No trained policy found for {robot}. Training new policy...")

    if not loaded:
        # Train the policy
        train_policy(train_datasets, contacts_frame, agent, robot, save_policy=True)
        # Load the best policy
        checkpoint = torch.load(f'policy/ppo/trained_policy_{robot}.pth')
        actor = Actor(state_dim, action_dim, min_action).to(device)
        critic = Critic(state_dim).to(device)
        best_policy = PPO(actor, critic, params, device=device)
        best_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        best_policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded optimal trained policy for {robot} from 'policy/ppo/trained_policy_{robot}.pth'")
        evaluate_policy(test_dataset, contacts_frame, best_policy, robot)
    else:
        # Just evaluate the loaded policy
        evaluate_policy(test_dataset, contacts_frame, agent, robot)
