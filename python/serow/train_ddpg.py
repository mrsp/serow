#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import serow
import os

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
        self.mean_layer = nn.Linear(32, params['action_dim'])
        
        # Initialize weights with smaller values to prevent large outputs
        nn.init.xavier_uniform_(self.layer1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.layer2.weight, gain=0.1)
        
        # Initialize mean layer to output 1.0 for each action dimension
        nn.init.zeros_(self.mean_layer.weight)
        nn.init.constant_(self.mean_layer.bias, 1.0)  # This will make softplus(1.0) ≈ 1.0
        
        
        # Initialize biases to small positive values
        nn.init.constant_(self.layer1.bias, 0.1)
        nn.init.constant_(self.layer2.bias, 0.1)
        
        self.min_action = params['min_action']
        self.action_dim = params['action_dim']

        self.noise = OUNoise(params['action_dim'], sigma=1.0)
        self.noise_scale = params['noise_scale']
        self.noise_decay = params['noise_decay']

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return F.softplus(self.mean_layer(x)) 

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(next(self.parameters()).device)
            action = self.forward(state).cpu().numpy()[0]

            if not deterministic:
                action = action + self.noise.sample() * self.noise_scale
                self.noise_scale *= self.noise_decay

            action = np.where(action < self.min_action, self.min_action, action)
            return action

class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(params['state_dim'] + params['action_dim'], 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
def train_policy(datasets, contacts_frame, agent, robot, save_policy=True):
    episode_rewards = []
    collected_steps = 0
    converged = False
    best_rewards = float('-inf')
    reward_history = []  # Track recent rewards for convergence check
    window_size = 10  # Size of window for moving averages
    reward_threshold = 0.01  # Minimum change in reward to be considered improvement

    for i, dataset in enumerate(datasets):
        # Get the measurements and the ground truth
        imu_measurements = dataset['imu']
        joint_measurements = dataset['joints']
        force_torque_measurements = dataset['ft']
        base_pose_ground_truth = dataset['base_pose_ground_truth']
        # contact_states = dataset['contact_states']

        # Reset to initial state
        # initial_base_state = dataset['base_states'][0]
        # initial_contact_state = dataset['contact_states'][0]
        # initial_joint_state = dataset['joint_states'][0]

        max_episodes = 250
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

            episode_reward = 0.0
            for step, (imu, joints, ft, gt) in enumerate(zip(imu_measurements, 
                                                             joint_measurements, 
                                                             force_torque_measurements, 
                                                             base_pose_ground_truth)):
                _, state, rewards, done = run_step(imu, joints, ft, gt, serow_framework, state, agent)

                # Accumulate the rewards
                for reward in rewards.values():
                    if reward is not None:
                        episode_reward += reward
                        collected_steps += 1

                # Train policy if we've collected enough steps
                if collected_steps >= update_steps:
                    agent.train()
                    collected_steps = 0

                for cf in contacts_frame:
                    if done[cf] > 0.5:
                        break

                if step == max_steps - 1:
                    print(f"Episode {episode} completed without diverging the filter with reward {episode_reward}")

                if step % 5000 == 0 or step == max_steps - 1:  # Print progress 
                    print(f"Episode {episode}/{max_episodes}, Step {step}/{max_steps - 1}, Reward: {episode_reward:.2f}, Best Reward: {best_rewards:.2f}")
            
            # Update reward histories and check for convergence
            episode_rewards.append(episode_reward)
            reward_history.append(episode_reward)
            if len(reward_history) > window_size:
                reward_history.pop(0)

            if (episode_reward > best_rewards):
                best_rewards = episode_reward
                # Save the trained policy for each contact frame
                if save_policy:
                    # Create policy directory if it doesn't exist
                    os.makedirs('policy/ddpg', exist_ok=True)
                        
                    torch.save({
                        'actor_state_dict': agent.actor.state_dict(),
                        'critic_state_dict': agent.critic.state_dict(),
                        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                    }, f'policy/ddpg/trained_policy_{robot}.pth')
                    print(f"Saved policy for {robot} to 'policy/ddpg/trained_policy_{robot}.pth'")

            print(f"Dataset {i}, Episode {episode}/{max_episodes}, Reward: {episode_reward:.2f}, Best Reward: {best_rewards:.2f}")

            # Check reward convergence
            if len(reward_history) >= window_size:
                recent_rewards = np.array(reward_history)
                reward_std = np.std(recent_rewards)
                reward_mean = np.mean(recent_rewards)
                reward_cv = reward_std / (abs(reward_mean) + 1e-6)  # Coefficient of variation
                
                # Check if reward has converged
                if reward_cv < reward_threshold:
                    print(f"Reward has converged!")
                    converged = True
            
            # If agent has converged, stop training
            if converged:
                print(f"Training converged at episode {episode}")
                break

    # Convert episode_rewards to numpy arrays and compute a smoothed reward curve using a low pass filter
    smoothed_episode_rewards = []
    episode_rewards = np.array(episode_rewards)
    smoothed_episode_reward = episode_rewards[0]
    alpha = 0.55
    for i in range(len(episode_rewards)):
        smoothed_episode_reward = alpha * smoothed_episode_reward + (1.0 - alpha) * episode_rewards[i]
        smoothed_episode_rewards.append(smoothed_episode_reward)

    # Create a single figure for all rewards
    plt.figure(figsize=(10, 6))
    
    # Plot the cummulative rewards
    plt.plot(episode_rewards, label='Episode Rewards', alpha=0.5)
    plt.plot(smoothed_episode_rewards, label='Smoothed Rewards', linewidth=2)
    plt.fill_between(range(len(episode_rewards)), 
                    episode_rewards - np.std(episode_rewards), 
                    episode_rewards + np.std(episode_rewards), 
                    alpha=0.1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards for All Contact Frames')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_policy(dataset, contacts_frame, agents, robot):
        # After training, evaluate the policy
        print(f"\nEvaluating trained DDPG policy for {robot}...")
        
        # Get the measurements and the ground truth
        imu_measurements = dataset['imu']
        joint_measurements = dataset['joints']
        force_torque_measurements = dataset['ft']
        base_pose_ground_truth = dataset['base_pose_ground_truth']

        # Reset to initial state
        # initial_base_state = dataset['base_states'][0]
        # initial_contact_state = dataset['contact_states'][0]
        # initial_joint_state = dataset['joint_states'][0]

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

        for step, (imu, joints, ft, gt) in enumerate(zip(imu_measurements, 
                                                         joint_measurements, 
                                                         force_torque_measurements, 
                                                         base_pose_ground_truth)):
            print("-------------------------------------------------")
            print(f"Evaluating DDPGpolicy for {robot} at step {step}")
            timestamp, state, rewards, _ = run_step(imu, joints, ft, gt, serow_framework, state, agent, deterministic=True)
            
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
        print("\n DDPG Policy Evaluation Metrics:")
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
    state_dim = 4  
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    min_action = 1e-6

    params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': None,
        'min_action': min_action,
        'gamma': 0.99,
        'tau': 0.01,
        'batch_size': 64,  
        'actor_lr': 5e-3, 
        'critic_lr': 1e-3,
        'noise_scale': 2.0,
        'noise_decay': 0.995,
        'buffer_size': 1000000,
    }

    # Initialize the actor and critic
    agents = {}
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    loaded = False
    robot = "go2"
    print(f"Initializing agent for {robot}")
    actor = Actor(params).to(device)
    critic = Critic(params).to(device)
    agent = DDPG(actor, critic, params, device=device)

    # Try to load a trained policy for this robot if it exists
    try:
        checkpoint = torch.load(f'policy/ddpg/trained_policy_{robot}.pth', weights_only=True)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded trained policy for {robot} from 'policy/ddpg/trained_policy_{robot}.pth'")
    except FileNotFoundError:
        print(f"No trained policy found for {robot}. Training new policy...")

    if not loaded:
        # Train the policy
        train_policy(train_datasets, contacts_frame, agent, robot, save_policy=True)
        # Load the best policy
        checkpoint = torch.load(f'policy/ddpg/trained_policy_{robot}.pth')
        actor = Actor(params).to(device)
        critic = Critic(params).to(device)
        best_policy = DDPG(actor, critic, params, device=device)
        best_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        best_policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded optimal trained policy for {robot} from 'policy/ddpg/trained_policy_{robot}.pth'")
        evaluate_policy(test_dataset, contacts_frame, best_policy, robot)
    else:
        # Just evaluate the loaded policy
        evaluate_policy(test_dataset, contacts_frame, agents, robot)
