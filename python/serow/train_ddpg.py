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
    sync_and_align_data,
    quaternion_to_rotation_matrix
)

class Actor(nn.Module):
    def __init__(self, params):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(params['state_dim'], 64)
        self.layer2 = nn.Linear(64, 64)
        self.mean_layer = nn.Linear(64, params['action_dim'])
        
        # Proper initialization for approximating identity function initially
        nn.init.orthogonal_(self.layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer2.weight, gain=np.sqrt(2))

        # Initialize bias to produce an output close to 1.0
        nn.init.constant_(self.mean_layer.bias, 1.0) 
        
        self.min_action = params['min_action']
        self.action_dim = params['action_dim']
        self.log_noise_sigma = params['log_noise_sigma']
        self.min_log_noise_sigma = params['min_log_noise_sigma']

    def forward(self, state):
        x = F.leaky_relu(self.layer1(state))
        x = F.leaky_relu(self.layer2(x))
        return F.softplus(self.mean_layer(x)) + self.min_action

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(next(self.parameters()).device)
            action = self.forward(state).cpu().numpy()[0]
            if not deterministic:
                # 70% to apply noise
                if np.random.rand() < 0.7:
                    # Apply logarithmic space noise with the adjusted sigma
                    log_action = np.log(action)
                    log_noise = np.random.normal(0, self.log_noise_sigma, size=action.shape)
                    noisy_log_action = log_action + log_noise
                    
                    # Convert back to linear space
                    action = np.exp(noisy_log_action)

            # Ensure minimum action value and positivity
            action = np.maximum(action, self.min_action)
            return action

class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        # Process state first through separate network
        self.state_layer1 = nn.Linear(params['state_dim'], 64)
        self.state_layer2 = nn.Linear(64, 64)
        
        # Process action separately
        self.action_layer = nn.Linear(params['action_dim'], 64)
        
        # Combine state and action processing
        self.combined_layer1 = nn.Linear(128, 64)
        self.combined_layer2 = nn.Linear(64, 1)
        
        # Orthogonal initialization for better gradient flow
        nn.init.orthogonal_(self.state_layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.state_layer2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.action_layer.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.combined_layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.combined_layer2.weight, gain=1.0)
        
        # Initialize final layer with small weights
        nn.init.constant_(self.combined_layer2.bias, 0.0)
    
    def forward(self, state, action):
        # Process state separately
        s = F.leaky_relu(self.state_layer1(state))
        s = F.leaky_relu(self.state_layer2(s))
        
        # Process action separately
        a = F.leaky_relu(self.action_layer(action))
        
        # Combine state and action processing
        x = torch.cat([s, a], dim=1)
        x = F.leaky_relu(self.combined_layer1(x))
        return self.combined_layer2(x)

def train_policy(datasets, contacts_frame, agent, robot, save_policy=True):
    episode_rewards = []
    converged = False
    best_reward = float('-inf')
    reward_history = []
    convergence_threshold = 0.1  # How close to best reward we need to be (as a fraction)
    window_size = 10  # Window size for convergence check

    # Learning rate schedulers for adaptive learning
    actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        agent.actor_optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5
    )
    critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        agent.critic_optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5
    )

    # Initialize noise decay
    initial_noise = agent.actor.log_noise_sigma
    min_noise = agent.actor.min_log_noise_sigma
    noise_decay = 0.98  # Slower decay

    # Save best model callback
    def save_best_model(episode_reward):
        nonlocal best_reward
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            
            if save_policy:
                os.makedirs('policy/ddpg', exist_ok=True)
                torch.save({
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                }, f'policy/ddpg/trained_policy_{robot}.pth')
                print(f"Saved better policy with reward {episode_reward:.4f}")
                
                # Export to ONNX
                export_models_to_onnx(agent, robot, params)
            return True
        return False

    # Training statistics tracking
    stats = {
        'critic_losses': [],
        'actor_losses': [],
        'rewards': [],
        'episode_lengths': []
    }
    
    # Training loop with evaluation phases
    for i, dataset in enumerate(datasets):
        print(f"Training on dataset {i+1}/{len(datasets)}")
        
        # Get dataset measurements
        imu_measurements = dataset['imu']
        joint_measurements = dataset['joints']
        force_torque_measurements = dataset['ft']
        base_pose_ground_truth = dataset['base_pose_ground_truth']

        # Set proper limits on number of episodes
        max_episodes = 100  # Reduced episode count with better early stopping
        max_steps = min(len(imu_measurements), 10000)  # Cap steps per episode
        
        for episode in range(max_episodes):
            # Initialize framework for this episode
            serow_framework = serow.Serow()
            serow_framework.initialize(f"{robot}_rl.json")
            state = serow_framework.get_state(allow_invalid=True)

            # Episode tracking variables
            episode_reward = 0.0
            collected_steps = 0
            train_freq = 50  # Train after every N steps
            episode_critic_losses = []
            episode_actor_losses = []
            update_step_counter = 0
            
            # Run episode
            for step, (imu, joints, ft, gt) in enumerate(zip(
                imu_measurements[:max_steps], 
                joint_measurements[:max_steps], 
                force_torque_measurements[:max_steps], 
                base_pose_ground_truth[:max_steps]
            )):
                # Run step with current policy
                _, state, rewards, done = run_step(imu, joints, ft, gt, serow_framework, state, agent)

                # Accumulate rewards
                step_reward = 0
                for cf, reward in rewards.items():
                    if reward is not None:
                        step_reward += reward
                        collected_steps += 1
                
                episode_reward += step_reward
                
                # Check for early termination due to filter divergence
                should_terminate = False
                for cf in contacts_frame:
                    if done[cf] > 0.5:
                        should_terminate = True
                        break
                
                if should_terminate:
                    print(f"Episode {episode} terminated early at step {step}/{max_steps} due to filter divergence")
                    break
                
                # Train policy periodically, not every step
                if collected_steps >= train_freq:
                    critic_loss, actor_loss = agent.train()
                    if critic_loss is not None and actor_loss is not None:
                        episode_critic_losses.append(critic_loss)
                        episode_actor_losses.append(actor_loss)
                    collected_steps = 0
                    update_step_counter += 1
                
                # Progress logging
                if step % 500 == 0 or step == max_steps - 1:
                    print(f"Episode {episode}/{max_episodes}, Step {step}/{max_steps}, " 
                          f"Reward: {episode_reward:.2f}, Best: {best_reward:.2f}, "
                          f"Noise: {agent.actor.log_noise_sigma:.4f}")
            
            # End of episode processing
            episode_rewards.append(episode_reward)
            reward_history.append(episode_reward)
            if len(reward_history) > window_size:
                reward_history.pop(0)
            
            # Update learning rates based on episode performance
            actor_scheduler.step(episode_reward)
            critic_scheduler.step(episode_reward)
            
            # Save best model if improved
            save_best_model(episode_reward)
            
            # Check convergence by comparing recent rewards to best reward
            if len(reward_history) >= window_size:
                recent_rewards = np.array(reward_history)
                # Calculate how close recent rewards are to best reward
                reward_ratios = recent_rewards / (abs(best_reward) + 1e-6)  # Avoid division by zero
                # Check if all recent rewards are within threshold of best reward
                if np.all(reward_ratios >= (1.0 - convergence_threshold)):
                    print(f"Training converged! Recent rewards are within {convergence_threshold*100}% of best reward {best_reward:.2f}")
                    converged = True
                    break  # Break out of episode loop
            
            # Decay noise
            agent.actor.log_noise_sigma = max(
                min_noise,
                agent.actor.log_noise_sigma * noise_decay
            )
            
            # Store episode statistics
            if episode_critic_losses:
                stats['critic_losses'].append(np.mean(episode_critic_losses))
            if episode_actor_losses:
                stats['actor_losses'].append(np.mean(episode_actor_losses))
            stats['rewards'].append(episode_reward)
            stats['episode_lengths'].append(step)
            
            # Print episode summary
            print(f"Episode {episode} completed with reward {episode_reward:.2f}")
            if episode_critic_losses:
                print(f"Average critic loss: {np.mean(episode_critic_losses):.4f}")
            if episode_actor_losses:
                print(f"Average actor loss: {np.mean(episode_actor_losses):.4f}")
        
        if converged:
            break  # Break out of dataset loop
    
    # Plot training curves
    plot_training_curves(stats, episode_rewards)
    
    return agent, stats

def export_models_to_onnx(agent, robot, params):
    """Export the trained models to ONNX format"""
    os.makedirs('policy/ddpg', exist_ok=True)
    
    # Export actor model
    dummy_input = torch.randn(1, params['state_dim']).to(agent.device)
    torch.onnx.export(
        agent.actor,
        dummy_input,
        f'policy/ddpg/trained_policy_{robot}_actor.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}}
    )
    
    # Export critic model
    dummy_state = torch.randn(1, params['state_dim']).to(agent.device)
    dummy_action = torch.randn(1, params['action_dim']).to(agent.device)
    torch.onnx.export(
        agent.critic,
        (dummy_state, dummy_action),
        f'policy/ddpg/trained_policy_{robot}_critic.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['state', 'action'],
        output_names=['output'],
        dynamic_axes={'state': {0: 'batch_size'},
                    'action': {0: 'batch_size'},
                    'output': {0: 'batch_size'}}
    )

def evaluate_policy_brief(dataset, contacts_frame, agent, robot, num_steps=1000):
    """Perform a quick evaluation of the policy"""
    # Get the measurements and ground truth data
    imu_measurements = dataset['imu'][:num_steps]
    joint_measurements = dataset['joints'][:num_steps]
    force_torque_measurements = dataset['ft'][:num_steps]
    base_pose_ground_truth = dataset['base_pose_ground_truth'][:num_steps]

    # Initialize SEROW
    serow_framework = serow.Serow()
    serow_framework.initialize(f"{robot}_rl.json")
    state = serow_framework.get_state(allow_invalid=True)

    # Run evaluation
    rewards = []
    diverged = False
    
    for step, (imu, joints, ft, gt) in enumerate(zip(
        imu_measurements, joint_measurements, force_torque_measurements, base_pose_ground_truth
    )):
        _, state, step_rewards, done = run_step(imu, joints, ft, gt, serow_framework, state, agent, deterministic=True)
        
        # Check for divergence
        for cf in contacts_frame:
            if done[cf] > 0.5:
                diverged = True
                break
        
        if diverged:
            break
            
        # Collect rewards
        step_reward = 0
        for reward in step_rewards.values():
            if reward is not None:
                step_reward += reward
        rewards.append(step_reward)
    
    # Print brief evaluation results
    avg_reward = np.mean(rewards) if rewards else float('-inf')
    print(f"Evaluation: Steps completed: {len(rewards)}/{num_steps}, " 
          f"Avg reward: {avg_reward:.4f}, Diverged: {diverged}")
    return avg_reward, diverged

def plot_training_curves(stats, episode_rewards):
    """Plot training curves to visualize progress"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label='Episode Rewards', alpha=0.7)
    
    # Apply smoothing
    window_size = min(len(episode_rewards) // 5, 10)
    if window_size > 1:
        smoothed = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(episode_rewards)), smoothed, 'r-', linewidth=2, label='Smoothed')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.legend()
    
    # Plot losses if available
    if stats['critic_losses']:
        plt.subplot(2, 2, 2)
        plt.plot(stats['critic_losses'], label='Critic Loss')
        plt.xlabel('Training Updates')
        plt.ylabel('Loss')
        plt.title('Critic Loss')
        plt.grid(True)
        plt.legend()
    
    if stats['actor_losses']:
        plt.subplot(2, 2, 3)
        plt.plot(stats['actor_losses'], label='Actor Loss')
        plt.xlabel('Training Updates')
        plt.ylabel('Loss')
        plt.title('Actor Loss')
        plt.grid(True)
        plt.legend()
    
    if stats['episode_lengths']:
        plt.subplot(2, 2, 4)
        plt.plot(stats['episode_lengths'], label='Episode Length')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Length')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'policy/ddpg/training_curves.png')
    plt.show()

def evaluate_policy(dataset, contacts_frame, agent, robot):
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
            print(f"Evaluating DDPG policy for {robot} at step {step}")
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

    # Get the contacts frame
    contacts_frame = set(contact_states[0].contacts_status.keys())
    print(f"Contacts frame: {contacts_frame}")

    # Compute max and min state values
    feet_positions = []
    for base_state in base_states:
        for cf in contacts_frame:
            if base_state.contacts_position[cf] is not None:
                R_base = quaternion_to_rotation_matrix(base_state.base_orientation).transpose()
                local_pos = R_base @ (base_state.base_position - base_state.contacts_position[cf])
                feet_positions.append(local_pos)
    
    # Convert feet_positions to numpy array for easier manipulation
    feet_positions = np.array(feet_positions)
    # Create max and min state values with correct dimensions
    # First 3 dimensions are for position, last dimension is for contact probability
    max_state_value = np.concatenate([np.max(feet_positions, axis=0), [1.0]])
    min_state_value = np.concatenate([np.min(feet_positions, axis=0), [0.0]])

    print(f"Max state value: {max_state_value}")
    print(f"Min state value: {min_state_value}")

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

    # Define the dimensions of your state and action spaces
    state_dim = 4  
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    min_action = 1e-10

    params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': None,
        'min_action': min_action,
        'gamma': 0.99,
        'tau': 0.005,
        'batch_size': 512,  # Increased for more stable gradients
        'actor_lr': 1e-4, 
        'critic_lr': 2e-4,  # Reduced to be closer to actor learning rate
        'log_noise_sigma': 0.5,  # Reduced initial exploration noise
        'min_log_noise_sigma': 0.01,  # Reduced minimum noise
        'buffer_size': 200000,  # Reduced buffer size
        'max_state_value': max_state_value,
        'min_state_value': min_state_value
    }

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    loaded = False
    robot = "go2"
    print(f"Initializing agent for {robot}")
    actor = Actor(params).to(device)
    critic = Critic(params).to(device)
    agent = DDPG(actor, critic, params, device=device, normalize_state=True)

    # Try to load a trained policy for this robot if it exists
    try:
        checkpoint = torch.load(f'policy/ddpg/trained_policy_{robot}.pth', weights_only=True)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded trained policy for {robot} from 'policy/ddpg/trained_policy_{robot}.pth'")
        loaded = True
    except FileNotFoundError:
        print(f"No trained policy found for {robot}. Training new policy...")

    if not loaded:
        # Train the policy
        train_policy(train_datasets, contacts_frame, agent, robot, save_policy=True)
        # Load the best policy
        checkpoint = torch.load(f'policy/ddpg/trained_policy_{robot}.pth', weights_only=True)
        actor = Actor(params).to(device)
        critic = Critic(params).to(device)
        best_policy = DDPG(actor, critic, params, device=device)
        best_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        best_policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded optimal trained policy for {robot} from 'policy/ddpg/trained_policy_{robot}.pth'")
        evaluate_policy(test_dataset, contacts_frame, best_policy, robot)
    else:
        # Just evaluate the loaded policy
        evaluate_policy(test_dataset, contacts_frame, agent, robot)
