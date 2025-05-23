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
    read_imu_measurements, 
    read_base_pose_ground_truth,
    read_joint_states
)
from utils import(
    run_step,
    plot_trajectories,
    sync_and_align_data,
    filter,
    quaternion_to_rotation_matrix,
)

class Actor(nn.Module):
    class OUNoise:
        """
        Ornstein-Uhlenbeck Process noise generator for exploration in continuous action spaces.
        This noise process generates temporally correlated noise that helps with exploration
        in continuous action spaces.
        """
        def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, decay=0.995, min_sigma=0.01):
            """
            Initialize OU Noise generator.
            
            Args:
                action_dim (int): Dimension of the action space
                mu (float): Mean of the noise
                theta (float): Rate of mean reversion
                sigma (float): Initial scale of the noise
                dt (float): Time step size (must be positive)
                decay (float): Decay rate for noise (between 0 and 1)
                min_sigma (float): Minimum noise scale
            """
            self.action_dim = action_dim
            self.mu = mu * np.ones(action_dim)
            self.theta = theta
            self.initial_sigma = sigma
            self.sigma = sigma
            self.dt = dt
            self.decay = decay
            self.min_sigma = min_sigma
            self.reset()
        
        def reset(self):
            """Reset the noise process to its initial state."""
            self.state = np.copy(self.mu)
        
        def update_noise(self, episode):
            """
            Update the noise scale based on the episode number.
            
            Args:
                episode (int): Current episode number
            """
            self.sigma = max(self.initial_sigma * (self.decay ** episode), self.min_sigma)
        
        def sample(self):
            """
            Generate a noise sample using the Ornstein-Uhlenbeck process.
            
            Returns:
                numpy.ndarray: Noise sample of shape (action_dim,)
            """
            dx = self.theta * (self.mu - self.state) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
            self.state += dx
            return self.state

    def __init__(self, params):
        super(Actor, self).__init__()
        self.noise = self.OUNoise(
            params['action_dim'], 
            sigma=params['noise_sigma'], 
            theta=params['theta'], 
            dt=params['dt'],
            decay=params['noise_decay'],
            min_sigma=params['min_noise_sigma']
        )
        self.layer1 = nn.Linear(params['state_dim'], 128)
        self.dropout1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.layer3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.1)
        self.mean_layer = nn.Linear(64, params['action_dim'])
        
        # Proper initialization for approximating identity function initially
        nn.init.orthogonal_(self.layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer3.weight, gain=np.sqrt(2))

        # Initialize bias to produce an output close to 1.0
        nn.init.constant_(self.mean_layer.bias, 1.0) 
        
        self.min_action = params['min_action']
        self.action_dim = params['action_dim']

    def forward(self, state):
        x = F.leaky_relu(self.layer1(state))
        x = self.dropout1(x)
        x = F.leaky_relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.layer3(x))
        x = self.dropout3(x)
        return F.softplus(self.mean_layer(x)) 

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(next(self.parameters()).device)
            action = self.forward(state).cpu().numpy()[0]
            if not deterministic:
                action = action + np.abs(self.noise.sample())

            # Ensure minimum action value and positivity
            action = np.maximum(action, self.min_action)
            return action

class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        # Process state first through separate network
        self.state_layer1 = nn.Linear(params['state_dim'], 64)
        self.dropout1 = nn.Dropout(0.1)
        self.state_layer2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.1)
        
        # Process action separately
        self.action_layer = nn.Linear(params['action_dim'], 64)
        self.dropout3 = nn.Dropout(0.1)
        
        # Combine state and action processing
        self.combined_layer1 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.1)
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
        s = self.dropout1(s)
        s = F.leaky_relu(self.state_layer2(s))
        s = self.dropout2(s)
        
        # Process action separately
        a = F.leaky_relu(self.action_layer(action))
        a = self.dropout3(a)
        
        # Combine state and action processing
        x = torch.cat([s, a], dim=1)
        x = F.leaky_relu(self.combined_layer1(x))
        x = self.dropout4(x)
        return self.combined_layer2(x)

def train_policy(datasets, contacts_frame, agent, robot, save_policy=True):
    episode_rewards = []
    converged = False
    best_reward = float('-inf')
    reward_history = []
    convergence_threshold = 0.1  # How close to best reward we need to be (as a fraction) to mark convergence
    window_size = 10  # Window size for convergence check
    train_freq = 100  # Call train() every train_freq steps

    # Learning rate schedulers
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
        'episode_lengths': [],
        'noise_scales': [] 
    }
    
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

        # Set proper limits on number of episodes
        max_episodes = 700 
        max_steps = len(imu_measurements) - 1 
        
        for episode in range(max_episodes):
            # Update noise scale for this episode
            agent.actor.noise.update_noise(episode)
            stats['noise_scales'].append(agent.actor.noise.sigma)
            
            # Initialize SEROW
            serow_framework = serow.Serow()
            serow_framework.initialize(f"{robot}_rl.json")
            state = serow_framework.get_state(allow_invalid=True)
            state.set_joint_state(joint_states[0])
            state.set_base_state(base_states[0])  
            state.set_contact_state(contact_states[0])
            serow_framework.set_state(state)

            # Episode tracking variables
            episode_reward = 0.0
            collected_steps = 0
            episode_critic_losses = []
            episode_actor_losses = []
            baseline = True if episode == 0 or episode % 10 == 0 else False
            if baseline:
                print(f"Episode {episode}, Evaluating baseline policy")
            
            # Run episode
            for step, (imu, joints, ft, gt, cs, next_cs) in enumerate(zip(
                imu_measurements[:max_steps], 
                joint_measurements[:max_steps], 
                force_torque_measurements[:max_steps], 
                base_pose_ground_truth[:max_steps],
                contact_states[:max_steps],
                contact_states[1:max_steps + 1]
            )):
                # Run step with current policy
                _, state, rewards, done = run_step(imu, joints, ft, gt, serow_framework, state, 
                                                   agent, contact_state=cs, 
                                                   next_contact_state=next_cs, deterministic=False, 
                                                   baseline=baseline)

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
                
                # Train policy periodically
                if collected_steps >= train_freq:
                    critic_loss, actor_loss = agent.train()
                    if critic_loss is not None and actor_loss is not None:
                        episode_critic_losses.append(critic_loss)
                        episode_actor_losses.append(actor_loss)
                    collected_steps = 0
                
                # Progress logging
                if step % 500 == 0 or step == max_steps - 1:
                    print(f"Episode {episode}/{max_episodes}, Step {step}/{max_steps}, " 
                          f"Reward: {episode_reward:.2f}, Best: {best_reward:.2f}, "
                          f"Noise: {agent.actor.noise.sigma:.4f}")
            
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
    
    if stats['noise_scales']:
        plt.subplot(2, 2, 5)
        plt.plot(stats['noise_scales'], label='Noise Scale')
        plt.xlabel('Episode')
        plt.ylabel('Noise Scale')
        plt.title('Noise Scale')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'policy/ddpg/training_curves.png')
    plt.show()

def evaluate_policy(dataset, contacts_frame, agent, robot):
        # After training, evaluate the policy
        print(f"\nEvaluating trained DDPG policy for {robot}...")
        max_steps = len(dataset['imu']) - 1
        
        # Get the measurements and the ground truth
        imu_measurements = dataset['imu'][:max_steps]
        joint_measurements = dataset['joints'][:max_steps]
        force_torque_measurements = dataset['ft'][:max_steps]
        base_pose_ground_truth = dataset['base_pose_ground_truth'][:max_steps]
        contact_states = dataset['contact_states'][:max_steps]
        joint_states = dataset['joint_states'][:max_steps]
        base_states = dataset['base_states'][:max_steps]
        next_contact_states = dataset['contact_states'][1:max_steps + 1]

        # Initialize SEROW
        serow_framework = serow.Serow()
        serow_framework.initialize(f"{robot}_rl.json")
        state = serow_framework.get_state(allow_invalid=True)
        state.set_joint_state(joint_states[0])
        state.set_base_state(base_states[0])  
        state.set_contact_state(contact_states[0])
        serow_framework.set_state(state)

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

        for step, (imu, joints, ft, gt, cs, next_cs) in enumerate(zip(imu_measurements, 
                                                                      joint_measurements, 
                                                                      force_torque_measurements, 
                                                                      base_pose_ground_truth,
                                                                      contact_states,
                                                                      next_contact_states)):
            print("-------------------------------------------------")
            print(f"Evaluating DDPG policy for {robot} at step {step}")
            timestamp, state, rewards, _ = run_step(imu, joints, ft, gt, serow_framework, state, 
                                                    agent, contact_state=cs, 
                                                    next_contact_state=next_cs, deterministic=True)
            
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
        timestamps, base_positions, base_orientations, gt_positions, gt_orientations = \
            sync_and_align_data(timestamps, base_positions, base_orientations, gt_timestamps, 
                                gt_positions, gt_orientations, align = True)

        # Plot the trajectories
        plot_trajectories(timestamps, base_positions, base_orientations, gt_positions, 
                          gt_orientations, cumulative_rewards)

        # Print evaluation metrics
        print("\n DDPG Policy Evaluation Metrics:")
        for cf in contacts_frame:
            print(f"Average Cumulative Reward for {cf}: {np.mean(cumulative_rewards[cf]):.4f}")
            print(f"Max Cumulative Reward for {cf}: {np.max(cumulative_rewards[cf]):.4f}")
            print(f"Min Cumulative Reward for {cf}: {np.min(cumulative_rewards[cf]):.4f}")
            print("-------------------------------------------------")

if __name__ == "__main__":
    # Load and preprocess the data
    imu_measurements  = read_imu_measurements("/tmp/serow_measurements.mcap")
    joint_measurements = read_joint_measurements("/tmp/serow_measurements.mcap")
    force_torque_measurements = read_force_torque_measurements("/tmp/serow_measurements.mcap")
    base_pose_ground_truth = read_base_pose_ground_truth("/tmp/serow_measurements.mcap")
    base_states = read_base_states("/tmp/serow_proprioception.mcap")
    contact_states = read_contact_states("/tmp/serow_proprioception.mcap")
    joint_states = read_joint_states("/tmp/serow_proprioception.mcap")

    SYNC_AND_ALIGN = True
    if (SYNC_AND_ALIGN):
        # Initialize SEROW
        serow_framework = serow.Serow()
        serow_framework.initialize("go2_rl.json")
        state = serow_framework.get_state(allow_invalid=True)
        state.set_joint_state(joint_states[0])
        state.set_base_state(base_states[0])  
        state.set_contact_state(contact_states[0])
        serow_framework.set_state(state)
        
        contact_states = []
        timestamps, base_position_aligned, base_orientation_aligned, \
            gt_position_aligned, gt_orientation_aligned, cumulative_rewards, \
            contact_states = filter( imu_measurements, joint_measurements, force_torque_measurements,
                                    base_pose_ground_truth, serow_framework, state, align=True)

        # Reform the ground truth data
        base_pose_ground_truth = []
        for i in range(len(timestamps)):
            gt = serow.BasePoseGroundTruth()
            gt.timestamp = timestamps[i]
            gt.position = gt_position_aligned[i]
            gt.orientation = gt_orientation_aligned[i]
            base_pose_ground_truth.append(gt)

    # Plot the baseline trajectory vs the ground truth
    plot_trajectories(timestamps, base_position_aligned, base_orientation_aligned, 
                      gt_position_aligned, gt_orientation_aligned, None)

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
    contact_probabilities = []
    for contact_state in contact_states:
        for cf in contacts_frame:
            contact_probabilities.append(contact_state.contacts_probability[cf])
    contact_probabilities = np.array(contact_probabilities)

    # Compute the dt
    dt = []
    for i in range(len(imu_measurements) - 1):
        dt.append(imu_measurements[i+1].timestamp - imu_measurements[i].timestamp)
    dt = np.median(np.array(dt))
    print(f"dt: {dt}")

    # Create max and min state values with correct dimensions
    # First 3 dimensions are for position, last dimension is for contact probability
    max_state_value = np.concatenate([np.max(feet_positions, axis=0), 
                                      [np.max(contact_probabilities)]])
    min_state_value = np.concatenate([np.min(feet_positions, axis=0), 
                                      [np.min(contact_probabilities)]])

    print(f"RL state max values: {max_state_value}")
    print(f"RL state min values: {min_state_value}")

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
        'tau': 0.001,
        'batch_size': 512,  
        'actor_lr': 5e-4, 
        'critic_lr': 1e-4,  
        'buffer_size': 5000000,  
        'max_state_value': max_state_value,
        'min_state_value': min_state_value,
        'train_for_batches': 5,
        'noise_sigma': 1.5,
        'theta': 0.15,
        'dt': dt,
        'noise_decay': 0.99, 
        'min_noise_sigma': 0.01  
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
