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

# Actor network per leg end-effector
class Actor(nn.Module):
    def __init__(self, params):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(params['state_dim'], 128)
        self.dropout1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.layer3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.1)
        self.mean_layer = nn.Linear(64, params['action_dim'])
        self.log_std_layer = nn.Linear(64, params['action_dim'])
        self.min_action = params['min_action']
        self.action_dim = params['action_dim']

        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.layer1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.layer2.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.layer3.weight, gain=np.sqrt(2))
        
        # Initialize bias to produce an output close to 1.0
        nn.init.constant_(self.mean_layer.bias, 1.0) 
        
        # Initialize log_std layer with small negative values
        nn.init.zeros_(self.log_std_layer.weight)
        nn.init.constant_(self.log_std_layer.bias, -1.0)  # Start with small std

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        x = self.dropout3(x)

        mean = F.softplus(self.mean_layer(x)) 
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=0.5)  # Prevent extreme values
        return mean, log_std

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(next(self.parameters()).device)
            mean, log_std = self.forward(state)
            # Convert to numpy arrays
            mean = mean.detach().cpu().numpy().reshape(-1, 1)
            log_std = log_std.detach().cpu().item()
            
            if deterministic:
                action = mean
                log_prob = 0.0
            else:
                std = np.exp(log_std)
                std = np.clip(std, a_min=1e-6, a_max=1.0)  # Prevent too small or large std
                z = np.random.normal(mean, std)
                action = z
                log_prob = -0.5 * ((z - mean) / std)**2 - np.log(std) - 0.5 * np.log(2 * np.pi)
            action = np.maximum(action, self.min_action)
            return action, log_prob

class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(params['state_dim'], 128)
        self.dropout1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.1)
        self.layer3 = nn.Linear(64, 1)
        
        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.layer1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.layer2.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.layer3.weight, gain=np.sqrt(2))
        
        # Initialize final layer with small weights
        nn.init.constant_(self.layer3.bias, 0.0)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        return self.layer3(x)

def train_policy(datasets, contacts_frame, agent, robot, save_policy=True):
    episode_rewards = []
    converged = False
    best_reward = float('-inf')
    best_reward_episode = 0
    reward_history = []
    convergence_threshold = 0.05  # How close to best reward we need to be (as a fraction) to mark convergence
    critic_convergence_threshold = 0.05  # How much the critic loss can vary before considering it converged
    window_size = 10  # Window size for convergence check
    train_freq = 100  # Call train() every train_freq steps

    # Save best model callback
    def save_best_model(episode, episode_reward):
        nonlocal best_reward
        nonlocal best_reward_episode

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_reward_episode = episode

            if save_policy:
                path = 'policy/ppo/best'
                os.makedirs(path, exist_ok=True)
                torch.save({
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                }, f'{path}/trained_policy_{robot}.pth')
                print(f"Saved better policy with reward {episode_reward:.4f}")
                
                # Export to ONNX
                export_models_to_onnx(agent, robot, params, path)
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
        max_episodes = 1000
        max_steps = len(imu_measurements) - 1 

        for episode in range(max_episodes):
            # Update noise scale for this episode
            # Placeholder for noise scale update

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

            # Run episode
            for step, (imu, joints, ft, gt, cs, next_cs) in enumerate(zip(
                imu_measurements[:max_steps], 
                joint_measurements[:max_steps], 
                force_torque_measurements[:max_steps], 
                base_pose_ground_truth[:max_steps],
                contact_states[:max_steps],
                contact_states[1:max_steps+1]
            )):
                # Run step with current policy
                _, state, rewards, done = run_step(imu, joints, ft, gt, serow_framework, state, step,
                                                   agent, contact_state=cs, 
                                                   next_contact_state=next_cs, deterministic=False, 
                                                   baseline=False)
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
                    if done[cf] is not None and done[cf] == 1.0:
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
                          f"Episode reward: {episode_reward:.2f}, Best: {best_reward:.2f}, " 
                          f"in episode {best_reward_episode}")

            # End of episode processing
            episode_rewards.append(episode_reward)
            reward_history.append(episode_reward)
            if len(reward_history) > window_size:
                reward_history.pop(0)
            
            # Save best model if improved
            save_best_model(episode, episode_reward)

            # Check convergence by comparing recent rewards to best reward and critic loss
            if len(reward_history) >= window_size:
                recent_rewards = np.array(reward_history)
                # Calculate how close recent rewards are to best reward
                reward_ratios = recent_rewards / (abs(best_reward) + 1e-6)  # Avoid division by zero
                
                # Check if rewards have converged
                rewards_converged = np.all(reward_ratios >= (1.0 - convergence_threshold))
                
                # Check if critic loss has converged
                critic_loss_converged = False
                if len(stats['critic_losses']) >= window_size:
                    recent_critic_losses = np.array(stats['critic_losses'][-window_size:])
                    critic_loss_std = np.std(recent_critic_losses)
                    critic_loss_mean = np.mean(recent_critic_losses)
                    critic_loss_converged = critic_loss_std <= (critic_loss_mean * critic_convergence_threshold)
                
                # Both rewards and critic loss must be converged
                if rewards_converged and critic_loss_converged:
                    print(f"Training converged!")
                    print(f"Recent rewards are within {convergence_threshold*100}% of best episode reward {best_reward:.2f} in episode {best_reward_episode}")
                    print(f"Critic loss has stabilized with std/mean ratio: {critic_loss_std/critic_loss_mean:.4f}")
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
            print(f"Episode {episode} completed with episode reward {episode_reward:.2f}")
            if episode_critic_losses:
                print(f"Average critic loss: {np.mean(episode_critic_losses):.4f}")
            if episode_actor_losses:
                print(f"Average actor loss: {np.mean(episode_actor_losses):.4f}")
        
        if converged or episode == max_episodes - 1:
            if save_policy:
                # Save the final policy
                path = 'policy/ppo/final'
                os.makedirs(path, exist_ok=True)
                torch.save({
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                }, f'{path}/trained_policy_{robot}.pth')
            print(f"Saved final policy with reward {episode_reward:.4f}")
            # Export to ONNX
            export_models_to_onnx(agent, robot, params, path)
            break  # Break out of dataset loop
    
    # Plot training curves
    plot_training_curves(stats, episode_rewards)
    
    return agent, stats
   
def plot_training_curves(stats, episode_rewards):
    """Plot training curves to visualize progress"""
    plt.figure(figsize=(15, 15))
    
    # Plot rewards
    plt.subplot(3, 2, 1)
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
        plt.subplot(3, 2, 2)
        plt.plot(stats['critic_losses'], label='Critic Loss')
        plt.xlabel('Training Updates')
        plt.ylabel('Loss')
        plt.title('Critic Loss')
        plt.grid(True)
        plt.legend()
    
    if stats['actor_losses']:
        plt.subplot(3, 2, 3)
        plt.plot(stats['actor_losses'], label='Actor Loss')
        plt.xlabel('Training Updates')
        plt.ylabel('Loss')
        plt.title('Actor Loss')
        plt.grid(True)
        plt.legend()
    
    if stats['episode_lengths']:
        plt.subplot(3, 2, 4)
        plt.plot(stats['episode_lengths'], label='Episode Length')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Length')
        plt.grid(True)
        plt.legend()
    
    if stats['noise_scales']:
        plt.subplot(3, 2, 5)
        plt.plot(stats['noise_scales'], label='Noise Scale')
        plt.xlabel('Episode')
        plt.ylabel('Noise Scale')
        plt.title('Noise Scale')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'policy/ppo/training_curves.png')
    plt.show()


def evaluate_policy(dataset, contacts_frame, agent, robot):
        # After training, evaluate the policy
        print(f"\nEvaluating trained PPO policy for {robot}...")
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
        instantaneous_rewards = {}
        for cf in contacts_frame:
            cumulative_rewards[cf] = []
            instantaneous_rewards[cf] = []

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
            print(f"Evaluating PPO policy for {robot} at step {step}")
            timestamp, state, rewards, _ = run_step(imu, joints, ft, gt, serow_framework, state, step,
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
                    instantaneous_rewards[cf].append(rewards[cf])
                    if len(cumulative_rewards[cf]) == 0:
                        cumulative_rewards[cf].append(rewards[cf])
                    else:
                        cumulative_rewards[cf].append(cumulative_rewards[cf][-1] + rewards[cf])

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
        print("\n PPO Policy Evaluation Metrics:")
        for cf in contacts_frame:
            print(f"Average Reward for {cf}: {np.mean(instantaneous_rewards[cf]):.4f}")
            print(f"Max Reward for {cf}: {np.max(instantaneous_rewards[cf]):.4f} at step {np.argmax(instantaneous_rewards[cf])}")
            print(f"Min Reward for {cf}: {np.min(instantaneous_rewards[cf]):.4f} at step {np.argmin(instantaneous_rewards[cf])}")
            print("-------------------------------------------------")


def export_models_to_onnx(agent, robot, params, path):
    """Export the trained models to ONNX format"""
    os.makedirs(path, exist_ok=True)
    
    # Export actor model
    dummy_input = torch.randn(1, params['state_dim']).to(agent.device)
    torch.onnx.export(
        agent.actor,
        dummy_input,
        f'{path}/trained_policy_{robot}_actor.onnx',
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
    torch.onnx.export(
        agent.critic,
        dummy_state,
        f'{path}/trained_policy_{robot}_critic.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['output'],
        dynamic_axes={'state': {0: 'batch_size'},
                    'output': {0: 'batch_size'}}
    )

if __name__ == "__main__":
    # Load and preprocess the data
    imu_measurements  = read_imu_measurements("/tmp/serow_measurements.mcap")
    joint_measurements = read_joint_measurements("/tmp/serow_measurements.mcap")
    force_torque_measurements = read_force_torque_measurements("/tmp/serow_measurements.mcap")
    base_pose_ground_truth = read_base_pose_ground_truth("/tmp/serow_measurements.mcap")
    base_states = read_base_states("/tmp/serow_proprioception.mcap")
    contact_states = read_contact_states("/tmp/serow_proprioception.mcap")
    joint_states = read_joint_states("/tmp/serow_proprioception.mcap")

    # Define the dimensions of your state and action spaces
    state_dim = 7  
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    min_action = 1e-10
    robot = "go2"

    SYNC_AND_ALIGN = True
    if (SYNC_AND_ALIGN):
        # Initialize SEROW
        serow_framework = serow.Serow()
        serow_framework.initialize(f"{robot}_rl.json")
        state = serow_framework.get_state(allow_invalid=True)
        state.set_joint_state(joint_states[0])
        state.set_base_state(base_states[0])  
        state.set_contact_state(contact_states[0])
        serow_framework.set_state(state)
        
        contact_states = []
        timestamps, base_position_aligned, base_orientation_aligned, \
            gt_position_aligned, gt_orientation_aligned, cumulative_rewards, \
            contact_states = filter(imu_measurements, joint_measurements, force_torque_measurements,
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
    # plot_trajectories(timestamps, base_position_aligned, base_orientation_aligned, 
    #                   gt_position_aligned, gt_orientation_aligned, cumulative_rewards)

    # Get the contacts frame
    contacts_frame = set(contact_states[0].contacts_status.keys())
    print(f"Contacts frame: {contacts_frame}")

    # Compute max and min state values
    feet_positions = []
    base_linear_velocities = []
    for base_state in base_states:
        R_base = quaternion_to_rotation_matrix(base_state.base_orientation).transpose()
        base_linear_velocities.append(R_base @ base_state.base_linear_velocity)
        for cf in contacts_frame:
            if base_state.contacts_position[cf] is not None:
                local_pos = R_base @ (base_state.base_position - base_state.contacts_position[cf])
                feet_positions.append(np.array([abs(local_pos[0]), abs(local_pos[1]), local_pos[2]]))
    
    # Convert base_linear_velocities and feet_positions to numpy array for easier manipulation
    base_linear_velocities = np.array(base_linear_velocities)
    feet_positions = np.array(feet_positions)
    contact_probabilities = []
    for contact_state in contact_states:
        for cf in contacts_frame:
            if contact_state.contacts_status[cf]:
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
                                      np.max(base_linear_velocities, axis=0),
                                      [np.max(contact_probabilities)]])
    # max_state_value = np.outer(max_state_value, max_state_value).reshape(state_dim, 1)
    min_state_value = np.concatenate([np.min(feet_positions, axis=0), 
                                      np.min(base_linear_velocities, axis=0),
                                      [np.min(contact_probabilities)]])
    # min_state_value = np.outer(min_state_value, min_state_value).reshape(state_dim, 1)

    print(f"RL state max values: {max_state_value}")
    print(f"RL state min values: {min_state_value}")
    
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

    params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': None,
        'min_action': min_action,
        'clip_param': 0.2,  
        'value_loss_coef': 1.0,  
        'entropy_coef': 0.01,  
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_epochs': 10,  
        'batch_size': 64,  
        'max_grad_norm': 10.0,  
        'buffer_size': 100000,  
        'actor_lr': 1e-5, 
        'critic_lr': 1e-5,  
        'max_state_value': max_state_value,
        'min_state_value': min_state_value,
        'noise_sigma': 1.0,
        'dt': dt,
    }

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    loaded = False
    print(f"Initializing agent for {robot}")
    actor = Actor(params).to(device)
    critic = Critic(params).to(device)
    agent = PPO(actor, critic, params, device=device, normalize_state=True)

    policy_path = 'policy/ppo/final'
    # Try to load a trained policy for this robot if it exists
    try:
        checkpoint = torch.load(f'{policy_path}/trained_policy_{robot}.pth', weights_only=True)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded trained policy for {robot} from '{policy_path}/trained_policy_{robot}.pth'")
        loaded = True
    except FileNotFoundError:
        print(f"No trained policy found for {robot}. Training new policy...")

    if not loaded:
        # Train the policy
        train_policy(train_datasets, contacts_frame, agent, robot, save_policy=True)
        # Load the best policy
        checkpoint = torch.load(f'{policy_path}/trained_policy_{robot}.pth', weights_only=True)
        actor = Actor(params).to(device)
        critic = Critic(params).to(device)
        best_policy = PPO(actor, critic, params, device=device)
        best_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        best_policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded optimal trained policy for {robot} from '{policy_path}/trained_policy_{robot}.pth'")
        evaluate_policy(test_dataset, contacts_frame, best_policy, robot)
    else:
        # Just evaluate the loaded policy
        evaluate_policy(test_dataset, contacts_frame, agent, robot)
