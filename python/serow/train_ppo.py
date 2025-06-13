#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import serow
import os

from env import SerowEnv
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
    quaternion_to_rotation_matrix,
    export_models_to_onnx
)

class SharedNetwork(nn.Module):
    def __init__(self, params):
        super(SharedNetwork, self).__init__()
        self.layer1 = nn.Linear(params['state_dim'], 64)
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
    
# Actor network per leg end-effector
class Actor(nn.Module):
    def __init__(self, params):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(params['state_dim'], 64)
        self.layer2 = nn.Linear(64, 64)
        nn.init.orthogonal_(self.layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer2.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.layer1.bias, 0.0)
        torch.nn.init.constant_(self.layer2.bias, 0.0)

        self.mean_layer = nn.Linear(64, params['action_dim'])
        self.log_std = nn.Parameter(torch.zeros(params['action_dim']))

        self.min_action = params['min_action']
        self.action_dim = params['action_dim']

        # Initialize weights with larger gain for better initial exploration
        nn.init.orthogonal_(self.mean_layer.weight, gain=np.sqrt(2))  
        torch.nn.init.constant_(self.mean_layer.bias, 0.0)

    def forward(self, state):
        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        mean = self.mean_layer(x) 
        # Clamp log_std for numerical stability
        log_std = self.log_std.clamp(-20, 2)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
        
        # Use softplus with a small epsilon for numerical stability
        action_scaled = F.softplus(action) + self.min_action
        
        # Calculate log probability
        if not deterministic:
            # Log prob with softplus correction
            log_prob = normal.log_prob(action).sum(dim=-1)
            # Apply softplus correction (derivative of softplus is sigmoid)
            log_prob -= torch.log(torch.sigmoid(action) + 1e-8).sum(dim=-1)
        else:
            log_prob = torch.zeros(1)
        
        return action_scaled.detach().cpu().numpy(), log_prob.detach().cpu().item()

    def evaluate_actions(self, states, actions):
        """Evaluate log probabilities and entropy for given state-action pairs"""
        mean, log_std = self.forward(states)
        std = log_std.exp()
        
        # Convert actions back to raw space (inverse of softplus scaling)
        actions_raw = torch.log(actions - self.min_action + 1e-8)
        
        # Calculate log probabilities
        normal = torch.distributions.Normal(mean, std)
        log_probs = normal.log_prob(actions_raw).sum(dim=-1, keepdim=True)
        
        # Apply softplus correction (derivative of softplus is sigmoid)
        log_probs -= torch.log(torch.sigmoid(actions_raw) + 1e-8).sum(dim=-1, keepdim=True)
        
        # Calculate entropy
        entropy = normal.entropy().sum(dim=-1)
        return log_probs, entropy

class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(params['state_dim'], 128)
        self.layer2 = nn.Linear(128, 128)
        nn.init.orthogonal_(self.layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer2.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.layer1.bias, 0.0)
        torch.nn.init.constant_(self.layer2.bias, 0.0)

        self.value_layer = nn.Linear(128, 1)
        nn.init.orthogonal_(self.value_layer.weight, gain=1.0)
        torch.nn.init.constant_(self.value_layer.bias, 0.0)

    def forward(self, state):
        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return self.value_layer(x)

def train_ppo(datasets, agent, params):
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

        # Set proper limits on number of episodes
        max_episodes = params['max_episodes']
        max_steps = len(imu_measurements) - 1 

        for episode in range(max_episodes):
            serow_env = SerowEnv(robot, joint_states[0], base_states[0], contact_states[0])
            
            # Episode tracking variables
            episode_return = 0.0
            collected_steps = 0

            # Run episode
            for time_step, (imu, joints, ft, gt, cs, next_cs) in enumerate(zip(
                imu_measurements[:max_steps], 
                joint_measurements[:max_steps], 
                force_torque_measurements[:max_steps], 
                base_pose_ground_truth[:max_steps],
                contact_states[:max_steps],
                contact_states[1:max_steps+1]
            )):
                contact_frames = serow_env.contact_frames
                rewards = {cf: None for cf in contact_frames}
                dones = {cf: None for cf in contact_frames}

                # Run the predict step
                kin, prior_state = serow_env.predict_step(imu, joints, ft)

                for cf in contact_frames:
                    if (prior_state.get_contact_position(cf) is not None 
                            and cs.contacts_status[cf] and next_cs.contacts_status[cf]):
                    
                        # Compute the state
                        x = serow_env.compute_state(cf, prior_state, cs)

                        # Compute the action
                        action, value, log_prob = agent.get_action(x, deterministic=False)

                        # Run the update step
                        post_state, reward, done = serow_env.update_step(cf, kin, action, gt, time_step)
                        rewards[cf] = reward
                        dones[cf] = done

                        # Compute the next state
                        next_x = serow_env.compute_state(cf, post_state, next_cs)

                        # Add to buffer
                        if reward is not None:
                            agent.add_to_buffer(x, action, reward, next_x, done, value, log_prob)

                    else:
                        action = np.ones(serow_env.action_dim)
                        # Just run the update step
                        post_state, _, _ = serow_env.update_step(cf, kin, action, gt, time_step)

                    # Update the prior state
                    prior_state = post_state

                # Finish the update
                serow_env.finish_update(imu, kin)

                # Accumulate rewards
                step_reward = 0
                for _, reward in rewards.items():
                    if reward is not None:
                        step_reward += reward
                        collected_steps += 1
                episode_return = params['gamma'] * episode_return + step_reward
                
                # Train policy periodically
                if collected_steps >= params['n_steps']:
                    actor_loss, critic_loss, _, converged = agent.train()
                    if actor_loss is not None and critic_loss is not None:
                        print(f"Policy Loss: {actor_loss:.4f}, Value Loss: {critic_loss:.4f}")
                    collected_steps = 0

                # Check for early termination due to filter divergence
                terminated = False
                for cf in contact_frames:
                    if dones[cf] is not None and dones[cf]:
                        terminated = True
                        break
                
                if terminated:
                    print(f"Episode {episode + 1} terminated early at step {time_step + 1}/{max_steps} due to " 
                          f"filter divergence")
                    break
                
            print(f"Episode {episode + 1}/{max_episodes}, Step {time_step + 1}/{max_steps}, " 
                  f"Episode return: {episode_return}, Best: {best_return}, " 
                  f"in episode {best_return_episode}")

            # End of episode processing
            if episode_return > best_return:
                best_return = episode_return
                best_return_episode = episode
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
            break  # Break out of dataset loop
    
    # Plot training curves
    agent.logger.plot_training_curves()
    return agent

if __name__ == "__main__":
    # Load and preprocess the data
    imu_measurements  = read_imu_measurements("/tmp/serow_measurements.mcap")
    joint_measurements = read_joint_measurements("/tmp/serow_measurements.mcap")
    force_torque_measurements = read_force_torque_measurements("/tmp/serow_measurements.mcap")
    base_pose_ground_truth = read_base_pose_ground_truth("/tmp/serow_measurements.mcap")
    base_states = read_base_states("/tmp/serow_proprioception.mcap")
    contact_states = read_contact_states("/tmp/serow_proprioception.mcap")
    joint_states = read_joint_states("/tmp/serow_proprioception.mcap")

    # Compute the dt
    dt = []
    for i in range(len(imu_measurements) - 1):
        dt.append(imu_measurements[i+1].timestamp - imu_measurements[i].timestamp)
    dt = np.median(np.array(dt))
    print(f"dt: {dt}")

    # Define the dimensions of your state and action spaces
    state_dim = 7  
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    min_action = 1e-10
    robot = "go2"

    serow_env = SerowEnv(robot, joint_states[0], base_states[0], contact_states[0])
    test_dataset = {
        'imu': imu_measurements,
        'joints': joint_measurements,
        'ft': force_torque_measurements,
        'base_states': base_states,
        'contact_states': contact_states,
        'joint_states': joint_states,
        'base_pose_ground_truth': base_pose_ground_truth
    }
    
    timestamps, base_positions, base_orientations, gt_positions, gt_orientations, cumulative_rewards = \
        serow_env.evaluate(test_dataset, agent=None)

    # Reform the ground truth data
    base_pose_ground_truth = []
    for i in range(len(timestamps)):
        gt = serow.BasePoseGroundTruth()
        gt.timestamp = timestamps[i]
        gt.position = gt_positions[i]
        gt.orientation = gt_orientations[i]
        base_pose_ground_truth.append(gt)

    # Get the contacts frame
    contact_frames = serow_env.contact_frames
    print(f"Contacts frame: {contact_frames}")

    # Compute max and min state values
    feet_positions = []
    base_linear_velocities = []
    for base_state in base_states:
        R_base = quaternion_to_rotation_matrix(base_state.base_orientation).transpose()
        base_linear_velocities.append(R_base @ base_state.base_linear_velocity)
        for cf in contact_frames:
            if base_state.contacts_position[cf] is not None:
                local_pos = R_base @ (base_state.base_position - base_state.contacts_position[cf])
                feet_positions.append(np.array([abs(local_pos[0]), abs(local_pos[1]), local_pos[2]]))
    
    # Convert base_linear_velocities and feet_positions to numpy array for easier manipulation
    base_linear_velocities = np.array(base_linear_velocities)
    feet_positions = np.array(feet_positions)
    contact_probabilities = []
    for contact_state in contact_states:
        for cf in contact_frames:
            if contact_state.contacts_status[cf]:
                contact_probabilities.append(contact_state.contacts_probability[cf])
    contact_probabilities = np.array(contact_probabilities)

    # Create max and min state values with correct dimensions
    # First 3 dimensions are for position, last dimension is for contact probability
    max_state_value = np.concatenate([np.max(feet_positions, axis=0), 
                                      np.max(base_linear_velocities, axis=0),
                                      [np.max(contact_probabilities)]])
    min_state_value = np.concatenate([np.min(feet_positions, axis=0), 
                                      np.min(base_linear_velocities, axis=0),
                                      [np.min(contact_probabilities)]])
    print(f"RL state max values: {max_state_value}")
    print(f"RL state min values: {min_state_value}")

    train_datasets = [test_dataset]

    params = {
        'robot': robot,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': None,
        'min_action': min_action,
        'clip_param': 0.2,  
        'value_clip_param': 0.2,
        'value_loss_coef': 0.5,  
        'entropy_coef': 0.01,  
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_epochs': 5,  
        'batch_size': 64,  
        'max_grad_norm': 0.3,  
        'buffer_size': 10000,  
        'max_episodes': 30,
        'actor_lr': 5e-5, 
        'critic_lr': 1e-4,  
        'max_state_value': max_state_value,
        'min_state_value': min_state_value,
        'update_lr': True,
        'n_steps': 512,
        'convergence_threshold': 0.1,
        'critic_convergence_threshold': 0.1,
        'return_window_size': 15,
        'value_loss_window_size': 15,
        'checkpoint_dir': 'policy/ppo',
        'total_steps': 10000, 
        'final_lr_ratio': 0.01,  # Learning rate will decay to 1% of initial value
        'check_value_loss': True,
    }

    device = 'cpu'
    loaded = False
    print(f"Initializing agent for {robot}")
    actor = Actor(params).to(device)
    critic = Critic(params).to(device)
    agent = PPO(actor, critic, params, device=device, normalize_state=True)

    policy_path = params['checkpoint_dir']
    # Try to load a trained policy for this robot if it exists
    try:
        agent.load_checkpoint(f'{policy_path}/trained_policy_{robot}.pth')
        print(f"Loaded trained policy for {robot} from '{policy_path}/trained_policy_{robot}.pth'")
        loaded = True
    except FileNotFoundError:
        print(f"No trained policy found for {robot}. Training new policy...")

    if not loaded:
        # Train the policy
        train_ppo(train_datasets, agent, params)
        # Load the best policy
        checkpoint = torch.load(f'{policy_path}/trained_policy_{robot}.pth')
        actor = Actor(params).to(device)
        critic = Critic(params).to(device)
        best_policy = PPO(actor, critic, params, device=device)
        best_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        best_policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded optimal trained policy for {robot} from " 
              f"'{policy_path}/trained_policy_{robot}.pth'")
        serow_env.evaluate(test_dataset, agent)
    else:
        # Just evaluate the loaded policy
        serow_env.evaluate(test_dataset, agent)
