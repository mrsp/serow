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
    read_imu_measurements, 
    read_base_pose_ground_truth,
    read_joint_states
)
from utils import(
    train_policy,
    evaluate_policy,
    plot_trajectories,
    filter,
    quaternion_to_rotation_matrix
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
    plot_trajectories(timestamps, base_position_aligned, base_orientation_aligned, 
                      gt_position_aligned, gt_orientation_aligned, cumulative_rewards)

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

    convergence_threshold = 0.05  # How close to best reward we need to be (as a fraction) to mark convergence
    critic_convergence_threshold = 0.05  # How much the critic loss can vary before considering it converged
    window_size = 10  # Window size for convergence check
    train_freq = 100  # Call train() every train_freq steps

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
        'theta': None,
        'noise_sigma': None,
        'noise_decay': None, 
        'min_noise_sigma': None,
        'dt': dt,
        'convergence_threshold': convergence_threshold,
        'critic_convergence_threshold': critic_convergence_threshold,
        'window_size': window_size,
        'train_freq': train_freq,
        'max_episodes': 1000,
    }

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    loaded = False
    print(f"Initializing agent for {robot}")
    actor = Actor(params).to(device)
    critic = Critic(params).to(device)
    agent = PPO(actor, critic, params, device=device, normalize_state=True)

    policy_path = 'policy/ppo'
    # Try to load a trained policy for this robot if it exists
    try:
        checkpoint = torch.load(f'{policy_path}/final/trained_policy_{robot}.pth', weights_only=True)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded trained policy for {robot} from '{policy_path}/final/trained_policy_{robot}.pth'")
        loaded = True
    except FileNotFoundError:
        print(f"No trained policy found for {robot}. Training new policy...")

    if not loaded:
        # Train the policy
        train_policy(train_datasets, contacts_frame, agent, robot, params, policy_path=policy_path)
        # Load the best policy
        checkpoint = torch.load(f'{policy_path}/final/trained_policy_{robot}.pth', weights_only=True)
        actor = Actor(params).to(device)
        critic = Critic(params).to(device)
        best_policy = PPO(actor, critic, params, device=device)
        best_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        best_policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded optimal trained policy for {robot} from '{policy_path}/final/trained_policy_{robot}.pth'")
        evaluate_policy(test_dataset, contacts_frame, best_policy, robot, policy_path)
    else:
        # Just evaluate the loaded policy
        evaluate_policy(test_dataset, contacts_frame, agent, robot, policy_path)
