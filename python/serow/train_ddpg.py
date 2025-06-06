#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import serow
import torch.serialization
from numpy.core.multiarray import scalar

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
    train_policy,
    evaluate_policy,
    plot_trajectories,
    filter,
    quaternion_to_rotation_matrix
)

torch.serialization.add_safe_globals([scalar])

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
        
        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.layer1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.layer2.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.layer3.weight, gain=np.sqrt(2))

        # Initialize bias to produce an output close to 1.0
        nn.init.constant_(self.mean_layer.bias, 1.0) 
        
        self.min_action = params['min_action']
        self.action_dim = params['action_dim']
        self.state_dim = params['state_dim']

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        x = self.dropout3(x)
        return F.relu(self.mean_layer(x)) 

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(next(self.parameters()).device)
            action = self.forward(state).cpu().numpy()[0]
            if not deterministic:
                action = action + self.noise.sample()

            # Ensure minimum action value and positivity
            action = np.maximum(action, self.min_action)
            return action

class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.state_layer = nn.Linear(params['state_dim'], 64)
        self.dropout1 = nn.Dropout(0.1)
        
        self.action_layer = nn.Linear(params['action_dim'], 64)
        self.dropout2 = nn.Dropout(0.1)
        
        self.combined_layer1 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.1)
        self.combined_layer2 = nn.Linear(64, 1)
        
        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.state_layer.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.action_layer.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.combined_layer1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.combined_layer2.weight, gain=np.sqrt(2))
        
        # Initialize final layer with small weights
        nn.init.constant_(self.combined_layer2.bias, 0.0)
    
    def forward(self, state, action):
        s = F.relu(self.state_layer(state))
        s = self.dropout1(s)
        a = F.relu(self.action_layer(action))
        a = self.dropout2(a)
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.combined_layer1(x))
        x = self.dropout3(x)
        return self.combined_layer2(x)

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

    params = {
        'robot': robot,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': None,
        'min_action': min_action,
        'gamma': 0.99,
        'tau': 0.0005,
        'batch_size': 128,  
        'actor_lr': 1e-5, 
        'critic_lr': 1e-5,  
        'buffer_size': 1000000,  
        'max_state_value': max_state_value,
        'min_state_value': min_state_value,
        'train_for_batches': 10,
        'dt': dt,
        'noise_sigma': 1.0,
        'theta': 0.15,
        'noise_decay': 0.985, 
        'min_noise_sigma': 1e-4,
        'convergence_threshold': convergence_threshold,
        'critic_convergence_threshold': critic_convergence_threshold,
        'window_size': window_size,
        'n_steps': 2000,
        'max_episodes': 2,
        'update_lr': True,
        'final_lr_ratio': 0.1,
        'total_steps': 1000000,
        'max_grad_norm': 0.5,
        'checkpoint_dir': 'policy/ddpg',
    }

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    loaded = False
    print(f"Initializing agent for {robot}")
    actor = Actor(params).to(device)
    critic = Critic(params).to(device)
    agent = DDPG(actor, critic, params, device=device, normalize_state=True)
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
        train_policy(train_datasets, contacts_frame, agent, robot, params, policy_path=policy_path)
        # Load the best policy
        checkpoint = torch.load(f'{policy_path}/trained_policy_{robot}.pth')
        actor = Actor(params).to(device)
        critic = Critic(params).to(device)
        best_policy = DDPG(actor, critic, params, device=device)
        best_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        best_policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded optimal trained policy for {robot} from '{policy_path}/trained_policy_{robot}.pth'")
        evaluate_policy(test_dataset, contacts_frame, best_policy, robot, policy_path)
    else:
        # Just evaluate the loaded policy
        evaluate_policy(test_dataset, contacts_frame, agent, robot, policy_path)
