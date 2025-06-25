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

class Actor(nn.Module):
    def __init__(self, params):
        super(Actor, self).__init__()
        self.state_dim = params['state_dim']
        self.device = params['device']
        self.action_dim = params['action_dim']
        self.min_action = torch.from_numpy(params['min_action']).to(self.device)
        self.max_action = torch.from_numpy(params['max_action']).to(self.device)
        
        # Shared layers
        self.layer1 = nn.Linear(self.state_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 32)
        
        # Output layers
        self.mean_layer = nn.Linear(32, self.action_dim)
        self.log_std_layer = nn.Linear(32, self.action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.layer1, self.layer2, self.layer3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, -1.0)
    
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-4, 1)
        
        return mean, log_std
    
    def get_distribution(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        dist = self.get_distribution(state)
        
        if deterministic:
            raw_actions = dist.mean
        else:
            raw_actions = dist.sample()
        
        # Clip actions
        actions = torch.clamp(raw_actions, self.min_action, self.max_action)
        
        # Compute log probability on the clipped actions
        log_prob = dist.log_prob(actions).sum(dim=-1)
        
        return actions.squeeze(0).detach().cpu().numpy(), log_prob.squeeze(0).detach().cpu().item()
    
    def evaluate_actions(self, states, actions):
        dist = self.get_distribution(states)
        
        # Clip actions
        actions = torch.clamp(actions, self.min_action, self.max_action)
        
        # Compute log probabilities with proper handling of boundary cases
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
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
            serow_env = SerowEnv(robot, joint_states[0], base_states[0], contact_states[0],  
                                 params['action_dim'], params['state_dim'], params['dt'])
            
            # Episode tracking variables
            episode_return = 0.0
            collected_steps = 0

            # Run episode
            for time_step, (imu, joints, ft, gt) in enumerate(zip(
                imu_measurements[:max_steps], 
                joint_measurements[:max_steps], 
                force_torque_measurements[:max_steps], 
                base_pose_ground_truth[:max_steps]
            )):
                contact_frames = serow_env.contact_frames
                rewards = {cf: None for cf in contact_frames}
                dones = {cf: None for cf in contact_frames}

                # Run the predict step
                kin, prior_state = serow_env.predict_step(imu, joints, ft)

                for cf in contact_frames:
                    if (prior_state.get_contact_position(cf) is not None and kin.contacts_status[cf]):
                    
                        # Compute the state
                        x = serow_env.compute_state(cf, prior_state, kin)

                        # Compute the action
                        action, value, log_prob = agent.get_action(x, deterministic=False)

                        # Run the update step
                        post_state, reward, done = serow_env.update_step(cf, kin, action, gt, time_step, max_steps)
                        rewards[cf] = reward
                        dones[cf] = done

                        # Compute the next state
                        next_x = serow_env.compute_state(cf, post_state, kin)

                        # Add to buffer
                        if reward is not None:
                            agent.add_to_buffer(x, action, reward, next_x, done, value, log_prob)

                    else:
                        action = np.zeros(serow_env.action_dim)
                        # Just run the update step
                        post_state, _, _ = serow_env.update_step(cf, kin, action, gt, time_step, max_steps)

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
                episode_return += step_reward
                
                # Train policy periodically
                if collected_steps >= params['n_steps']:
                    actor_loss, critic_loss, _, converged = agent.learn()
                    if actor_loss is not None and critic_loss is not None:
                        print(f"[Episode {episode + 1}/{max_episodes}] Step [{time_step + 1}/{max_steps}] Policy Loss: {actor_loss}, Value Loss: {critic_loss}")
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
    dataset_size = len(imu_measurements) - 1    
    for i in range(dataset_size):
        dt.append(imu_measurements[i+1].timestamp - imu_measurements[i].timestamp)
    dt = np.median(np.array(dt))
    print(f"dt: {dt}")

    # Define the dimensions of your state and action spaces
    state_dim = 4  
    action_dim = 6  # Based on the action vector used in ContactEKF.setAction()
    min_action = np.array([1e-10, 1e-10, 1e-10, -10, -10, -10])
    max_action = np.array([1e2, 1e2, 1e2, 10, 10, 10])
    robot = "go2"

    serow_env = SerowEnv(robot, joint_states[0], base_states[0], contact_states[0], action_dim, state_dim, dt)
    test_dataset = {
        'imu': imu_measurements,
        'joints': joint_measurements,
        'ft': force_torque_measurements,
        'base_states': base_states,
        'contact_states': contact_states,
        'joint_states': joint_states,
        'base_pose_ground_truth': base_pose_ground_truth
    }
    
    timestamps, base_positions, base_orientations, gt_positions, gt_orientations, cumulative_rewards, kinematics = \
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
    local_contact_positions = []
    contact_probabilities = []
    for (base_state, kin) in zip(base_states, kinematics):
        # Check if we have at least one contact
        if sum(kin.contacts_status[frame] for frame in contact_frames) > 0.0:
            R_base = quaternion_to_rotation_matrix(base_state.base_orientation).transpose()
            for cf in contact_frames:
                if base_state.contacts_position[cf] is not None \
                    and kin.contacts_position[cf] is not None \
                    and kin.contacts_status[cf] > 0.0:
                    local_pos = R_base @ (base_state.contacts_position[cf] - base_state.base_position)
                    local_kin_pos = kin.contacts_position[cf]
                    R = kin.contacts_position_noise[cf] + kin.position_cov
                    e = np.absolute(local_pos - local_kin_pos)
                    e = np.linalg.inv(R) @ e
                    local_contact_positions.append(e)
                    contact_probabilities.append(kin.contacts_probability[cf])

    # Convert to numpy array for easier manipulation
    local_contact_positions = np.array(local_contact_positions)
    contact_probabilities = np.array(contact_probabilities)

    # Create max and min state values with correct dimensions
    max_state_value = np.concatenate([np.max(local_contact_positions, axis=0), 
                                      [np.max(contact_probabilities)]])
    min_state_value = np.concatenate([np.min(local_contact_positions, axis=0), 
                                      [np.min(contact_probabilities)]])
    print(f"RL state max values: {max_state_value}")
    print(f"RL state min values: {min_state_value}")

    train_datasets = [test_dataset]
    device = 'cpu'

    max_episodes = 120
    n_steps = 1024
    total_steps = max_episodes * dataset_size * len(contact_frames)
    total_training_steps = total_steps // n_steps
    print(f"Total training steps: {total_training_steps}")

    params = {
        'device': device,
        'robot': robot,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'min_action': min_action,
        'clip_param': 0.2,  
        'value_clip_param': None,
        'value_loss_coef': 0.5,  
        'entropy_coef': 0.01,  
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_epochs': 5,  
        'batch_size': 64,  
        'max_grad_norm': 0.5,  
        'buffer_size': 10000,  
        'max_episodes': max_episodes,
        'actor_lr': 1e-6, 
        'critic_lr': 1e-5,  
        'target_kl': 0.05,
        'max_state_value': max_state_value,
        'min_state_value': min_state_value,
        'n_steps': n_steps,
        'convergence_threshold': 0.15,
        'critic_convergence_threshold': 0.15,
        'return_window_size': 10,
        'value_loss_window_size': 10,
        'checkpoint_dir': 'policy/ppo',
        'total_steps': total_steps, 
        'final_lr_ratio': 0.01,  # Learning rate will decay to 1% of initial value
        'check_value_loss': False,
        'total_training_steps': total_training_steps,
        'dt': dt
    }

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
        best_policy = PPO(actor, critic, params, device=device, normalize_state=True)
        best_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        best_policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded optimal trained policy for {robot} from " 
              f"'{policy_path}/trained_policy_{robot}.pth'")
        serow_env.evaluate(test_dataset, agent)
    else:
        # Just evaluate the loaded policy
        serow_env.evaluate(test_dataset, agent)
