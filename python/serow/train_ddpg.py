#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import serow
import os

from env import SerowEnv
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
    Normalizer,
    export_models_to_onnx
)

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.3):
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
        self.state_dim = params['state_dim']
        self.device = params['device']
        self.action_dim = params['action_dim']
        self.min_action = torch.FloatTensor(params['min_action']).to(self.device)
        self.max_action = torch.FloatTensor(params['max_action']).to(self.device)
        
        self.layer1 = nn.Linear(self.state_dim, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 128)
        
        # Output layers
        self.mean_layer = nn.Linear(128, self.action_dim) 
        self._init_weights()
        
        # Exploration noise
        self.noise = OUNoise(params['action_dim'], sigma=0.1)
        self.noise_scale = params['noise_scale']
        self.noise_decay = params['noise_decay']

        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

    def _init_weights(self):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)
    
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        mean = self.mean_layer(x) 
        # First three elements are the diagonal of the action covariance matrix and must be strictly positive
        diag = F.softplus(mean[:, :3]) + self.min_action[:3]
        off_diag = self.action_scale[3:] * F.tanh(mean[:, 3:]) + self.action_bias[3:]
        return torch.cat([diag, off_diag], dim=-1)
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        mean = self.forward(state)

        if deterministic:
            action = mean
        else:
            noise = torch.FloatTensor(self.noise.sample() * self.noise_scale).to(self.device)
            action = mean + noise
            self.noise_scale *= self.noise_decay
            action[:, :3] = torch.clamp(action[:, :3], min=self.min_action[:3])
            action[:, 3:] = torch.clamp(action[:, 3:], min=self.min_action[3:], max=self.max_action[3:])

        return action.squeeze(0).detach().cpu().numpy()
    
class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.state_layer = nn.Linear(params['state_dim'], 256)
        self.action_layer = nn.Linear(params['action_dim'], 256)

        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 1)
        nn.init.orthogonal_(self.state_layer.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.action_layer.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer4.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.state_layer.bias, 0.0)
        torch.nn.init.constant_(self.action_layer.bias, 0.0)
        torch.nn.init.constant_(self.layer2.bias, 0.0)
        torch.nn.init.constant_(self.layer3.bias, 0.0)
        torch.nn.init.constant_(self.layer4.bias, 0.0)

    def forward(self, state, action):
        s = F.relu(self.state_layer(state))
        a = F.relu(self.action_layer(action))
        x = torch.cat([s, a], dim=-1) # Concatenate state and action
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        # No activation on final layer to allow negative values
        return self.layer4(x)

def train_ddpg(datasets, agent, params):
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
        kinematics = dataset['kinematics']
        # Set proper limits on number of episodes
        max_episodes = params['max_episodes']
        max_steps = len(imu_measurements) - 1 

        # Warm-up phase: collect initial experiences without training
        warmup_episodes = 0

        for episode in range(max_episodes + warmup_episodes):
            serow_env = SerowEnv(robot, joint_states[0], base_states[0], contact_states[0],  
                                 params['action_dim'], params['state_dim'], params['history_buffer_size'])
            
            # Episode tracking variables
            episode_return = 0.0
            collected_steps = 0

            # Run episode
            for time_step, (imu, joints, ft, gt, next_kin) in enumerate(zip(
                imu_measurements[:max_steps], 
                joint_measurements[:max_steps], 
                force_torque_measurements[:max_steps], 
                base_pose_ground_truth[:max_steps],
                kinematics[1:max_steps+1]
            )):
                contact_frames = serow_env.contact_frames

                # Run the predict step
                kin, prior_state = serow_env.predict_step(imu, joints, ft)

                # Run the update step
                done = 0.0
                post_state = prior_state
                for cf in contact_frames:
                    if (post_state.get_contact_position(cf) is not None and kin.contacts_status[cf] and next_kin.contacts_status[cf]):
                        # Compute the state
                        x = serow_env.compute_state(cf, post_state, kin)

                        # Compute the action
                        if episode < warmup_episodes:
                            action = np.zeros((params['action_dim'],))
                        else:
                            action = agent.get_action(x, deterministic=False)

                        # Run the update step
                        post_state, reward, done = serow_env.update_step(cf, kin, action, gt, time_step, max_steps)

                        # Compute the next state
                        next_x = serow_env.compute_state(cf, post_state, next_kin)

                        # Add to buffer
                        agent.add_to_buffer(x, action, reward, next_x, done)
                        # Accumulate rewards
                        collected_steps += 1
                        episode_return += reward
                    else:
                        action = np.zeros((serow_env.action_dim, 1), dtype=np.float64)
                        # Just run the update step
                        post_state, _, _ = serow_env.update_step(cf, kin, action, gt, time_step, max_steps)

                    if done:
                        break

                # Finish the update
                serow_env.finish_update(imu, kin)
                
                # Train policy periodically
                if collected_steps >= params['n_steps']:
                    actor_loss, critic_loss, converged = agent.learn()
                    if actor_loss is not None and critic_loss is not None:
                        print(f"[Episode {episode + 1}/{max_episodes}] Step [{time_step + 1}/{max_steps}] Policy Loss: {actor_loss}, Value Loss: {critic_loss}")
                    collected_steps = 0

                # Check for early termination due to filter divergence
                if done:
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

    history_buffer_size = 10
    state_dim = 3 + 3 * 3 + 3 * 3 * history_buffer_size + 3 * history_buffer_size
    action_dim = 6  # Based on the action vector used in ContactEKF.setAction()
    min_action = np.array([1e-10, 1e-10, 1e-10, -1e2, -1e2, -1e2])
    max_action = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2])
    robot = "go2"
    
    normalizer = Normalizer()
    serow_env = SerowEnv(robot, joint_states[0], base_states[0], contact_states[0], action_dim, state_dim, history_buffer_size)
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
    test_dataset['kinematics'] = kinematics

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

    train_datasets = [test_dataset]
    device = 'cpu'

    max_episodes = 1200
    n_steps = 256
    total_steps = max_episodes * dataset_size * len(contact_frames)
    total_training_steps = total_steps // n_steps
    print(f"Total training steps: {total_training_steps}")

    params = {
        'history_buffer_size': history_buffer_size,
        'device': device,
        'robot': robot,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'min_action': min_action,
        'gamma': 0.99,
        'batch_size': 64,  
        'max_grad_norm': 1.0, 
        'tau': 0.01,
        'buffer_size': 1000000,
        'max_episodes': max_episodes,
        'actor_lr': 1e-5, 
        'critic_lr': 1e-5,  
        'noise_scale': 0.1,
        'noise_decay': 0.9999,
        'n_steps': n_steps,
        'train_for_batches': 5,
        'convergence_threshold': 0.25,
        'critic_convergence_threshold': 0.15,
        'return_window_size': 20,
        'value_loss_window_size': 20,
        'checkpoint_dir': 'policy/ddpg',
        'total_steps': total_steps, 
        'final_lr_ratio': 0.01,  # Learning rate will decay to 1% of initial value
        'check_value_loss': False,
        'total_training_steps': total_training_steps,
    }

    loaded = False
    print(f"Initializing agent for {robot}")
    actor = Actor(params).to(device)
    critic = Critic(params).to(device)
    normalize_state = False
    agent = DDPG(actor, critic, params, device=device)

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
        train_ddpg(train_datasets, agent, params)
        # Load the best policy
        checkpoint = torch.load(f'{policy_path}/trained_policy_{robot}.pth')
        actor = Actor(params).to(device)
        critic = Critic(params).to(device)
        best_policy = DDPG(actor, critic, params, device=device)
        best_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        best_policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded optimal trained policy for {robot} from " 
              f"'{policy_path}/trained_policy_{robot}.pth'")
        serow_env.evaluate(test_dataset, agent)
    else:
        # Just evaluate the loaded policy
        serow_env.evaluate(test_dataset, agent)
