#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import serow
import os

from env import SerowEnv
from ppo import PPO

from utils import(
    Normalizer,
    export_models_to_onnx
)

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
        self.log_std_layer = nn.Linear(128, self.action_dim)
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
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
        x = F.relu(self.layer4(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-4, 1)
        
        return mean, log_std
    
    def get_distribution(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)
    
    def _transform_actions(self, raw_actions):
        """Transform raw actions and compute log determinant of Jacobian"""
        
        # Apply transformations
        actions = torch.exp(raw_actions) + self.min_action
        
        # Compute log determinant of Jacobian
        # For exp transformation: d/dx exp(x) = exp(x)
        # Since the transformation is element-wise, the Jacobian is diagonal
        # det(J) = ∏_i exp(x_i) = exp(∑_i x_i)
        # log(det(J)) = ∑_i x_i
        log_det_jacobian = raw_actions.sum(dim=-1)

        return actions, log_det_jacobian
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        dist = self.get_distribution(state)
        
        if deterministic:
            raw_actions = dist.mean
        else:
            raw_actions = dist.sample()

        # Transform actions and get corrected log probability
        actions, log_det_jacobian = self._transform_actions(raw_actions)
        
        # Compute log probability of raw actions, then adjust for transformation
        raw_log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        log_prob = raw_log_prob - log_det_jacobian  # Change of variables formula
        
        return actions.squeeze(0).detach().cpu().numpy(), log_prob.squeeze(0).detach().cpu().item()
    
    def evaluate_actions(self, states, actions):
        """
        For evaluate_actions, we need to work backwards from the transformed actions
        to get the raw actions, then compute probabilities correctly.
        """
        dist = self.get_distribution(states)
        
        # Work backwards to get raw actions from transformed actions
        # The transformation is: actions = exp(raw_actions) + min_action
        # So the inverse is: raw_actions = log(actions - min_action)
        raw_actions = torch.log(actions - self.min_action + 1e-8)
        
        # Compute log probabilities and adjust for transformation
        _, log_det_jacobian = self._transform_actions(raw_actions)
        raw_log_probs = dist.log_prob(raw_actions).sum(dim=-1)
        log_probs = raw_log_probs - log_det_jacobian
        
        # For entropy, we need to be more careful since it's not just a simple transformation
        # The entropy of the transformed distribution is:
        # H(Y) = H(X) - E[log|det(J)|] where Y = f(X) and J is the Jacobian
        raw_entropy = dist.entropy().sum(dim=-1)
        entropy = raw_entropy - log_det_jacobian
        
        return log_probs, entropy
    

class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(params['state_dim'], 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 128)
        
        # Initialize weights with smaller gains for value function
        nn.init.orthogonal_(self.layer1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.layer4.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.layer1.bias, 0.0)
        torch.nn.init.constant_(self.layer2.bias, 0.0)
        torch.nn.init.constant_(self.layer3.bias, 0.0)
        torch.nn.init.constant_(self.layer4.bias, 0.0)

        self.value_layer = nn.Linear(128, 1)
        # Use smaller gain for value function to prevent initial large predictions
        nn.init.orthogonal_(self.value_layer.weight, gain=0.1)
        torch.nn.init.constant_(self.value_layer.bias, 0.0)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
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
        kinematics = dataset['kinematics']
        
        # Set proper limits on number of episodes
        max_episodes = params['max_episodes']
        max_steps = len(imu_measurements) - 1 

        for episode in range(max_episodes):
            serow_env = SerowEnv(robot, joint_states[0], base_states[0], contact_states[0],  
                                 params['action_dim'], params['state_dim'], 
                                 params['history_buffer_size'], params['state_normalizer'])
            
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
                        action, value, log_prob = agent.get_action(x, deterministic=False)

                        # Run the update step
                        post_state, reward, done = serow_env.update_step(cf, kin, action, gt, time_step, max_steps)

                        # Compute the next state
                        next_x = serow_env.compute_state(cf, post_state, next_kin)

                        # Add to buffer
                        agent.add_to_buffer(x, action, reward, next_x, done, value, log_prob)

                        # Accumulate rewards
                        collected_steps += 1
                        episode_return += reward
                    else:
                        action = np.zeros(serow_env.action_dim)
                        # Just run the update step
                        post_state, _, _ = serow_env.update_step(cf, kin, action, gt, time_step, max_steps)

                    if done:
                        break

                # Finish the update
                serow_env.finish_update(imu, kin)

                # Train policy periodically
                if collected_steps >= params['n_steps']:
                    actor_loss, critic_loss, entropy, converged = agent.learn()
                    if actor_loss is not None and critic_loss is not None:
                        print(f"[Episode {episode + 1}/{max_episodes}] Step [{time_step + 1}/{max_steps}] Policy Loss: {actor_loss}, Value Loss: {critic_loss}, Entropy: {entropy}")
                    collected_steps = 0

                # Check for early termination due to filter divergence
                if done:
                    print(f"Episode {episode + 1} terminated early at step {time_step + 1}/{max_steps} due to " 
                          f"filter divergence")
                    break
                
            print(f"Episode {episode + 1}/{max_episodes}, Step {time_step + 1}/{max_steps}, " 
                  f"Episode return: {episode_return}, Best: {best_return}, " 
                  f"in episode {best_return_episode}, "
                  f"Normalization stats: {params['state_normalizer'].get_normalization_stats() if params['state_normalizer'] is not None else 'None'}")

            # End of episode processing
            if episode_return > best_return:
                best_return = episode_return
                best_return_episode = episode
                if params['state_normalizer'] is not None:
                    params['state_normalizer'].save_stats(agent.checkpoint_dir, '/state_normalizer')
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
            if params['state_normalizer'] is not None:  
                params['state_normalizer'].save_stats(agent.checkpoint_dir, '/state_normalizer')
            break  # Break out of dataset loop
    
    # Plot training curves
    agent.logger.plot_training_curves()
    return agent

if __name__ == "__main__":
    # Load and preprocess the data
    dataset = np.load("go2_training_dataset.npz", allow_pickle=True)
    test_dataset = dataset
    imu_measurements = dataset['imu']
    contact_states = dataset['contact_states']
    dt = dataset['dt']
    dataset_size = len(imu_measurements) - 1

    # Define the dimensions of your state and action spaces
    normalizer = None
    history_buffer_size = 10
    state_dim = 3 + 3 * 3 + 3 * 3 * history_buffer_size + 3 * history_buffer_size
    action_dim = 3  # Based on the action vector used in ContactEKF.setAction()
    min_action = np.array([1e-8, 1e-8, 1e-8])
    max_action = np.array([1e2, 1e2, 1e2])
    robot = "go2"

    # Create the evaluation environment and get the contacts frames
    serow_env = SerowEnv(robot, dataset['joint_states'][0], dataset['base_states'][0], 
                         dataset['contact_states'][0], action_dim, state_dim, 
                         history_buffer_size, normalizer)
    contact_frames = serow_env.contact_frames
    print(f"Contacts frame: {contact_frames}")
    train_datasets = [dataset]
    device = 'cpu'

    max_episodes = 2
    n_steps = 512
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
        'clip_param': 0.2,  
        'value_clip_param': 0.2,
        'value_loss_coef': 0.35,  
        'entropy_coef': 0.01,  
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_epochs': 5,  
        'batch_size': 64,  
        'max_grad_norm': 0.5,  
        'buffer_size': 10000,  
        'max_episodes': max_episodes,
        'actor_lr': 1e-5, 
        'critic_lr': 1e-5,
        'target_kl': 0.03,
        'n_steps': n_steps,
        'convergence_threshold': 0.15,
        'critic_convergence_threshold': 0.15,
        'return_window_size': 10,
        'value_loss_window_size': 10,
        'checkpoint_dir': 'policy/ppo',
        'total_steps': total_steps, 
        'final_lr_ratio': 0.01,
        'check_value_loss': True,
        'total_training_steps': total_training_steps,
        'dt': dt,
        'state_normalizer': normalizer
    }

    loaded = False
    print(f"Initializing agent for {robot}")
    actor = Actor(params).to(device)
    critic = Critic(params).to(device)
    agent = PPO(actor, critic, params, device=device)

    policy_path = params['checkpoint_dir']
    # Try to load a trained policy for this robot if it exists
    try:
        agent.load_checkpoint(f'{policy_path}/trained_policy_{robot}.pth')
        if params['state_normalizer'] is not None:
            params['state_normalizer'].load_stats(policy_path, '/state_normalizer')
            print(f"Stats loaded: {params['state_normalizer'].get_normalization_stats()}")
        print(f"Loaded trained policy for {robot} from '{policy_path}/trained_policy_{robot}.pth'")
        loaded = True
    except FileNotFoundError:
        print(f"No trained policy found for {robot}. Training new policy...")

    if not loaded:
        # Train the policy
        if params['state_normalizer'] is not None:
            params['state_normalizer'].reset_stats()
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
