#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import torch
from serow import ContactEKF
from ddpg import DDPG
from read_mcap import read_base_states, read_kinematic_measurements, read_imu_measurements
from ac import Actor, Critic

def train_policy(datasets, contacts_frame, point_feet, g, imu_rate, outlier_detection, agent):
    for dataset in datasets:
        # Get the kinematic measurements and imu measurements from the dataset
        kinematic_measurements = dataset['kinematic']
        imu_measurements = dataset['imu']
        
        # Reset to initial state
        initial_state = dataset['states'][0]

        best_reward = float('-inf')
        max_episode = 40
        
        # Convergence check parameters
        window_size = 10  # Number of episodes to consider for convergence
        reward_window = []  # Store rewards for convergence check
        convergence_threshold = 0.1  # Threshold for reward variation
        min_episodes = 20  # Minimum episodes before checking convergence

        for episode in range(max_episode):            
            # Initialize the EKF
            ekf = ContactEKF()
            ekf.init(initial_state, contacts_frame, point_feet, g, imu_rate, outlier_detection)
            
            # Initialize the state
            state = initial_state   

            episode_reward = 0.0
            episode_steps = 0
            done = False
            for imu, kin in zip(imu_measurements, kinematic_measurements):
                # Get the current state - properly format as a flat array
                x = np.concatenate([
                    state.base_position,  # 3D position
                    state.base_linear_velocity,  # 3D velocity
                    state.base_orientation  # 4D quaternion
                ])
                
                # Set action
                action = agent.get_action(x, add_noise=True)
                ekf.set_action(action)

                # Predict step
                ekf.predict(state, imu, kin)
                
                # Update step (pass None for both optional parameters)
                ekf.update(state, kin, None, None)

                # Compute the reward based on NIS
                reward = float('-inf')
                nis = []
                for cf in contacts_frame:
                    success = False
                    innovation = np.zeros(3)
                    covariance = np.zeros((3, 3))
                    success, innovation, covariance = ekf.get_contact_position_innovation(cf)
                    if success:
                        # nis.append(innovation.dot(innovation))
                        nis.append(innovation.dot(np.linalg.inv(covariance).dot(innovation)))

                if len(nis) == 0:
                    continue
                
                reward = -np.mean(nis)
                episode_reward += reward
                episode_steps += 1

                # Compute the next state
                next_x = np.concatenate([
                    state.base_position,
                    state.base_linear_velocity,
                    state.base_orientation
                ])

                # Add to buffer
                agent.add_to_buffer(x, action, reward, next_x, done)

                # Train the agent
                agent.train()
                
                if done:
                    break
                
            if episode_reward > best_reward:
                best_reward = episode_reward
                
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Best Reward: {best_reward:.2f}")
            
            # Update reward window and check convergence
            reward_window.append(episode_reward)
            if len(reward_window) > window_size:
                reward_window.pop(0)
                
            if episode >= min_episodes and len(reward_window) == window_size:
                # Calculate mean and standard deviation of rewards in the window
                mean_reward = np.mean(reward_window)
                std_reward = np.std(reward_window)
                
                # Check if rewards have converged (low variation)
                if std_reward < convergence_threshold:
                    print(f"\nTraining converged after {episode + 1} episodes!")
                    print(f"Mean reward in last {window_size} episodes: {mean_reward:.2f}")
                    print(f"Standard deviation: {std_reward:.2f}")
                    break
        break

def evaluate_policy(dataset, contacts_frame, point_feet, g, imu_rate, outlier_detection, agent, save_policy=False):
        # After training, evaluate the policy
        print("\nEvaluating trained policy...")
        
        # Get the kinematic measurements and imu measurements from the dataset
        kinematic_measurements = dataset['kinematic']
        imu_measurements = dataset['imu']
        
        # Reset to initial state
        state = dataset['states'][0]
        
        # Initialize the EKF
        ekf = ContactEKF()
        ekf.init(state, contacts_frame, point_feet, g, imu_rate, outlier_detection)
        
        # Store trajectories for visualization
        positions = []
        velocities = []
        orientations = []
        rewards = []
        
        # Run evaluation episodes
        step = 0
        for imu, kin in zip(imu_measurements, kinematic_measurements):
            # Get state and action
            x = np.concatenate([
                state.base_position,
                state.base_linear_velocity,
                state.base_orientation
            ])
            
            # Get action from trained policy
            action = agent.get_action(x)
            ekf.set_action(action)
            
            # Predict and update
            ekf.predict(state, imu, kin)
            ekf.update(state, kin, None, None)
            
            # Store trajectories
            positions.append(np.array(state.base_position))
            velocities.append(np.array(state.base_linear_velocity))
            orientations.append(np.array(state.base_orientation))
            
            # Print state changes every 100 steps
            if step % 100 == 0:
                print(f"\nStep {step}:")
                print(f"Position: {state.base_position}")
                print(f"Velocity: {state.base_linear_velocity}")
                print(f"Action: {action}")
            
            # Calculate reward
            reward = 0
            nis = []
            for cf in contacts_frame:
                innovation = np.zeros(3)
                covariance = np.zeros((3, 3))
                success, innovation, covariance = ekf.get_contact_position_innovation(cf)
                if success:
                    nis.append(innovation.dot(np.linalg.inv(covariance).dot(innovation)))
            
            if len(nis) == 0:
                continue
            
            reward = -np.mean(nis)
            rewards.append(reward)
            step += 1
            
        # Convert to numpy arrays for plotting
        positions = np.array(positions)
        velocities = np.array(velocities)
        orientations = np.array(orientations)
        rewards = np.array(rewards)

        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Position trajectory
        plt.subplot(2, 2, 1)
        plt.plot(positions[:, 0], label='x')
        plt.plot(positions[:, 1], label='y')
        plt.plot(positions[:, 2], label='z')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.title('Base Position Trajectory')
        plt.legend()
        plt.grid(True)
        
        # Velocity components
        plt.subplot(2, 2, 2)
        plt.plot(velocities[:, 0], label='Vx')
        plt.plot(velocities[:, 1], label='Vy')
        plt.plot(velocities[:, 2], label='Vz')
        plt.xlabel('Time Step')
        plt.ylabel('Velocity')
        plt.title('Base Linear Velocity')
        plt.legend()
        plt.grid(True)
        
        # Orientation (quaternion components)
        plt.subplot(2, 2, 3)
        plt.plot(orientations[:, 0], label='w')
        plt.plot(orientations[:, 1], label='x')
        plt.plot(orientations[:, 2], label='y')
        plt.plot(orientations[:, 3], label='z')
        plt.xlabel('Time Step')
        plt.ylabel('Quaternion Components')
        plt.title('Base Orientation')
        plt.legend()
        plt.grid(True)
        
        # Reward over time
        plt.subplot(2, 2, 4)
        plt.plot(rewards)
        plt.xlabel('Time Step')
        plt.ylabel('Reward')
        plt.title('Reward Over Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print evaluation metrics
        print("\nPolicy Evaluation Metrics:")
        print(f"Average Reward: {np.mean(rewards):.4f}")
        print(f"Max Reward: {np.max(rewards):.4f}")
        print(f"Min Reward: {np.min(rewards):.4f}")
        print(f"Final Position: {positions[-1]}")
        print(f"Final Velocity: {velocities[-1]}")
        print(f"Final Orientation: {orientations[-1]}")

        # Save the trained policy
        if (save_policy):
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
            }, 'trained_policy.pth')

if __name__ == "__main__":
    # Read the measurement mcap file
    kinematic_measurements = read_kinematic_measurements("/tmp/serow_measurements.mcap")
    imu_measurements  = read_imu_measurements("/tmp/serow_measurements.mcap")
    states = read_base_states("/tmp/serow_proprioception.mcap")

    # Calculate the size of each dataset
    total_size = len(kinematic_measurements)
    N = 10  # Number of datasets
    dataset_size = total_size // N  # Size of each dataset

    # Create N contiguous datasets
    train_datasets = []
    for i in range(N):
        start_idx = i * dataset_size
        end_idx = start_idx + dataset_size
        
        # Create a dataset with measurements and states from start_idx to end_idx
        dataset = {
            'kinematic': kinematic_measurements[start_idx:end_idx],
            'imu': imu_measurements[start_idx:end_idx],
            'states': states[start_idx:end_idx]
        }
        train_datasets.append(dataset)

    # Pick a random dataset for testing
    test_dataset = train_datasets[np.random.randint(0, N)]
    train_datasets.remove(test_dataset)

    # Get the contacts frame
    contacts_frame = set(states[0].contacts_position.keys())
    print(f"Contacts frame: {contacts_frame}")
        
    # Parameters
    point_feet = True  # Assuming point feet or flat feet
    g = 9.81  # Gravity constant
    imu_rate = 500.0  # IMU update rate in Hz
    outlier_detection = False  # Enable outlier detection

    # Define the dimensions of your state and action spaces
    state_dim = 10  # 3 position, 3 velocity, 4 orientation
    action_dim = 7  # Based on the action vector used in ContactEKF.setAction()
    max_action = 100.0  # Maximum value for actions
    min_action = 1e-6  # Minimum value for actions

    # Initialize the actor and critic
    actor = Actor(state_dim, action_dim, max_action)
    critic = Critic(state_dim, action_dim)

    # Initialize the DDPG agent
    agent = DDPG(actor, critic, state_dim, action_dim, max_action, min_action, device='cuda')

    # Try to load a trained policy if it exists
    try:
        checkpoint = torch.load('trained_policy.pth')
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print("Loaded trained policy from 'trained_policy.pth'")
    except FileNotFoundError:
        print("No trained policy found. Training new policy...")
        train_policy(train_datasets, contacts_frame, point_feet, g, imu_rate, outlier_detection, agent)
        # Save the trained policy
        evaluate_policy(test_dataset, contacts_frame, point_feet, g, imu_rate, outlier_detection, agent, save_policy=True)
    else:
        # Just evaluate the loaded policy
        evaluate_policy(test_dataset, contacts_frame, point_feet, g, imu_rate, outlier_detection, agent, save_policy=False)
