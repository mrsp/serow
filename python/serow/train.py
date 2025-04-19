#!/usr/bin/env python3

import numpy as np
from serow import ContactEKF
from ddpg import DDPG
from read_mcap import read_initial_base_state, read_kinematic_measurements, read_imu_measurements
import matplotlib.pyplot as plt
import torch
from ac import Actor, Critic

def train_policy(initial_state, contacts_frame, point_feet, g, imu_rate, outlier_detection, agent, kinematic_measurements, imu_measurements):
    # Initialize the EKF
    ekf.init(initial_state, contacts_frame, point_feet, g, imu_rate, outlier_detection)
    
    # Initialize the state
    state = initial_state   

    # Run a few prediction/update steps
    step = 0
    for imu, kin in zip(imu_measurements, kinematic_measurements):
        # Get the current state - properly format as a flat array
        x = np.concatenate([
            state.base_position,  # 3D position
            state.base_linear_velocity,  # 3D velocity
            state.base_orientation  # 4D quaternion
        ])
        
        # Set action
        action = agent.get_action(x)
        ekf.set_action(action)

        # Predict step
        ekf.predict(state, imu, kin)
        
        # Update step (pass None for both optional parameters)
        ekf.update(state, kin, None, None)

        # Compute the reward based on NIS
        reward = 0
        for cf in contacts_frame:
            innovation = np.zeros(3)
            covariance = np.zeros((3, 3))
            success, innovation, covariance = ekf.get_contact_position_innovation(cf)
            if success:
                nis = innovation.dot(np.linalg.inv(covariance).dot(innovation))
                # Ideal NIS value is 3 (dimension of measurement)
                # Reward peaks at NIS = 3 and decreases as NIS deviates from 3
                # Using a Gaussian-like reward centered at 3
                reward += np.exp(-0.5 * (nis - 3.0)**2)

        # Compute the next state
        next_x = np.concatenate([
            state.base_position,
            state.base_linear_velocity,
            state.base_orientation
        ])

        # Add to buffer
        agent.add_to_buffer(x, action, reward, next_x, False)

        print(f"\nStep {step}:")
        print(f"reward: {reward}")

        # Train the agent
        agent.train()
        step += 1

def evaluate_policy(initial_state, contacts_frame, point_feet, g, imu_rate, outlier_detection, agent, kinematic_measurements, imu_measurements, save_policy=False):
        # After training, evaluate the policy
        print("\nEvaluating trained policy...")
        
        # Reset to initial state
        state = initial_state
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
            for cf in contacts_frame:
                innovation = np.zeros(3)
                covariance = np.zeros((3, 3))
                success, innovation, covariance = ekf.get_contact_position_innovation(cf)
                if success:
                    nis = innovation.dot(np.linalg.inv(covariance).dot(innovation))
                    # Ideal NIS value is 3 (dimension of measurement)
                    # Reward peaks at NIS = 3 and decreases as NIS deviates from 3
                    # Using a Gaussian-like reward centered at 3
                    reward += np.exp(-0.5 * (nis - 3.0)**2)
            
            
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
    initial_state = read_initial_base_state("/tmp/serow_proprioception.mcap")

    # Get the contacts frame
    contacts_frame = set(initial_state.contacts_position.keys())
    print(f"Contacts frame: {contacts_frame}")
    
    # Initialize the EKF
    ekf = ContactEKF()

    # Initialize the EKF
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
    agent = DDPG(actor, critic, state_dim, action_dim, max_action, min_action)

    # Try to load a trained policy if it exists
    try:
        checkpoint = torch.load('trained_policy.pth')
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print("Loaded trained policy from 'trained_policy.pth'")
    except FileNotFoundError:
        print("No trained policy found. Training new policy...")
        train_policy(initial_state, contacts_frame, point_feet, g, imu_rate, outlier_detection, agent, kinematic_measurements, imu_measurements)
        # Save the trained policy
        evaluate_policy(initial_state, contacts_frame, point_feet, g, imu_rate, outlier_detection, agent, kinematic_measurements, imu_measurements, save_policy=True)
    else:
        # Just evaluate the loaded policy
        evaluate_policy(initial_state, contacts_frame, point_feet, g, imu_rate, outlier_detection, agent, kinematic_measurements, imu_measurements, save_policy=False)
