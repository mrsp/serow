#!/usr/bin/env python3

import numpy as np
import serow
import matplotlib.pyplot as plt

from read_mcap import(
    read_base_states, 
    read_contact_states, 
    read_force_torque_measurements, 
    read_joint_measurements, 
    read_imu_measurements, 
    read_base_pose_ground_truth,
    read_joint_states
)

USE_GROUND_TRUTH = True

def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    
    Parameters: 
    R : numpy array with shape (3, 3)
        The rotation matrix
        
    Returns:
    numpy array with shape (4,)
        The corresponding quaternion
    """
    # Compute the trace of the matrix
    trace = np.trace(R)
    q = np.array([1.0, 0.0, 0.0, 0.0])

    # Check if the matrix is close to a pure rotation matrix
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2.0  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
        q = np.array([qw, qx, qy, qz])   
    else:
        # Compute the largest diagonal element
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # S=4*qx
            qx = 0.25 * S
            qy = (R[1, 0] + R[0, 1]) / S
            qz = (R[2, 0] + R[0, 2]) / S
            qw = (R[2, 1] - R[1, 2]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # S=4*qy  
            qy = 0.25 * S       
            qx = (R[1, 0] + R[0, 1]) / S
            qz = (R[2, 1] + R[1, 2]) / S
            qw = (R[0, 2] - R[2, 0]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # S=4*qz  
            qz = 0.25 * S
            qx = (R[2, 0] + R[0, 2]) / S
            qy = (R[2, 1] + R[1, 2]) / S
            qw = (R[1, 0] - R[0, 1]) / S
        q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a rotation matrix.
    
    Parameters:
    q : numpy array with shape (4,)
        The quaternion in the form [w, x, y, z]
        
    Returns:
    numpy array with shape (3, 3)
        The corresponding rotation matrix
    """
    # Ensure q is normalized
    q = q / np.linalg.norm(q)
    
    # Extract the values from q
    w, x, y, z = q
    
    # Compute the rotation matrix
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

def logMap(R):
    R11 = R[0, 0];
    R12 = R[0, 1];
    R13 = R[0, 2];
    R21 = R[1, 0];
    R22 = R[1, 1];
    R23 = R[1, 2];
    R31 = R[2, 0];
    R32 = R[2, 1];
    R33 = R[2, 2];

    trace = R.trace();

    omega = np.zeros(3)

    # Special case when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    if (trace + 1.0 < 1e-3) :
        if (R33 > R22 and R33 > R11) :
            # R33 is the largest diagonal, a=3, b=1, c=2
            W = R21 - R12;
            Q1 = 2.0 + 2.0 * R33;
            Q2 = R31 + R13
            Q3 = R23 + R32
            r = np.sqrt(Q1)
            one_over_r = 1 / r
            norm = np.sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W)
            sgn_w = -1.0 if W < 0 else 1.0
            mag = np.pi - (2 * sgn_w * W) / norm
            scale = 0.5 * one_over_r * mag
            omega = sgn_w * scale * np.array([Q2, Q3, Q1])
        elif (R22 > R11):
            # R22 is the largest diagonal, a=2, b=3, c=1
            W = R13 - R31;
            Q1 = 2.0 + 2.0 * R22;
            Q2 = R23 + R32;
            Q3 = R12 + R21;
            r = np.sqrt(Q1);
            one_over_r = 1 / r;
            norm = np.sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
            sgn_w = -1.0 if W < 0 else 1.0;
            mag = np.pi - (2 * sgn_w * W) / norm;
            scale = 0.5 * one_over_r * mag;
            omega = sgn_w * scale * np.array([Q3, Q1, Q2]);
        else:
            # R11 is the largest diagonal, a=1, b=2, c=3
            W = R32 - R23;
            Q1 = 2.0 + 2.0 * R11;
            Q2 = R12 + R21;
            Q3 = R31 + R13;
            r = np.sqrt(Q1);
            one_over_r = 1 / r;
            norm = np.sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
            sgn_w = -1.0 if W < 0 else 1.0;
            mag = np.pi - (2 * sgn_w * W) / norm;
            scale = 0.5 * one_over_r * mag;
            omega = sgn_w * scale * np.array([Q1, Q2, Q3])
    else:
        magnitude = 0.0;
        tr_3 = trace - 3.0;  # could be non-negative if the matrix is off orthogonal
        if (tr_3 < -1e-6):
            # this is the normal case -1 < trace < 3
            theta = np.arccos((trace - 1.0) / 2.0)
            magnitude = theta / (2.0 * np.sin(theta))
        else:
            # when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            # use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            # see https://github.com/borglab/gtsam/issues/746 for details
            magnitude = 0.5 - tr_3 / 12.0 + tr_3 * tr_3 / 60.0;

        omega = magnitude * np.array([R32 - R23, R13 - R31, R21 - R12]);
    return omega;

def normalize_quaternion(quaternion):
    """
    Normalize a quaternion to ensure it represents a valid rotation.
    
    Args:
        quaternion (list or numpy array): A quaternion [x, y, z, w]

    Returns:
        numpy array: A normalized quaternion
    """
    # Convert quaternion to numpy array if it's a list
    if isinstance(quaternion, list):
        quaternion = np.array(quaternion)

    # Normalize the quaternion
    quaternion = quaternion / np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion

    return quaternion

def normalize_vector(vector, min_value, max_value, target_range=(0, 1)):
    """
    Normalize a vector to ensure it is within the specified target range.
    
    Args:
        vector (numpy array or list): The vector to normalize
        min_value (numpy array or list): The minimum value of the input range. 
        max_value (numpy array or list): The maximum value of the input range. 
        target_range (tuple, optional): The target range (min, max) for normalization. Defaults to (0, 1)

    Returns:
        numpy array: The normalized vector
    """
    # Convert to numpy array if input is a list
    if isinstance(vector, list):
        vector = np.array(vector)
    if isinstance(min_value, list):
        min_value = np.array(min_value)
    if isinstance(max_value, list):
        max_value = np.array(max_value)

    # Check if min and max are different to avoid division by zero
    if np.all(min_value == max_value):
        return np.full_like(vector, target_range[0])
    
    # Normalize to [0, 1] first
    normalized = (vector - min_value) / (max_value - min_value)
    
    # Scale to target range
    return normalized * (target_range[1] - target_range[0]) + target_range[0]

def compute_reward(cf, serow_framework, state, gt, step):
    success = False
    innovation = np.zeros(3)
    covariance = np.zeros((3, 3))
    success, _, _, innovation, covariance = serow_framework.get_contact_position_innovation(cf)
    reward = None
    done = None

    if success:
        done = 0.0
        INNOVATION_SCALE = 10.0     
        POSITION_SCALE = 1000.0         
        ORIENTATION_SCALE = 5000.0     
        STEP_REWARD = 0.01
        DIVERGENCE_PENALTY = -5.0  
        TIME_SCALE = 0.01  # Controls how quickly the time penalty increases

        # Check if innovation is too large or if covariance is not positive definite
        try:
            # Add regularization to prevent numerical issues
            reg_covariance = covariance + np.eye(3) * 1e-6
            nis = innovation.dot(np.linalg.inv(reg_covariance).dot(innovation))
        except np.linalg.LinAlgError:
            nis = float('inf')
        
        # Handle divergence cases 
        if nis > 15.0 or nis <= 0.0: 
            reward = DIVERGENCE_PENALTY 
            done = 1.0  
        else:                
            # Main reward: Use bounded function to prevent extreme values
            # Map NIS to range [0, 1] using sigmoid-like function
            nis_normalized = 1.0 / (1.0 + INNOVATION_SCALE * nis)
            innovation_reward = nis_normalized  # Range: [0, 1]
            reward = innovation_reward + STEP_REWARD
            
            if USE_GROUND_TRUTH:
                # Calculate time-dependent scaling factor
                time_factor = 1.0 + TIME_SCALE * step;
                # print(f"Time factor: {time_factor}")
                
                # Position error component with bounded reward and time scaling
                position_error = np.linalg.norm(state.get_base_position() - gt.position)
                position_reward = 1.0 / (1.0 + POSITION_SCALE * position_error * time_factor)  # Range: [0, 1]
                # print(f"Position reward: {position_reward}")

                #  Orientation error component with bounded reward and time scaling
                # orientation_error = np.linalg.norm(
                #   logMap(quaternion_to_rotation_matrix(gt.orientation).transpose() @ 
                    # quaternion_to_rotation_matrix(state.get_base_orientation())))
                orientation_error = np.linalg.norm(state.get_base_orientation() - gt.orientation)
                orientation_reward = 1.0 / (1.0 + ORIENTATION_SCALE * orientation_error * time_factor)  # Range: [0, 1]
                # print(f"Orientation reward: {orientation_reward}")
                reward += position_reward + orientation_reward
            
            # Final reward is in reasonable range
            # print(f"[Reward Debug] cf={cf} Good estimate: reward={reward:.4f}, NIS={nis:.4f}")
            reward /= abs(DIVERGENCE_PENALTY)
    return reward, done

def run_step(imu, joint, ft, gt, serow_framework, state, step, agent = None, contact_state = None, next_contact_state = None, deterministic = False, baseline = False):
    contact_frames = state.get_contacts_frame()
    
    # Run the filter
    imu, kin, ft = serow_framework.process_measurements(imu, joint, ft, None)

    # Predict the base state
    serow_framework.base_estimator_predict_step(imu, kin)

    # Update the base state with the contact position
    rewards = {}
    done = {}
    actions = {}
    log_probs = {}
    values = {}
    x = {}
    next_x = {}

    for cf in contact_frames:
        actions[cf] = np.ones(agent.actor.action_dim) if agent is not None else np.ones(1)
        log_probs[cf] = None
        values[cf] = None
        x[cf] = None
        next_x[cf] = None
        rewards[cf] = None
        done[cf] = 0.0

    for cf in contact_frames:
        if agent is not None and not baseline:
            # Compute the action
            prior_state = serow_framework.get_state(allow_invalid=True)
            if contact_state.contacts_status[cf] and prior_state.get_contact_position(cf) is not None:
                R_base = quaternion_to_rotation_matrix(prior_state.get_base_orientation()).transpose()
                local_pos = R_base @ (prior_state.get_base_position() - prior_state.get_contact_position(cf))
                local_pos = np.array([abs(local_pos[0]), abs(local_pos[1]), local_pos[2]])
                x[cf] = np.concatenate((local_pos, np.array([contact_state.contacts_probability[cf]])), axis=0)

                if agent.name == "PPO":
                    actions[cf], log_probs[cf] = agent.actor.get_action(x[cf], deterministic=deterministic)
                    values[cf] = agent.critic(torch.FloatTensor(x[cf]).reshape(1, -1).to(next(agent.critic.parameters()).device)).item()
                else:
                    actions[cf] = agent.get_action(x[cf], deterministic=deterministic)

                if (deterministic):
                    print(f"Action for {cf}: {actions[cf]}")

        # Set the action
        serow_framework.set_action(cf, actions[cf])
    
        # Run the update step with the contact position
        serow_framework.base_estimator_update_with_contact_position(cf, kin)
        
        # Get the post state
        post_state = serow_framework.get_state(allow_invalid=True)

        # Compute the reward
        rewards[cf], done[cf] = compute_reward(cf, serow_framework, post_state, gt, step)

        if agent is not None:
            if (x[cf] is not None and post_state.get_contact_position(cf) is not None):
                R_base = quaternion_to_rotation_matrix(post_state.get_base_orientation()).transpose()
                local_pos = R_base @ (post_state.get_base_position() - post_state.get_contact_position(cf))
                local_pos = np.array([abs(local_pos[0]), abs(local_pos[1]), local_pos[2]])
                next_x[cf] = np.concatenate((local_pos, np.array([next_contact_state.contacts_probability[cf]])), axis=0)
                if agent.name == "PPO":
                    agent.add_to_buffer(x[cf], actions[cf], rewards[cf], next_x[cf], done[cf], values[cf], log_probs[cf])
                else:
                    agent.add_to_buffer(x[cf], actions[cf], rewards[cf], next_x[cf], done[cf])

        if done[cf] is not None and done[cf] == 1.0:
            print(f"Diverged for {cf}")
            break

    serow_framework.base_estimator_finish_update(imu, kin)
    state = serow_framework.get_state(allow_invalid=True)
    return imu.timestamp, state, rewards, done

def sync_and_align_data(base_timestamps, base_position, base_orientation, gt_timestamps, 
                        gt_position, gt_orientation, align = False):
    # Find the common time range
    start_time = max(base_timestamps[0], gt_timestamps[0])
    end_time = min(base_timestamps[-1], gt_timestamps[-1])

    # Create a common time grid with actual data length
    num_points = min(len(base_timestamps), len(gt_timestamps))
    common_timestamps = np.linspace(start_time, end_time, num_points)
    
    # Interpolate base position
    base_position_interp = np.zeros((len(common_timestamps), 3))
    for i in range(3):  # x, y, z
        base_position_interp[:, i] = np.interp(common_timestamps, base_timestamps, 
                                               base_position[:, i])
    
    # Interpolate ground truth position
    gt_position_interp = np.zeros((len(common_timestamps), 3))
    for i in range(3):  # x, y, z
        gt_position_interp[:, i] = np.interp(common_timestamps, gt_timestamps, gt_position[:, i])

    # Interpolate orientations
    base_orientation_interp = np.zeros((len(common_timestamps), 4))
    gt_orientation_interp = np.zeros((len(common_timestamps), 4))
    for i in range(4):  # w, x, y, z
        base_orientation_interp[:, i] = np.interp(common_timestamps, base_timestamps, 
                                                  base_orientation[:, i])
        gt_orientation_interp[:, i] = np.interp(common_timestamps, gt_timestamps, 
                                                gt_orientation[:, i])

    # Exclude entries that are before the first common timestamp
    base_position_interp = base_position_interp[common_timestamps >= start_time]
    base_orientation_interp = base_orientation_interp[common_timestamps >= start_time]
    gt_position_interp = gt_position_interp[common_timestamps >= start_time]
    gt_orientation_interp = gt_orientation_interp[common_timestamps >= start_time]

    if align:
        # Compute the initial rigid body transformation from the first timestamp
        R_gt = quaternion_to_rotation_matrix(gt_orientation_interp[0])
        R_base = quaternion_to_rotation_matrix(base_orientation_interp[0])
        R = R_gt.transpose() @ R_base
        t = base_position_interp[0] - R @ gt_position_interp[0]

        # Apply transformation to base position and orientation
        for i in range(len(common_timestamps)):
            gt_position_interp[i] = R @ gt_position_interp[i] + t
            gt_orientation_interp[i] = rotation_matrix_to_quaternion(R @ quaternion_to_rotation_matrix(gt_orientation_interp[i]))

        # Print transformation details
        print("Rotation matrix from gt to base:")
        print(R)
        print("\nTranslation vector from gt to base:")
        print(t)
    else:
        print("Not spatially aligning data")

    return common_timestamps, base_position_interp, base_orientation_interp, gt_position_interp, \
        gt_orientation_interp

def filter(imu_measurements, joint_measurements, force_torque_measurements, base_pose_ground_truth, 
           serow_framework, state, align = False):
    base_positions = []
    base_orientations = []
    base_timestamps = []
    contact_states = []
    cumulative_rewards = {}
    for cf in state.get_contacts_frame():
        cumulative_rewards[cf] = []

    for step, (imu, joint, ft, gt) in enumerate(zip(imu_measurements, joint_measurements, force_torque_measurements, 
                                  base_pose_ground_truth)):
        timestamp, state, rewards, _ = run_step(imu, joint, ft, gt, serow_framework, state, step,
                                                 deterministic=True)
        base_timestamps.append(timestamp)
        base_positions.append(state.get_base_position())
        contact_states.append(state.get_contact_state())
        base_orientations.append(state.get_base_orientation()) 
        for cf in rewards:
            if rewards[cf] is not None:
                if len(cumulative_rewards[cf]) == 0:
                    cumulative_rewards[cf].append(rewards[cf])
                else:
                    cumulative_rewards[cf].append(cumulative_rewards[cf][-1] + rewards[cf])
            else:
                if len(cumulative_rewards[cf]) == 0:
                    cumulative_rewards[cf].append(0.0)
                else:
                    cumulative_rewards[cf].append(cumulative_rewards[cf][-1])

    # Convert to numpy arrays
    base_position = np.array(base_positions)
    base_orientation = np.array(base_orientations)
    base_timestamps = np.array(base_timestamps)
    cumulative_rewards = {cf: np.array(cumulative_rewards[cf]) for cf in state.get_contacts_frame()}
    # Print evaluation metrics
    print("\nPolicy Evaluation Metrics:")
    for cf in state.get_contacts_frame():
        print(f"Average Cumulative Reward for {cf}: {np.mean(cumulative_rewards[cf]):.4f}")
        print(f"Max Cumulative Reward for {cf}: {np.max(cumulative_rewards[cf]):.4f}")
        print(f"Min Cumulative Reward for {cf}: {np.min(cumulative_rewards[cf]):.4f}")
        print("-------------------------------------------------")
    
    # Extract ground truth data with timestamps
    gt_timestamps = np.array([gt.timestamp for gt in base_pose_ground_truth])
    base_ground_truth_position = np.array([gt.position for gt in base_pose_ground_truth])
    base_ground_truth_orientation = np.array([gt.orientation for gt in base_pose_ground_truth])

    timestamps, base_position_aligned, base_orientation_aligned, gt_position_aligned, \
        gt_orientation_aligned = sync_and_align_data(base_timestamps, base_position,
                                                   base_orientation, gt_timestamps,
                                                   base_ground_truth_position,
                                                   base_ground_truth_orientation, align)
    
    return timestamps, base_position_aligned, base_orientation_aligned, gt_position_aligned, \
        gt_orientation_aligned, cumulative_rewards, contact_states

def plot_trajectories(timestamps, base_position, base_orientation, gt_position, gt_orientation, 
                      cumulative_rewards = None):
    
     # Plot the synchronized and aligned data
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, gt_position[:, 0], label="gt x")
    plt.plot(timestamps, base_position[:, 0], label="base x (aligned)")
    plt.plot(timestamps, gt_position[:, 1], label="gt y")
    plt.plot(timestamps, base_position[:, 1], label="base y (aligned)")
    plt.plot(timestamps, gt_position[:, 2], label="gt z")
    plt.plot(timestamps, base_position[:, 2], label="base z (aligned)")
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Base Position vs Ground Truth (Spatially Aligned)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(timestamps, gt_orientation[:, 0], label="gt w")
    plt.plot(timestamps, base_orientation[:, 0], label="base w")
    plt.plot(timestamps, gt_orientation[:, 1], label="gt x")
    plt.plot(timestamps, base_orientation[:, 1], label="base x")
    plt.plot(timestamps, gt_orientation[:, 2], label="gt y")
    plt.plot(timestamps, base_orientation[:, 2], label="base y")
    plt.plot(timestamps, gt_orientation[:, 3], label="gt z")
    plt.plot(timestamps, base_orientation[:, 3], label="base z")
    plt.xlabel('Time (s)')
    plt.ylabel('Quaternion Components')
    plt.title('Base Orientation vs Ground Truth')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot 3D trajectories
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt_position[:, 0], gt_position[:, 1], gt_position[:, 2], 
            label='Ground Truth', color='blue')
    ax.plot(base_position[:, 0], base_position[:, 1], base_position[:, 2], 
            label='Base Position', color='green')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectories')
    ax.legend()
    plt.show()
    
    if cumulative_rewards is not None:
        n_cf = len(cumulative_rewards)
        fig, axes = plt.subplots(n_cf, 1, figsize=(12, 4*n_cf))
        if n_cf == 1:
            axes = [axes]  # Make axes iterable for single subplot case
        
        for ax, cf in zip(axes, cumulative_rewards):
            ax.plot(cumulative_rewards[cf])
            ax.set_xlabel('steps')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title(f'Cumulative Reward for {cf} over steps')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

def plot_joint_states(joint_states):
    """Plot joint positions and velocities over time."""
    # Debug print to check joint states
    print(f"Number of joint states: {len(joint_states)}")
    if len(joint_states) == 0:
        print("Error: No joint states available")
        return
        
    # Get all unique joint names
    joint_names = list(joint_states[0].joints_position.keys())
    print(f"Joint names: {joint_names}")
    n_joints = len(joint_names)
    
    if n_joints == 0:
        print("Error: No joints found in the first joint state")
        return
        
    print(f"Number of joints: {n_joints}")
    
    # Create figure with subplots for positions
    fig_pos, axes_pos = plt.subplots(n_joints, 1, figsize=(12, 4*n_joints))
    fig_pos.suptitle('Joint Positions Over Time')
    
    # Create figure with subplots for velocities
    fig_vel, axes_vel = plt.subplots(n_joints, 1, figsize=(12, 4*n_joints))
    fig_vel.suptitle('Joint Velocities Over Time')
    
    # Create time array
    times = np.arange(len(joint_states))
    
    # Plot each joint's position and velocity
    for i, joint_name in enumerate(joint_names):
        # Extract position and velocity data for this joint
        positions = [state.joints_position[joint_name] for state in joint_states]
        velocities = [state.joints_velocity[joint_name] for state in joint_states]
        
        # Plot position
        axes_pos[i].plot(times, positions)
        axes_pos[i].set_ylabel('Position (rad)')
        axes_pos[i].set_title(f'{joint_name} Position')
        axes_pos[i].grid(True)
        
        # Plot velocity
        axes_vel[i].plot(times, velocities)
        axes_vel[i].set_ylabel('Velocity (rad/s)')
        axes_vel[i].set_title(f'{joint_name} Velocity')
        axes_vel[i].grid(True)
    
    # Add x-label to bottom subplot only
    axes_pos[-1].set_xlabel('Time Steps')
    axes_vel[-1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.show()

def plot_contact_states(contact_states):
    """Plot contact states over time.
    
    Args:
        contact_states: List of ContactState objects containing contact information
    """
    if not contact_states:
        print("No contact states to plot")
        return
        
    # Get all unique contact names from the first state
    contact_names = list(contact_states[0].contacts_status.keys())
    n_contacts = len(contact_names)
    
    if n_contacts == 0:
        print("No contacts found in the contact states")
        return
        
    # Create figure with subplots for status and probability
    fig, axes = plt.subplots(n_contacts, 2, figsize=(15, 4*n_contacts))
    fig.suptitle('Contact States Over Time')
    
    # Create time array
    times = np.arange(len(contact_states))
    
    # Plot each contact's status and probability
    for i, contact_name in enumerate(contact_names):
        # Extract status and probability data for this contact
        statuses = [state.contacts_status[contact_name] for state in contact_states]
        probabilities = [state.contacts_probability[contact_name] for state in contact_states]
        
        # Plot status
        ax_status = axes[i, 0] if n_contacts > 1 else axes[0]
        ax_status.plot(times, statuses, 'b-', label='Status')
        ax_status.set_ylabel('Contact Status')
        ax_status.set_title(f'{contact_name} Status')
        ax_status.set_ylim(-0.1, 1.1)  # Binary values
        ax_status.grid(True)
        
        # Plot probability
        ax_prob = axes[i, 1] if n_contacts > 1 else axes[1]
        ax_prob.plot(times, probabilities, 'r-', label='Probability')
        ax_prob.set_ylabel('Contact Probability')
        ax_prob.set_title(f'{contact_name} Probability')
        ax_prob.set_ylim(-0.1, 1.1)  # Probability range
        ax_prob.grid(True)
    
    # Add x-label to bottom subplots only
    if n_contacts > 1:
        axes[-1, 0].set_xlabel('Time Steps')
        axes[-1, 1].set_xlabel('Time Steps')
    else:
        axes[0].set_xlabel('Time Steps')
        axes[1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.show()

def plot_contact_forces_and_torques(contact_states):
    """Plot contact forces and torques over time.
    
    Args:
        contact_states: List of ContactState objects containing contact information
    """
    if not contact_states:
        print("No contact states to plot")
        return
        
    # Get all unique contact names from the first state
    contact_names = list(contact_states[0].contacts_force.keys())
    n_contacts = len(contact_names)
    
    if n_contacts == 0:
        print("No contacts found in the contact states")
        return
        
    # Create figure with subplots for forces and torques
    fig, axes = plt.subplots(n_contacts, 2, figsize=(15, 4*n_contacts))
    fig.suptitle('Contact Forces and Torques Over Time')
    
    # Create time array
    times = np.arange(len(contact_states))
    
    # Plot each contact's forces and torques
    for i, contact_name in enumerate(contact_names):
        # Extract force and torque data for this contact
        forces_x = [state.contacts_force[contact_name][0] for state in contact_states]
        forces_y = [state.contacts_force[contact_name][1] for state in contact_states]
        forces_z = [state.contacts_force[contact_name][2] for state in contact_states]
        
        # Plot forces
        ax_force = axes[i, 0] if n_contacts > 1 else axes[0]
        ax_force.plot(times, forces_x, 'r-', label='Fx')
        ax_force.plot(times, forces_y, 'g-', label='Fy')
        ax_force.plot(times, forces_z, 'b-', label='Fz')
        ax_force.set_ylabel('Force (N)')
        ax_force.set_title(f'{contact_name} Forces')
        ax_force.grid(True)
        ax_force.legend()
        
        # Plot torques if available
        ax_torque = axes[i, 1] if n_contacts > 1 else axes[1]
        if hasattr(contact_states[0], 'contacts_torque') and contact_states[0].contacts_torque:
            torques_x = [state.contacts_torque[contact_name][0] for state in contact_states]
            torques_y = [state.contacts_torque[contact_name][1] for state in contact_states]
            torques_z = [state.contacts_torque[contact_name][2] for state in contact_states]
            
            ax_torque.plot(times, torques_x, 'r-', label='Tx')
            ax_torque.plot(times, torques_y, 'g-', label='Ty')
            ax_torque.plot(times, torques_z, 'b-', label='Tz')
            ax_torque.set_ylabel('Torque (Nm)')
            ax_torque.set_title(f'{contact_name} Torques')
            ax_torque.grid(True)
            ax_torque.legend()
        else:
            ax_torque.text(0.5, 0.5, 'No torque data available', 
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=ax_torque.transAxes)
            ax_torque.set_title(f'{contact_name} Torques')
    
    # Add x-label to bottom subplots only
    if n_contacts > 1:
        axes[-1, 0].set_xlabel('Time Steps')
        axes[-1, 1].set_xlabel('Time Steps')
    else:
        axes[0].set_xlabel('Time Steps')
        axes[1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    robot = "go2"
    # Read the measurement mcap file
    imu_measurements  = read_imu_measurements("/tmp/serow_measurements.mcap")
    joint_measurements = read_joint_measurements("/tmp/serow_measurements.mcap")
    force_torque_measurements = read_force_torque_measurements("/tmp/serow_measurements.mcap")
    base_pose_ground_truth = read_base_pose_ground_truth("/tmp/serow_measurements.mcap")
    base_states = read_base_states("/tmp/serow_proprioception.mcap")
    contact_states = read_contact_states("/tmp/serow_proprioception.mcap")
    joint_states = read_joint_states("/tmp/serow_proprioception.mcap")
   
    # Initialize SEROW
    serow_framework = serow.Serow()
    serow_framework.initialize(f"{robot}_rl.json")
    state = serow_framework.get_state(allow_invalid=True)
    state.set_joint_state(joint_states[0])
    state.set_base_state(base_states[0])  
    state.set_contact_state(contact_states[0])
    serow_framework.set_state(state)

    # Run SEROW
    timestamps, base_position, base_orientation, gt_position, gt_orientation, cumulative_rewards, \
        _ = filter(imu_measurements, joint_measurements, force_torque_measurements, 
                   base_pose_ground_truth, serow_framework, state, align=True)
    
    # Plot the trajectories
    plot_trajectories(timestamps, base_position, base_orientation, gt_position, gt_orientation, \
                      cumulative_rewards)
    plot_joint_states(joint_states)
    plot_contact_states(contact_states)
    plot_contact_forces_and_torques(contact_states)

