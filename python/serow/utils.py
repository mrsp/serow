#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from mpl_toolkits.mplot3d import Axes3D


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
    R = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )

    return R


def logMap(R):
    """
    Compute the logarithm map of a rotation matrix.

    Parameters:
    R : numpy array with shape (3, 3)
        The rotation matrix in SO(3) Lie group.

    Returns:
    numpy array with shape (3,)
        The corresponding so(3) Lie algebra element.
    """
    R11 = R[0, 0]
    R12 = R[0, 1]
    R13 = R[0, 2]
    R21 = R[1, 0]
    R22 = R[1, 1]
    R23 = R[1, 2]
    R31 = R[2, 0]
    R32 = R[2, 1]
    R33 = R[2, 2]

    trace = R.trace()

    omega = np.zeros(3)

    # Special case when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    if trace + 1.0 < 1e-3:
        if R33 > R22 and R33 > R11:
            # R33 is the largest diagonal, a=3, b=1, c=2
            W = R21 - R12
            Q1 = 2.0 + 2.0 * R33
            Q2 = R31 + R13
            Q3 = R23 + R32
            r = np.sqrt(Q1)
            one_over_r = 1 / r
            norm = np.sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W)
            sgn_w = -1.0 if W < 0 else 1.0
            mag = np.pi - (2 * sgn_w * W) / norm
            scale = 0.5 * one_over_r * mag
            omega = sgn_w * scale * np.array([Q2, Q3, Q1])
        elif R22 > R11:
            # R22 is the largest diagonal, a=2, b=3, c=1
            W = R13 - R31
            Q1 = 2.0 + 2.0 * R22
            Q2 = R23 + R32
            Q3 = R12 + R21
            r = np.sqrt(Q1)
            one_over_r = 1 / r
            norm = np.sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W)
            sgn_w = -1.0 if W < 0 else 1.0
            mag = np.pi - (2 * sgn_w * W) / norm
            scale = 0.5 * one_over_r * mag
            omega = sgn_w * scale * np.array([Q3, Q1, Q2])
        else:
            # R11 is the largest diagonal, a=1, b=2, c=3
            W = R32 - R23
            Q1 = 2.0 + 2.0 * R11
            Q2 = R12 + R21
            Q3 = R31 + R13
            r = np.sqrt(Q1)
            one_over_r = 1 / r
            norm = np.sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W)
            sgn_w = -1.0 if W < 0 else 1.0
            mag = np.pi - (2 * sgn_w * W) / norm
            scale = 0.5 * one_over_r * mag
            omega = sgn_w * scale * np.array([Q1, Q2, Q3])
    else:
        magnitude = 0.0
        tr_3 = trace - 3.0  # could be non-negative if the matrix is off orthogonal
        if tr_3 < -1e-6:
            # this is the normal case -1 < trace < 3
            theta = np.arccos((trace - 1.0) / 2.0)
            magnitude = theta / (2.0 * np.sin(theta))
        else:
            # when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            # use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            # see https://github.com/borglab/gtsam/issues/746 for details
            magnitude = 0.5 - tr_3 / 12.0 + tr_3 * tr_3 / 60.0

        omega = magnitude * np.array([R32 - R23, R13 - R31, R21 - R12])
    return omega


def sync_and_align_data(
    base_timestamps,
    base_position,
    base_orientation,
    gt_timestamps,
    gt_position,
    gt_orientation,
    align=False,
):
    # Find the common time range
    start_time = max(base_timestamps[0], gt_timestamps[0])
    end_time = min(base_timestamps[-1], gt_timestamps[-1])

    # Create a common time grid with actual data length
    num_points = min(len(base_timestamps), len(gt_timestamps))
    common_timestamps = np.linspace(start_time, end_time, num_points)

    # Interpolate base position
    base_position_interp = np.zeros((len(common_timestamps), 3))
    for i in range(3):  # x, y, z
        base_position_interp[:, i] = np.interp(
            common_timestamps, base_timestamps, base_position[:, i]
        )

    # Interpolate ground truth position
    gt_position_interp = np.zeros((len(common_timestamps), 3))
    for i in range(3):  # x, y, z
        gt_position_interp[:, i] = np.interp(
            common_timestamps, gt_timestamps, gt_position[:, i]
        )

    # Interpolate orientations
    base_orientation_interp = np.zeros((len(common_timestamps), 4))
    gt_orientation_interp = np.zeros((len(common_timestamps), 4))
    for i in range(4):  # w, x, y, z
        base_orientation_interp[:, i] = np.interp(
            common_timestamps, base_timestamps, base_orientation[:, i]
        )
        gt_orientation_interp[:, i] = np.interp(
            common_timestamps, gt_timestamps, gt_orientation[:, i]
        )

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
            gt_orientation_interp[i] = rotation_matrix_to_quaternion(
                R @ quaternion_to_rotation_matrix(gt_orientation_interp[i])
            )

        # Print transformation details
        print("Rotation matrix from gt to base:")
        print(R)
        print("\nTranslation vector from gt to base:")
        print(t)
    else:
        print("Not spatially aligning data")

    return (
        common_timestamps,
        base_position_interp,
        base_orientation_interp,
        gt_position_interp,
        gt_orientation_interp,
    )


def plot_trajectories(
    timestamps, base_position, base_orientation, gt_position, gt_orientation
):

    # Plot the synchronized and aligned data
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, gt_position[:, 0], label="gt x")
    plt.plot(timestamps, base_position[:, 0], label="base x (aligned)")
    plt.plot(timestamps, gt_position[:, 1], label="gt y")
    plt.plot(timestamps, base_position[:, 1], label="base y (aligned)")
    plt.plot(timestamps, gt_position[:, 2], label="gt z")
    plt.plot(timestamps, base_position[:, 2], label="base z (aligned)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Base Position vs Ground Truth (Spatially Aligned)")
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
    plt.xlabel("Time (s)")
    plt.ylabel("Quaternion Components")
    plt.title("Base Orientation vs Ground Truth")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot 3D trajectories
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        gt_position[:, 0],
        gt_position[:, 1],
        gt_position[:, 2],
        label="Ground Truth",
        color="blue",
    )
    ax.plot(
        base_position[:, 0],
        base_position[:, 1],
        base_position[:, 2],
        label="Base Position",
        color="green",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectories")
    ax.legend()
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
    fig_pos, axes_pos = plt.subplots(n_joints, 1, figsize=(12, 4 * n_joints))
    fig_pos.suptitle("Joint Positions Over Time")

    # Create figure with subplots for velocities
    fig_vel, axes_vel = plt.subplots(n_joints, 1, figsize=(12, 4 * n_joints))
    fig_vel.suptitle("Joint Velocities Over Time")

    # Create time array
    times = np.arange(len(joint_states))

    # Plot each joint's position and velocity
    for i, joint_name in enumerate(joint_names):
        # Extract position and velocity data for this joint
        positions = [state.joints_position[joint_name] for state in joint_states]
        velocities = [state.joints_velocity[joint_name] for state in joint_states]

        # Plot position
        axes_pos[i].plot(times, positions)
        axes_pos[i].set_ylabel("Position (rad)")
        axes_pos[i].set_title(f"{joint_name} Position")
        axes_pos[i].grid(True)

        # Plot velocity
        axes_vel[i].plot(times, velocities)
        axes_vel[i].set_ylabel("Velocity (rad/s)")
        axes_vel[i].set_title(f"{joint_name} Velocity")
        axes_vel[i].grid(True)

    # Add x-label to bottom subplot only
    axes_pos[-1].set_xlabel("Time Steps")
    axes_vel[-1].set_xlabel("Time Steps")

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
    fig, axes = plt.subplots(n_contacts, 2, figsize=(15, 4 * n_contacts))
    fig.suptitle("Contact States Over Time")

    # Create time array
    times = np.arange(len(contact_states))

    # Plot each contact's status and probability
    for i, contact_name in enumerate(contact_names):
        # Extract status and probability data for this contact
        statuses = [state.contacts_status[contact_name] for state in contact_states]
        probabilities = [
            state.contacts_probability[contact_name] for state in contact_states
        ]

        # Plot status
        ax_status = axes[i, 0] if n_contacts > 1 else axes[0]
        ax_status.plot(times, statuses, "b-", label="Status")
        ax_status.set_ylabel("Contact Status")
        ax_status.set_title(f"{contact_name} Status")
        ax_status.set_ylim(-0.1, 1.1)  # Binary values
        ax_status.grid(True)

        # Plot probability
        ax_prob = axes[i, 1] if n_contacts > 1 else axes[1]
        ax_prob.plot(times, probabilities, "r-", label="Probability")
        ax_prob.set_ylabel("Contact Probability")
        ax_prob.set_title(f"{contact_name} Probability")
        ax_prob.set_ylim(-0.1, 1.1)  # Probability range
        ax_prob.grid(True)

    # Add x-label to bottom subplots only
    if n_contacts > 1:
        axes[-1, 0].set_xlabel("Time Steps")
        axes[-1, 1].set_xlabel("Time Steps")
    else:
        axes[0].set_xlabel("Time Steps")
        axes[1].set_xlabel("Time Steps")

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
    fig, axes = plt.subplots(n_contacts, 2, figsize=(15, 4 * n_contacts))
    fig.suptitle("Contact Forces and Torques Over Time")

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
        ax_force.plot(times, forces_x, "r-", label="Fx")
        ax_force.plot(times, forces_y, "g-", label="Fy")
        ax_force.plot(times, forces_z, "b-", label="Fz")
        ax_force.set_ylabel("Force (N)")
        ax_force.set_title(f"{contact_name} Forces")
        ax_force.grid(True)
        ax_force.legend()

        # Plot torques if available
        ax_torque = axes[i, 1] if n_contacts > 1 else axes[1]
        if (
            hasattr(contact_states[0], "contacts_torque")
            and contact_states[0].contacts_torque
        ):
            torques_x = [
                state.contacts_torque[contact_name][0] for state in contact_states
            ]
            torques_y = [
                state.contacts_torque[contact_name][1] for state in contact_states
            ]
            torques_z = [
                state.contacts_torque[contact_name][2] for state in contact_states
            ]

            ax_torque.plot(times, torques_x, "r-", label="Tx")
            ax_torque.plot(times, torques_y, "g-", label="Ty")
            ax_torque.plot(times, torques_z, "b-", label="Tz")
            ax_torque.set_ylabel("Torque (Nm)")
            ax_torque.set_title(f"{contact_name} Torques")
            ax_torque.grid(True)
            ax_torque.legend()
        else:
            ax_torque.text(
                0.5,
                0.5,
                "No torque data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax_torque.transAxes,
            )
            ax_torque.set_title(f"{contact_name} Torques")

    # Add x-label to bottom subplots only
    if n_contacts > 1:
        axes[-1, 0].set_xlabel("Time Steps")
        axes[-1, 1].set_xlabel("Time Steps")
    else:
        axes[0].set_xlabel("Time Steps")
        axes[1].set_xlabel("Time Steps")

    plt.tight_layout()
    plt.show()


def export_models_to_onnx(agent, robot, params, path):
    """Export the trained models to ONNX format"""
    os.makedirs(path, exist_ok=True)

    # Set models to evaluation mode to disable dropout
    agent.actor.eval()
    agent.critic.eval()

    # Export actor model
    dummy_state = torch.randn(1, params["state_dim"]).to(agent.device)
    torch.onnx.export(
        agent.actor,
        dummy_state,
        f"{path}/{robot}_ppo_actor.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Export critic model
    torch.onnx.export(
        agent.critic,
        dummy_state,
        f"{path}/{robot}_ppo_critic.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["output"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Set models back to training mode
    agent.actor.train()
    agent.critic.train()
