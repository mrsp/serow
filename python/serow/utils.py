#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation, Slerp


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


def _sort_by_timestamp(timestamps, *arrays):
    """Sort series by timestamp so np.interp receives monotonic x."""
    ts = np.asarray(timestamps, dtype=float)
    order = np.argsort(ts, kind="stable")
    out_ts = ts[order]
    out_arrays = tuple(np.asarray(a)[order] for a in arrays)
    return (out_ts,) + out_arrays


def _interp_vector(common_timestamps, xp, values):
    """Interpolate (N, D) values onto common_timestamps; xp must be sorted increasing."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    out = np.zeros((len(common_timestamps), values.shape[1]))
    for j in range(values.shape[1]):
        out[:, j] = np.interp(common_timestamps, xp, values[:, j])
    return out


def _normalize_quaternion_rows(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return q / n


def _quaternion_wxyz_to_xyzw(q):
    """SciPy Rotation uses quaternions as [x, y, z, w]; we use [w, x, y, z]."""
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        return np.array([q[1], q[2], q[3], q[0]], dtype=float)
    return np.concatenate([q[:, 1:4], q[:, :1]], axis=1)


def _quaternion_xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        return np.array([q[3], q[0], q[1], q[2]], dtype=float)
    return np.concatenate([q[:, 3:4], q[:, :3]], axis=1)


def _interp_quaternion_slerp(common_timestamps, xp, quat_wxyz):
    """Interpolate unit quaternions [w,x,y,z] onto common_timestamps using SciPy Slerp."""
    xp = np.asarray(xp, dtype=float)
    q = np.asarray(quat_wxyz, dtype=float)
    if q.ndim == 1:
        q = q.reshape(1, 4)
    if len(xp) == 0:
        raise ValueError("empty orientation timestamp series")
    if len(xp) == 1:
        q0 = _normalize_quaternion_rows(q)[0]
        return np.tile(q0, (len(common_timestamps), 1))
    rotations = Rotation.from_quat(_quaternion_wxyz_to_xyzw(_normalize_quaternion_rows(q)))
    slerp = Slerp(xp, rotations)
    return _quaternion_xyzw_to_wxyz(slerp(common_timestamps).as_quat())


def _umeyama_rigid_se3(source_xyz, target_xyz):
    """
    Rigid special case of Umeyama (scale fixed to 1): least-squares SE(3)
    mapping source -> target as target ≈ R @ source + t.

    source_xyz, target_xyz: (n, 3) corresponding points in the same frame.
    """
    X = np.asarray(source_xyz, dtype=float).reshape(-1, 3)
    Y = np.asarray(target_xyz, dtype=float).reshape(-1, 3)
    n = X.shape[0]
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    H = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = mu_y - R @ mu_x
    return R, t


def sync_and_align_data(
    base_timestamps,
    base_position,
    base_orientation,
    gt_timestamps,
    gt_position,
    gt_orientation,
    align=False,
    n_points=5000,
    base_linear_velocity=None,
    base_angular_velocity=None,
    gt_linear_velocity=None,
    gt_angular_velocity=None,
    gt_velocity_timestamps=None,
):
    """
    Resample base and ground-truth pose (and optionally velocities) onto a common
    time grid over the overlap of all timestamp ranges. Series are sorted by
    time before interpolation.

    Returns a 5-tuple (pose only) or a 9-tuple when all velocity arrays and
    gt_velocity_timestamps (or implicit gt_timestamps) are provided.

    When align=True, GT linear and angular velocity are rotated by the same R
    as GT position/orientation (translation does not affect velocity).
    """
    sync_vel = (
        base_linear_velocity is not None
        and base_angular_velocity is not None
        and gt_linear_velocity is not None
        and gt_angular_velocity is not None
    )
    if sync_vel:
        base_ts, base_position, base_orientation, base_lv, base_av = _sort_by_timestamp(
            base_timestamps,
            base_position,
            base_orientation,
            base_linear_velocity,
            base_angular_velocity,
        )
        gt_vts = gt_velocity_timestamps
        if gt_vts is None:
            gt_vts = gt_timestamps
        gt_vts, gt_lv, gt_av = _sort_by_timestamp(
            gt_vts, gt_linear_velocity, gt_angular_velocity
        )
    else:
        base_ts, base_position, base_orientation = _sort_by_timestamp(
            base_timestamps, base_position, base_orientation
        )
        base_lv = base_av = gt_lv = gt_av = None
        gt_vts = None
    gt_ts, gt_position, gt_orientation = _sort_by_timestamp(
        gt_timestamps, gt_position, gt_orientation
    )

    start_time = max(base_ts[0], gt_ts[0])
    end_time = min(base_ts[-1], gt_ts[-1])
    if sync_vel:
        start_time = max(start_time, gt_vts[0])
        end_time = min(end_time, gt_vts[-1])

    lengths = [len(base_ts), len(gt_ts)]
    if sync_vel:
        lengths.append(len(gt_vts))
    num_points = max(1, min(lengths))
    common_timestamps = np.linspace(start_time, end_time, num_points)

    base_position_interp = _interp_vector(common_timestamps, base_ts, base_position)
    gt_position_interp = _interp_vector(common_timestamps, gt_ts, gt_position)

    base_orientation_interp = _interp_quaternion_slerp(
        common_timestamps, base_ts, base_orientation
    )
    gt_orientation_interp = _interp_quaternion_slerp(
        common_timestamps, gt_ts, gt_orientation
    )

    if sync_vel:
        base_linear_vel_interp = _interp_vector(common_timestamps, base_ts, base_lv)
        base_angular_vel_interp = _interp_vector(common_timestamps, base_ts, base_av)
        gt_linear_vel_interp = _interp_vector(common_timestamps, gt_vts, gt_lv)
        gt_angular_vel_interp = _interp_vector(common_timestamps, gt_vts, gt_av)

    n = len(common_timestamps)
    if align:
        if n >= 2:
            n_min = min(n_points, len(gt_timestamps))
            print(f"Using {n_min} points for alignment")
            R, t = _umeyama_rigid_se3(gt_position_interp[:n_min], base_position_interp[:n_min])
        else:
            R_gt = quaternion_to_rotation_matrix(gt_orientation_interp[0])
            R_base = quaternion_to_rotation_matrix(base_orientation_interp[0])
            R = R_gt.T @ R_base
            t = base_position_interp[0] - R @ gt_position_interp[0]

        for i in range(n):
            gt_position_interp[i] = R @ gt_position_interp[i] + t
            gt_orientation_interp[i] = rotation_matrix_to_quaternion(
                R @ quaternion_to_rotation_matrix(gt_orientation_interp[i])
            )
        if sync_vel:
            gt_linear_vel_interp = (R @ gt_linear_vel_interp.T).T
            gt_angular_vel_interp = (R @ gt_angular_vel_interp.T).T

        print("Rotation matrix from gt to base:")
        print(R)
        print("\nTranslation vector from gt to base:")
        print(t)
    else:
        print("Not spatially aligning data")

    if sync_vel:
        return (
            common_timestamps,
            base_position_interp,
            base_orientation_interp,
            gt_position_interp,
            gt_orientation_interp,
            base_linear_vel_interp,
            base_angular_vel_interp,
            gt_linear_vel_interp,
            gt_angular_vel_interp,
        )
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

class BaseVelocityGroundTruth:
    def __init__(self, timestamp, linear_velocity, angular_velocity):
        self.timestamp = timestamp
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
