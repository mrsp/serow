import os
import json
import numpy as np
import serow
import matplotlib.pyplot as plt

from scipy import signal
from scipy.spatial.transform import Rotation
from read_mcap import (
    read_base_pose_ground_truth,
    read_base_states,
)

from utils import (
    BaseVelocityGroundTruth,
    logMap,
    quaternion_to_rotation_matrix,
    sync_and_align_data,
)

def generate_log(mcap_path):
    # Load Ground Truth
    base_pose_ground_truth = read_base_pose_ground_truth(
        mcap_path + "/serow_measurements.mcap"
    )
    # Load Estimates
    base_states = read_base_states(mcap_path + "/serow_proprioception.mcap")

    # Numerically compute the base velocity
    base_velocity_ground_truth = []
    gt_linear_velocity = []
    gt_angular_velocity = []
    gt_timestamps = []
    gt_prev = None
    for i, gt in enumerate(base_pose_ground_truth):
        if i == 0:
            w = np.zeros(3)
            v = np.zeros(3)
            gt_prev = gt
        else:
            dt = gt.timestamp - gt_prev.timestamp
            R = quaternion_to_rotation_matrix(gt.orientation)
            R_prev = quaternion_to_rotation_matrix(gt_prev.orientation)
            w = R @ logMap(R_prev.transpose() @ R) / dt
            v = (gt.position - gt_prev.position) / dt
            gt_prev = gt

        gt_timestamps.append(gt.timestamp)
        gt_linear_velocity.append(v)
        gt_angular_velocity.append(w)

    gt_linear_velocity = np.array(gt_linear_velocity)
    gt_angular_velocity = np.array(gt_angular_velocity)
    smooth_gt_linear_velocity = np.zeros_like(gt_linear_velocity)
    smooth_gt_angular_velocity = np.zeros_like(gt_angular_velocity)
    window_size = 101
    polyorder = 3
    for j in range(3):
        smooth_gt_linear_velocity[:, j] = signal.savgol_filter(
            gt_linear_velocity[:, j], window_size, polyorder, mode="nearest"
        )
        smooth_gt_angular_velocity[:, j] = signal.savgol_filter(
            gt_angular_velocity[:, j], window_size, polyorder, mode="nearest"
        )
    smooth_gt_linear_velocity = list(smooth_gt_linear_velocity)
    smooth_gt_angular_velocity = list(smooth_gt_angular_velocity)
   
    # Create BaseVelocityGroundTruth objects with smoothed data
    for timestamp, v, w in zip(
        gt_timestamps, smooth_gt_linear_velocity, smooth_gt_angular_velocity
    ):
        gt_vel = BaseVelocityGroundTruth(timestamp, v, w)
        base_velocity_ground_truth.append(gt_vel)

    dataset = {
        "base_states": base_states,
        "base_pose_ground_truth": base_pose_ground_truth,
        "base_velocity_ground_truth": base_velocity_ground_truth,
    }

    return dataset


log_path = "/tmp"
log = generate_log(log_path)

base_positions = np.array([base.base_position for base in log["base_states"]])
base_orientations = np.array([base.base_orientation for base in log["base_states"]])
base_linear_velocities = np.array([base.base_linear_velocity for base in log["base_states"]])
base_angular_velocities = np.array([base.base_angular_velocity for base in log["base_states"]])
timestamps = np.array([base.timestamp for base in log["base_states"]])

# Sync the data to the ground truth
gt_positions = np.array([gt.position for gt in log["base_pose_ground_truth"]])
gt_orientations = np.array([gt.orientation for gt in log["base_pose_ground_truth"]])
gt_position_timestamps = np.array([gt.timestamp for gt in log["base_pose_ground_truth"]])

gt_linear_velocities = np.array([gt.linear_velocity for gt in log["base_velocity_ground_truth"]])
gt_angular_velocities = np.array([gt.angular_velocity for gt in log["base_velocity_ground_truth"]])
gt_velocity_timestamps = np.array([gt.timestamp for gt in log["base_velocity_ground_truth"]])

(
    timestamps,
    base_positions,
    base_orientations,
    gt_positions,
    gt_orientations,
    base_linear_velocities,
    base_angular_velocities,
    gt_linear_velocities,
    gt_angular_velocities,
) = sync_and_align_data(
    timestamps,
    base_positions,
    base_orientations,
    gt_position_timestamps,
    gt_positions,
    gt_orientations,
    align=True, 
    n_points=5000,
    base_linear_velocity=base_linear_velocities,
    base_angular_velocity=base_angular_velocities,
    gt_linear_velocity=gt_linear_velocities,
    gt_angular_velocity=gt_angular_velocities,
    gt_velocity_timestamps=gt_velocity_timestamps,
)

assert len(timestamps) == len(gt_positions)
assert len(timestamps) == len(gt_orientations)
assert len(timestamps) == len(base_positions)
assert len(timestamps) == len(base_orientations)
assert len(timestamps) == len(base_linear_velocities)
assert len(timestamps) == len(base_angular_velocities)
assert len(timestamps) == len(gt_linear_velocities)
assert len(timestamps) == len(gt_angular_velocities)

# Plot the base and ground truth trajectories 
# Subtract each series' first sample so both curves start at 0 m (visual alignment only)
base_positions = base_positions - base_positions[0]
gt_positions = gt_positions - gt_positions[0]

axis_sub = ("x", "y", "z")
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axis_names = ("X", "Y", "Z")
for ax, i, name in zip(axes, range(3), axis_names):
    sub = axis_sub[i]
    ax.plot(timestamps, base_positions[:, i], label=rf"Est ${name}$")
    ax.plot(timestamps, gt_positions[:, i], label=rf"GT ${name}$", linestyle="--", color="black")
    ax.set_ylabel(rf"$p_{{\mathrm{{base}},{sub}}}$ (m)")
    ax.legend()
    ax.grid(True)
axes[-1].set_xlabel(r"$\mathrm{Time}$ (s)")
fig.suptitle(r"$\mathbf{p}_{\mathrm{base}}$ vs. $\mathbf{p}_{\mathrm{GT}}$ (position, m)")
plt.tight_layout()

# Plot the base and ground truth orientations as Euler angles 
def quat_wxyz_to_euler(quat_wxyz, seq="xyz", degrees=True):
    quat_xyzw = np.roll(np.asarray(quat_wxyz, dtype=float), -1, axis=-1)
    return Rotation.from_quat(quat_xyzw).as_euler(seq, degrees=degrees)

base_orientations_euler = quat_wxyz_to_euler(base_orientations)
gt_orientations_euler = quat_wxyz_to_euler(gt_orientations)

# Subtract each series' first sample so both curves start at 0 deg (visual alignment only)
base_orientations_euler = base_orientations_euler - base_orientations_euler[0]
gt_orientations_euler = gt_orientations_euler - gt_orientations_euler[0]



fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
euler_labels = (
    (r"$\phi$", "roll"),
    (r"$\theta$", "pitch"),
    (r"$\psi$", "yaw"),
)
for ax, i, (sym, name) in zip(axes, range(3), euler_labels):
    ax.plot(timestamps, base_orientations_euler[:, i], label=rf"Est ({name})")
    ax.plot(
        timestamps,
        gt_orientations_euler[:, i],
        label=rf"GT ({name})",
        linestyle="--",
        color="black",
    )
    ax.set_ylabel(rf"{sym} (deg)")
    ax.legend()
    ax.grid(True)
axes[-1].set_xlabel(r"$\mathrm{Time}$ (s)")
fig.suptitle(
    r"$\mathbf{R}_{\mathrm{base}}$ vs. $\mathbf{R}_{\mathrm{GT}}$ "
    r"(Euler $\phi,\theta,\psi$, deg)"
)
plt.tight_layout()

# Plot the base and ground truth linear velocities
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axis_names = ("X", "Y", "Z")
for ax, i, name in zip(axes, range(3), axis_names):
    sub = axis_sub[i]
    ax.plot(timestamps, base_linear_velocities[:, i], label=rf"Est ${name}$")
    ax.plot(
        timestamps,
        gt_linear_velocities[:, i],
        label=rf"GT ${name}$",
        linestyle="--",
        color="black",
    )
    ax.set_ylabel(rf"$v_{{\mathrm{{base}},{sub}}}$ (m/s)")
    ax.legend()
    ax.grid(True)
axes[-1].set_xlabel(r"$\mathrm{Time}$ (s)")
fig.suptitle(r"$\mathbf{v}_{\mathrm{base}}$ vs. $\mathbf{v}_{\mathrm{GT}}$ (linear, m/s)")
plt.tight_layout()

# Plot the base and ground truth angular velocities
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axis_names = ("X", "Y", "Z")
for ax, i, name in zip(axes, range(3), axis_names):
    sub = axis_sub[i]
    ax.plot(
        timestamps,
        base_angular_velocities[:, i] * 180 / np.pi,
        label=rf"Est ${name}$",
    )
    ax.plot(
        timestamps,
        gt_angular_velocities[:, i] * 180 / np.pi,
        label=rf"GT ${name}$",
        linestyle="--",
        color="black",
    )
    ax.set_ylabel(rf"$\omega_{{\mathrm{{base}},{sub}}}$ (deg/s)")
    ax.legend()
    ax.grid(True)
axes[-1].set_xlabel(r"$\mathrm{Time}$ (s)")
fig.suptitle(
    r"$\mathbf{\omega}_{\mathrm{base}}$ vs. $\mathbf{\omega}_{\mathrm{GT}}$ "
    r"(angular, deg/s)"
)
plt.tight_layout()

# Compute the Absolute Trajectory Error (ATE) for position, orientation, 
# linear velocity, and angular velocity
def error(gt, est):
    return np.sqrt(np.mean((gt - est) ** 2))

ate = error(gt_positions, base_positions)
ate_rot = error(gt_orientations_euler, base_orientations_euler)
print(f"Absolute Translation Error: {ate} m")
print(f"Absolute Rotation Error: {ate_rot} deg")

ave = error(gt_linear_velocities, base_linear_velocities)
ave_rot = error(gt_angular_velocities, base_angular_velocities)
print(f"Absolute Linear Velocity Error: {ave} m/s")
print(f"Absolute Angular Velocity Error: {ave_rot} deg/s")


# Fetch the imu biases from the base states
imu_linear_acceleration_biases = np.array([base.imu_linear_acceleration_bias for base in log["base_states"]])
imu_angular_velocity_biases = np.array([base.imu_angular_velocity_bias for base in log["base_states"]])
imu_timestamps = np.array([base.timestamp for base in log["base_states"]])

# Plot the imu biases — accelerometer (x, y, z)
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for ax, i in zip(axes, range(3)):
    sub = axis_sub[i]
    ax.plot(imu_timestamps, imu_linear_acceleration_biases[:, i])
    ax.set_ylabel(rf"$b_{{a,{sub}}}$ (m/s$^2$)")
    ax.grid(True)
axes[-1].set_xlabel(r"$\mathrm{Time}$ (s)")
fig.suptitle(r"IMU accelerometer bias")
plt.tight_layout()

# Plot the imu biases — gyroscope (x, y, z)
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for ax, i in zip(axes, range(3)):
    sub = axis_sub[i]
    ax.plot(
        imu_timestamps,
        imu_angular_velocity_biases[:, i] * 180 / np.pi,
    )
    ax.set_ylabel(rf"$b_{{\omega,{sub}}}$ (deg/s)")
    ax.grid(True)
axes[-1].set_xlabel(r"$\mathrm{Time}$ (s)")
fig.suptitle(r"IMU gyroscope bias")
plt.tight_layout()
plt.show()
