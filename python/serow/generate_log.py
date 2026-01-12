import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from read_mcap import (
    read_base_states,
    read_contact_states,
    read_force_torque_measurements,
    read_joint_measurements,
    read_imu_measurements,
    read_base_pose_ground_truth,
    read_joint_states,
)
from utils import (
    BaseVelocityGroundTruth,
    logMap,
    quaternion_to_rotation_matrix,
)


def generate_log(robot, mcap_path):
    # Load and preprocess the data
    imu_measurements = read_imu_measurements(mcap_path + "/serow_measurements.mcap")
    joint_measurements = read_joint_measurements(mcap_path + "/serow_measurements.mcap")
    force_torque_measurements = read_force_torque_measurements(
        mcap_path + "/serow_measurements.mcap"
    )
    base_pose_ground_truth = read_base_pose_ground_truth(
        mcap_path + "/serow_measurements.mcap"
    )
    base_states = read_base_states(mcap_path + "/serow_proprioception.mcap")
    contact_states = read_contact_states(mcap_path + "/serow_proprioception.mcap")
    joint_states = read_joint_states(mcap_path + "/serow_proprioception.mcap")

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
    window_size = 31
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

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot smooth linear velocity
    ax1.plot(gt_timestamps, smooth_gt_linear_velocity)
    ax1.set_title("Smooth Linear Velocity")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.legend(["X", "Y", "Z"])
    ax1.grid(True)

    # Plot smooth angular velocity
    ax2.plot(gt_timestamps, smooth_gt_angular_velocity)
    ax2.set_title("Smooth Angular Velocity")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular Velocity (rad/s)")
    ax2.legend(["X", "Y", "Z"])
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    # Create BaseVelocityGroundTruth objects with smoothed data
    for timestamp, v, w in zip(
        gt_timestamps, smooth_gt_linear_velocity, smooth_gt_angular_velocity
    ):
        gt_vel = BaseVelocityGroundTruth(timestamp, v, w)
        base_velocity_ground_truth.append(gt_vel)

    dataset = {
        "imu": imu_measurements,
        "joints": joint_measurements,
        "ft": force_torque_measurements,
        "base_states": base_states,
        "contact_states": contact_states,
        "joint_states": joint_states,
        "base_pose_ground_truth": base_pose_ground_truth,
        "base_velocity_ground_truth": base_velocity_ground_truth,
    }

    # Save dataset to a numpy file
    dataset_path = robot + "_log.npz"
    np.savez(
        dataset_path,
        imu=dataset["imu"],
        joints=dataset["joints"],
        ft=dataset["ft"],
        base_states=dataset["base_states"],
        contact_states=dataset["contact_states"],
        joint_states=dataset["joint_states"],
        base_pose_ground_truth=dataset["base_pose_ground_truth"],
        base_velocity_ground_truth=dataset["base_velocity_ground_truth"],
    )


if __name__ == "__main__":
    robot = "go2"
    mcap_path = "/tmp"
    generate_log(robot, mcap_path)
