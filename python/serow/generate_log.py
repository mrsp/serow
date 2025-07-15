import numpy as np

from read_mcap import (
    read_base_states,
    read_contact_states,
    read_force_torque_measurements,
    read_joint_measurements,
    read_imu_measurements,
    read_base_pose_ground_truth,
    read_joint_states,
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

    dataset = {
        "imu": imu_measurements,
        "joints": joint_measurements,
        "ft": force_torque_measurements,
        "base_states": base_states,
        "contact_states": contact_states,
        "joint_states": joint_states,
        "base_pose_ground_truth": base_pose_ground_truth,
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
    )


if __name__ == "__main__":
    robot = "go2"
    mcap_path = "/tmp"
    generate_log(robot, mcap_path)
