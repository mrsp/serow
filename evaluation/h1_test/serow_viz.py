# Visualizing data for SEROW

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import json

display_plots = True

config_file = "test_config.json"  # Path to your JSON config file
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration file {config_file} not found.")

with open(config_file, "r") as f:
    config = json.load(f)

serow_path = os.environ.get("SEROW_PATH")
if not serow_path:
    raise EnvironmentError("SEROW_PATH environment variable not set.")

# Resolve paths from the JSON configuration
base_path = config["Paths"]["base_path"]
experiment_type = config["Experiment"]["type"]
measurement_file = serow_path + config["Paths"]["data_file"].replace(
    "{base_path}", base_path
).replace("{type}", experiment_type)
prediction_file = serow_path + config["Paths"]["prediction_file"].replace(
    "{base_path}", base_path
).replace("{type}", experiment_type)


# Load the data from the HDF5 file
def load_gt_data(h5_file):
    with h5py.File(h5_file, "r") as f:
        positions = np.array(f["/h1_ground_truth_odometry/_pose/_pose/_position"])
        orientations = np.array(f["/h1_ground_truth_odometry/_pose/_pose/_orientation"])
        L_forces = np.array(f["/h1_left_ankle_force_torque_states/_wrench/_force"])
        R_forces = np.array(f["/h1_right_ankle_force_torque_states/_wrench/_force"])

        imu = np.array(f["/h1_imu/_linear_acceleration"])

    return (
        positions,
        orientations,
        imu,
        L_forces,
        R_forces
    )


def print_meta(h5_file):
    with h5py.File(h5_file, "r") as f:
        for name in f.keys():
            print(name)


def load_serow_preds(h5_file):
    with h5py.File(h5_file, "r") as f:
        com_pos_x = np.array(f["CoM_state/position/x"])
        com_pos_y = np.array(f["CoM_state/position/y"])
        com_pos_z = np.array(f["CoM_state/position/z"])

        com_vel_x = np.array(f["CoM_state/velocity/x"])
        com_vel_y = np.array(f["CoM_state/velocity/y"])
        com_vel_z = np.array(f["CoM_state/velocity/z"])

        pos_x = np.array(f["base_pose/position/x"])
        pos_y = np.array(f["base_pose/position/y"])
        pos_z = np.array(f["base_pose/position/z"])

        rot_x = np.array(f["base_pose/rotation/x"])
        rot_y = np.array(f["base_pose/rotation/y"])
        rot_z = np.array(f["base_pose/rotation/z"])
        rot_w = np.array(f["base_pose/rotation/w"])

        b_ax = np.array(f["imu_bias/accel/x"])
        b_ay = np.array(f["imu_bias/accel/y"])
        b_az = np.array(f["imu_bias/accel/z"])

        b_wx = np.array(f["imu_bias/angVel/x"])
        b_wy = np.array(f["imu_bias/angVel/y"])
        b_wz = np.array(f["imu_bias/angVel/z"])

        timestamps = np.array(f["timestamp/t"])
    return (
        timestamps,
        pos_x,
        pos_y,
        pos_z,
        com_pos_x,
        com_pos_y,
        com_pos_z,
        com_vel_x,
        com_vel_y,
        com_vel_z,
        rot_x,
        rot_y,
        rot_z,
        rot_w,
        b_ax,
        b_ay,
        b_az,
        b_wx,
        b_wy,
        b_wz,
    )


def compute_ATE_pos(gt_pos, est_x, est_y, est_z):
    est_pos = np.column_stack((est_x, est_y, est_z))
    error = np.linalg.norm(gt_pos - est_pos, axis=1)
    ate = np.sqrt(np.mean(error**2))
    return ate


def compute_ATE_rot(gt_rot, est_rot_w, est_rot_x, est_rot_y, est_rot_z):
    est_rot = np.column_stack((est_rot_w, est_rot_x, est_rot_y, est_rot_z))
    rotation_errors = np.zeros((gt_rot.shape[0]))
    for i in range(gt_rot.shape[0]):
        q_gt = gt_rot[i]
        q_est = est_rot[i]

        q_gt_conj = np.array(
            [q_gt[0], -q_gt[1], -q_gt[2], -q_gt[3]]
        )  # Conjugate of q_gt
        q_rel = quaternion_multiply(q_gt_conj, q_est)
        rotation_errors[i] = 2 * np.arccos(np.clip(q_rel[0], -1.0, 1.0))

    ate_rot = np.sqrt(np.mean(rotation_errors**2))
    return ate_rot


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions q1 and q2.

    Parameters:
    - q1: numpy array of shape (4,), quaternion (w, x, y, z).
    - q2: numpy array of shape (4,), quaternion (w, x, y, z).

    Returns:
    - q: numpy array of shape (4,), resulting quaternion.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def compute_com_vel(com_pos, timestamps):
    com_vel = np.zeros_like(com_pos)
    # Central difference for most points
    for i in range(1, len(com_pos) - 1):
        dt = timestamps[i + 1] - timestamps[i - 1]
        com_vel[i] = (com_pos[i + 1] - com_pos[i - 1]) / dt  # Using Central Difference
    # Forward difference for first point
    com_vel[0] = (com_pos[1] - com_pos[0]) / (timestamps[1] - timestamps[0])
    # Backward difference for last point
    com_vel[-1] = (com_pos[-1] - com_pos[-2]) / (timestamps[-1] - timestamps[-2])

    return com_vel


### Removes the bias from the initial pose (world frame is 0 0 0  0 0 0 1 at time t = 0 )
def remove_gt_bias(positions, orientations):
    # Get the initial position and orientation
    initial_position = positions[0]  # Initial position at time t=0
    q0 = orientations[0]  # Initial orientation at time t=0

    # Modify the position and orientation to be relative to the world frame (t=0 as origin)
    # Subtract initial position from all positions to set the initial position to (0, 0, 0)
    positions = positions - initial_position
    # You can modify the orientation similarly by applying the inverse of the initial quaternion.
    # Assuming orientations are in quaternion format (x, y, z, w), we need to inverse the initial orientation
    initial_orientation_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
    orientations = np.array(
        [quaternion_multiply(initial_orientation_inv, q) for q in orientations]
    )
    return positions, orientations


if __name__ == "__main__":
    (
        gt_pos,
        gt_rot,
        imu,
        L_forces,
        R_forces
    ) = load_gt_data(measurement_file)
   
    (
        timestamps,
        est_pos_x,
        est_pos_y,
        est_pos_z,
        com_pos_x,
        com_pos_y,
        com_pos_z,
        com_vel_x,
        com_vel_y,
        com_vel_z,
        est_rot_x,
        est_rot_y,
        est_rot_z,
        est_rot_w,
        b_ax,
        b_ay,
        b_az,
        b_wx,
        b_wy,
        b_wz,
    ) = load_serow_preds(prediction_file)

    gt_pos = gt_pos[:(-1)]
    gt_rot = gt_rot[:(-1)]
    R_forces = R_forces[:(-1)]
    L_forces = L_forces[:(-1)]

    gt_pos, gt_rot = remove_gt_bias(gt_pos, gt_rot)

    print("Base position ATE: ", compute_ATE_pos(gt_pos, est_pos_x, est_pos_y, est_pos_z))
    print("Base rotation ATE: ", compute_ATE_rot(gt_rot, est_rot_w, est_rot_x, est_rot_y, est_rot_z))

    # Plotting Ground Truth and Estimated Position (x, y, z)
    figPos_GT, axsPos_GT = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    figPos_GT.suptitle("Base position")

    axsPos_GT[0].plot(timestamps, gt_pos[:, 0], label="Ground Truth", color="blue")
    axsPos_GT[0].plot(timestamps, est_pos_x, label="Estimated", color="orange", linestyle="--")
    axsPos_GT[0].set_ylabel("base_pos_x")
    axsPos_GT[0].legend()

    axsPos_GT[1].plot(timestamps, gt_pos[:, 1], label="Ground Truth", color="blue")
    axsPos_GT[1].plot(timestamps, est_pos_y, label="Estimated", color="orange", linestyle="--")
    axsPos_GT[1].set_ylabel("base_pos_y")
    axsPos_GT[1].legend()

    axsPos_GT[2].plot(timestamps, gt_pos[:, 2], label="Ground Truth", color="blue")
    axsPos_GT[2].plot(timestamps, est_pos_z, label="Estimated", color="orange", linestyle="--")
    axsPos_GT[2].set_ylabel("base_pos_z")
    axsPos_GT[2].set_xlabel("Timestamp")
    axsPos_GT[2].legend()

    # Plotting Ground Truth and Estimated Orientation
    figRot_GT, axsRot_GT = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    figRot_GT.suptitle("Base Orientation")

    axsRot_GT[0].plot(timestamps, gt_rot[:, 0], label="Ground Truth", color="blue")
    axsRot_GT[0].plot(timestamps, est_rot_w, label="Estimated", color="orange", linestyle="--")
    axsRot_GT[0].set_ylabel("Orientation W")
    axsRot_GT[0].legend()

    axsRot_GT[1].plot(timestamps, gt_rot[:, 1], label="Ground Truth", color="blue")
    axsRot_GT[1].plot(timestamps, est_rot_x, label="Estimated", color="orange", linestyle="--")
    axsRot_GT[1].set_ylabel("Orientation X")
    axsRot_GT[1].legend()

    axsRot_GT[2].plot(timestamps, gt_rot[:, 2], label="Ground Truth", color="blue")
    axsRot_GT[2].plot(timestamps, est_rot_y, label="Estimated", color="orange", linestyle="--")
    axsRot_GT[2].set_ylabel("Orientation Y")
    axsRot_GT[2].legend()

    axsRot_GT[3].plot(timestamps, gt_rot[:, 3], label="Ground Truth", color="blue")
    axsRot_GT[3].plot(timestamps, est_rot_z, label="Estimated", color="orange", linestyle="--")
    axsRot_GT[3].set_ylabel("Orientation Z")
    axsRot_GT[3].set_xlabel("Timestamp")
    axsRot_GT[3].legend()



    # Plotting Ground Truth and Estimated Position (x, y, z)
    figForce, axsForce = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    figForce.suptitle("Feet forces (z-axis only)")

    axsForce[0].plot(timestamps, R_forces[:, 2], label="Ground Truth", color="blue")
    axsForce[0].set_ylabel("Right Foot Force")

    axsForce[1].plot(timestamps, L_forces[:, 2], label="Ground Truth", color="blue")
    axsForce[1].set_ylabel("Left Foot Force")
    axsForce[1].set_xlabel("Timestamp")

    # Plotting Ground Truth and Estimated Position (x, y, z)
    figIMU_b, axsIMU_b = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
    figIMU_b.suptitle("IMU estimated biases")

    axsIMU_b[0].plot(timestamps, b_ax, label="Ground Truth", color="blue")
    axsIMU_b[0].set_ylabel("bias ax")

    axsIMU_b[1].plot(timestamps, b_ay, color="blue")
    axsIMU_b[1].set_ylabel("bias ay")

    axsIMU_b[2].plot(timestamps, b_az, color="blue")
    axsIMU_b[2].set_ylabel("bias az")

    axsIMU_b[3].plot(timestamps, b_wx, color="blue")
    axsIMU_b[3].set_ylabel("bias wx")

    axsIMU_b[4].plot(timestamps, b_wy, color="blue")
    axsIMU_b[4].set_ylabel("bias wy")

    axsIMU_b[5].plot(timestamps, b_wz, color="blue")
    axsIMU_b[5].set_ylabel("bias wz")
    axsIMU_b[3].set_xlabel("Timestamp")

    plt.tight_layout()

    if display_plots:
        plt.show()
