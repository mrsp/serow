import matplotlib.pyplot as plt
import numpy as np
import os
import json
from mcap.reader import make_reader

display_plots = True

config_file = "test_config.json"  # Path to your JSON config file
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration file {config_file} not found.")

with open(config_file, "r") as f:
    config = json.load(f)

serow_path = os.environ.get("SEROW_PATH")
if not serow_path:
    serow_path = "" 

base_path = config["Paths"]["base_path"]
experiment_type = config["Experiment"]["type"]

def fix_extension(path):
    if path.endswith(".h5"):
        return path.replace(".h5", ".mcap")
    return path

raw_meas_path = config["Paths"]["data_file"].replace("{base_path}", base_path).replace("{type}", experiment_type)
raw_pred_path = config["Paths"]["prediction_file"].replace("{base_path}", base_path).replace("{type}", experiment_type)

measurement_file = serow_path + fix_extension(raw_meas_path)
prediction_file = serow_path + fix_extension(raw_pred_path)


# Load the Ground Truth data from the MCAP file
def load_gt_data(mcap_file):
    positions = []
    orientations = []
    timestamps = []
    com_positions = []
    FL_forces = []
    FR_forces = []
    RL_forces = []
    RR_forces = []
    imu_accel = []
    imu_gyro = []  

    print(f"Loading Ground Truth from: {mcap_file}")
    
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=["/robot_state"]):
            data = json.loads(message.data)
            
            timestamps.append(data["timestamp"])
            
            # Base Ground Truth
            gt = data["base_ground_truth"]
            positions.append([gt["position"]["x"], gt["position"]["y"], gt["position"]["z"]])
            # Quaternion order [w, x, y, z]
            orientations.append([gt["orientation"]["w"], gt["orientation"]["x"], gt["orientation"]["y"], gt["orientation"]["z"]])
            com_positions.append([gt["com_position"]["x"], gt["com_position"]["y"], gt["com_position"]["z"]])
            
            # IMU
            acc = data["imu"]["linear_acceleration"]
            gyr = data["imu"]["angular_velocity"] 
            
            imu_accel.append([acc["x"], acc["y"], acc["z"]])
            imu_gyro.append([gyr["x"], gyr["y"], gyr["z"]])
            
            # Forces
            ff = data["feet_forces"]
            FL_forces.append([ff["FL"]["x"], ff["FL"]["y"], ff["FL"]["z"]])
            FR_forces.append([ff["FR"]["x"], ff["FR"]["y"], ff["FR"]["z"]])
            RL_forces.append([ff["RL"]["x"], ff["RL"]["y"], ff["RL"]["z"]])
            RR_forces.append([ff["RR"]["x"], ff["RR"]["y"], ff["RR"]["z"]])

    return (
        np.array(positions),
        np.array(orientations),
        np.array(com_positions),
        np.array(imu_accel),
        np.array(imu_gyro), # Return Gyro
        np.array(timestamps),
        np.array(FL_forces),
        np.array(FR_forces),
        np.array(RL_forces),
        np.array(RR_forces),
    )


# Load the Estimated data from the MCAP file
def load_serow_preds(mcap_file):
    com_pos = []
    com_vel = []
    base_pos = []
    base_rot = []
    imu_bias_acc = []
    imu_bias_gyr = []

    print(f"Loading Predictions from: {mcap_file}")

    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=["serow_predictions"]):
            data = json.loads(message.data)

            # CoM
            com = data["CoM_state"]
            com_pos.append([com["position"]["x"], com["position"]["y"], com["position"]["z"]])
            com_vel.append([com["velocity"]["x"], com["velocity"]["y"], com["velocity"]["z"]])

            # Base Pose
            bp = data["base_pose"]
            base_pos.append([bp["position"]["x"], bp["position"]["y"], bp["position"]["z"]])
            
            base_rot.append([bp["rotation"]["w"], bp["rotation"]["x"], bp["rotation"]["y"], bp["rotation"]["z"]])

            # Bias
            bias = data["imu_bias"]
            imu_bias_acc.append([bias["accel"]["x"], bias["accel"]["y"], bias["accel"]["z"]])
            imu_bias_gyr.append([bias["angVel"]["x"], bias["angVel"]["y"], bias["angVel"]["z"]])

    # Convert to numpy
    com_pos = np.array(com_pos)
    com_vel = np.array(com_vel)
    base_pos = np.array(base_pos)
    base_rot = np.array(base_rot)
    imu_bias_acc = np.array(imu_bias_acc)
    imu_bias_gyr = np.array(imu_bias_gyr)
    
    return (
        base_pos[:, 0], base_pos[:, 1], base_pos[:, 2], # pos x, y, z
        com_pos[:, 0], com_pos[:, 1], com_pos[:, 2],    # com pos x, y, z
        com_vel[:, 0], com_vel[:, 1], com_vel[:, 2],    # com vel x, y, z
        base_rot[:, 1], base_rot[:, 2], base_rot[:, 3], base_rot[:, 0], # rot x, y, z, w 
        imu_bias_acc[:, 0], imu_bias_acc[:, 1], imu_bias_acc[:, 2], # b_ax, b_ay, b_az
        imu_bias_gyr[:, 0], imu_bias_gyr[:, 1], imu_bias_gyr[:, 2], # b_wx, b_wy, b_wz
    )


def compute_ATE_pos(gt_pos, est_x, est_y, est_z):
    est_pos = np.column_stack((est_x, est_y, est_z))
    # Ensure dimensions match for calculation
    n = min(gt_pos.shape[0], est_pos.shape[0])
    error = np.linalg.norm(gt_pos[:n] - est_pos[:n], axis=1)
    ate = np.sqrt(np.mean(error**2))
    return ate


def compute_ATE_rot(gt_rot, est_rot_w, est_rot_x, est_rot_y, est_rot_z):
    est_rot = np.column_stack((est_rot_w, est_rot_x, est_rot_y, est_rot_z))
    rotation_errors = []
    
    n = min(gt_rot.shape[0], est_rot.shape[0])

    for i in range(n):
        q_gt = gt_rot[i]
        q_est = est_rot[i]
        
        # Ensure quaternions are on the same hemisphere
        if np.dot(q_gt, q_est) < 0:
            q_est = -q_est
        
        q_gt_conj = np.array(
            [q_gt[0], -q_gt[1], -q_gt[2], -q_gt[3]]
        )
        q_rel = quaternion_multiply(q_gt_conj, q_est)
        rotation_errors.append(2 * np.arccos(np.clip(q_rel[0], -1.0, 1.0)))
    
    ate_rot = np.sqrt(np.mean(np.array(rotation_errors)**2))
    return ate_rot

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions q1 and q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


# Removes the bias from the initial pose (world frame is 0 0 0  0 0 0 1 at time t = 0 )
def remove_gt_bias(positions, orientations):
    initial_position = positions[0] 
    q0 = orientations[0] 

    positions = positions - initial_position
    initial_orientation_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
    orientations = np.array(
        [quaternion_multiply(initial_orientation_inv, q) for q in orientations]
    )
    return positions, orientations


if __name__ == "__main__":
    (
        gt_pos,
        gt_rot,
        com_pos,
        gt_acc,
        gt_gyr,
        timestamps,
        FL_forces,
        FR_forces,
        RL_forces,
        RR_forces,
    ) = load_gt_data(measurement_file)
    (
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

    # ---------------------------------------------------------
    # Synchronization Logic
    # ---------------------------------------------------------
    size_diff = gt_pos.shape[0] - est_pos_x.shape[0]
    
    # Check if dimensions are valid for slicing
    if size_diff < 0:
        print(f"Warning: Estimation has {abs(size_diff)} more frames than GT. Truncating Estimation.")
        est_pos_x = est_pos_x[:gt_pos.shape[0]]
        size_diff = 0
    
    # Slice the End of Estimates (remove last 10 frames as per original code)
    cut_end = -10
    
    est_pos_x = est_pos_x[:cut_end]
    est_pos_y = est_pos_y[:cut_end]
    est_pos_z = est_pos_z[:cut_end]
    est_rot_x = est_rot_x[:cut_end]
    est_rot_y = est_rot_y[:cut_end]
    est_rot_z = est_rot_z[:cut_end]
    est_rot_w = est_rot_w[:cut_end]

    com_pos_x = com_pos_x[:cut_end]
    com_pos_y = com_pos_y[:cut_end]
    com_pos_z = com_pos_z[:cut_end]
    com_vel_x = com_vel_x[:cut_end]
    com_vel_y = com_vel_y[:cut_end]
    com_vel_z = com_vel_z[:cut_end]

    b_ax = b_ax[:cut_end]
    b_ay = b_ay[:cut_end]
    b_az = b_az[:cut_end]
    b_wx = b_wx[:cut_end]
    b_wy = b_wy[:cut_end]
    b_wz = b_wz[:cut_end]

    # Slice the Start and End of Ground Truth
    gt_pos = gt_pos[(size_diff):cut_end]
    gt_rot = gt_rot[(size_diff):cut_end]

    gt_pos, gt_rot = remove_gt_bias(gt_pos, gt_rot)
    
    FL_forces = FL_forces[(size_diff):cut_end]
    FR_forces = FR_forces[(size_diff):cut_end]
    RL_forces = RL_forces[(size_diff):cut_end]
    RR_forces = RR_forces[(size_diff):cut_end]
    com_pos = com_pos[(size_diff):cut_end]
    gt_acc = gt_acc[(size_diff):cut_end]
    gt_gyr = gt_gyr[(size_diff):cut_end] 
    timestamps = timestamps[(size_diff):cut_end]

    print(
        "Base position ATE: ", compute_ATE_pos(gt_pos, est_pos_x, est_pos_y, est_pos_z)
    )
    print(
        "Base rotation ATE: ",
        compute_ATE_rot(gt_rot, est_rot_w, est_rot_x, est_rot_y, est_rot_z),
    )

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    
    # Plotting Ground Truth and Estimated Position (x, y, z)
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig1.suptitle("Base position")

    axs1[0].plot(timestamps, gt_pos[:, 0], label="Ground Truth", color="blue")
    axs1[0].plot(
        timestamps, est_pos_x, label="Estimated", color="orange", linestyle="--"
    )
    axs1[0].set_ylabel("base_pos_x")
    axs1[0].legend()

    axs1[1].plot(timestamps, gt_pos[:, 1], label="Ground Truth", color="blue")
    axs1[1].plot(
        timestamps, est_pos_y, label="Estimated", color="orange", linestyle="--"
    )
    axs1[1].set_ylabel("base_pos_y")
    axs1[1].legend()

    axs1[2].plot(timestamps, gt_pos[:, 2], label="Ground Truth", color="blue")
    axs1[2].plot(
        timestamps, est_pos_z, label="Estimated", color="orange", linestyle="--"
    )
    axs1[2].set_ylabel("base_pos_z")
    axs1[2].set_xlabel("Timestamp")
    axs1[2].legend()

    # Plotting Ground Truth and Estimated Orientation
    fig2, axs2 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig2.suptitle("Base Orientation")

    axs2[0].plot(timestamps, gt_rot[:, 0], label="Ground Truth", color="blue")
    axs2[0].plot(
        timestamps, est_rot_w, label="Estimated", color="orange", linestyle="--"
    )
    axs2[0].set_ylabel("Orientation W")
    axs2[0].legend()

    axs2[1].plot(timestamps, gt_rot[:, 1], label="Ground Truth", color="blue")
    axs2[1].plot(
        timestamps, est_rot_x, label="Estimated", color="orange", linestyle="--"
    )
    axs2[1].set_ylabel("Orientation X")
    axs2[1].legend()

    axs2[2].plot(timestamps, gt_rot[:, 2], label="Ground Truth", color="blue")
    axs2[2].plot(
        timestamps, est_rot_y, label="Estimated", color="orange", linestyle="--"
    )
    axs2[2].set_ylabel("Orientation Y")
    axs2[2].legend()

    axs2[3].plot(timestamps, gt_rot[:, 3], label="Ground Truth", color="blue")
    axs2[3].plot(
        timestamps, est_rot_z, label="Estimated", color="orange", linestyle="--"
    )
    axs2[3].set_ylabel("Orientation Z")
    axs2[3].set_xlabel("Timestamp")
    axs2[3].legend()

    # Plotting Ground Truth and Estimated CoM
    fig3, axs3 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig3.suptitle("CoM position")

    axs3[0].plot(timestamps, com_pos[:, 0], label="Ground Truth", color="blue")
    axs3[0].plot(
        timestamps, com_pos_x, label="Estimated", color="orange", linestyle="--"
    )
    axs3[0].set_ylabel("com_pos_x")
    axs3[0].legend()

    axs3[1].plot(timestamps, com_pos[:, 1], label="Ground Truth", color="blue")
    axs3[1].plot(
        timestamps, com_pos_y, label="Estimated", color="orange", linestyle="--"
    )
    axs3[1].set_ylabel("com_pos_y")
    axs3[1].legend()

    axs3[2].plot(timestamps, com_pos[:, 2], label="Ground Truth", color="blue")
    axs3[2].plot(
        timestamps, com_pos_z, label="Estimated", color="orange", linestyle="--"
    )
    axs3[2].set_ylabel("com_pos_z")
    axs3[2].legend()

    # Plotting Forces
    fig4, axs4 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig4.suptitle("Feet forces (z-axis only)")

    axs4[0].plot(timestamps, FL_forces[:, 2], label="Ground Truth", color="blue")
    axs4[0].set_ylabel("FORCE FL")

    axs4[1].plot(timestamps, FR_forces[:, 2], label="Ground Truth", color="blue")
    axs4[1].set_ylabel("FORCE FR")

    axs4[2].plot(timestamps, RL_forces[:, 2], label="Ground Truth", color="blue")
    axs4[2].set_ylabel("FORCE RL")

    axs4[3].plot(timestamps, RR_forces[:, 2], label="Ground Truth", color="blue")
    axs4[3].set_ylabel("FORCE RR")
    axs4[3].set_xlabel("Timestamp")

    # Plotting Biases
    fig5, axs5 = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
    fig5.suptitle("IMU estimated biases")

    axs5[0].plot(timestamps, b_ax, label="Estimated", color="blue")
    axs5[0].set_ylabel("bias ax")

    axs5[1].plot(timestamps, b_ay, color="blue")
    axs5[1].set_ylabel("bias ay")

    axs5[2].plot(timestamps, b_az, color="blue")
    axs5[2].set_ylabel("bias az")

    axs5[3].plot(timestamps, b_wx, color="blue")
    axs5[3].set_ylabel("bias wx")

    axs5[4].plot(timestamps, b_wy, color="blue")
    axs5[4].set_ylabel("bias wy")

    axs5[5].plot(timestamps, b_wz, color="blue")
    axs5[5].set_ylabel("bias wz")
    axs5[5].set_xlabel("Timestamp")

    # IMU measurements
    fig6, axs6 = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    fig6.suptitle("IMU Measurements (Ground Truth)")

    # Accel X
    axs6[0].plot(timestamps, gt_acc[:, 0], color="green", label="Accel X")
    axs6[0].set_ylabel("Accel X (m/s^2)")
    axs6[0].legend()

    # Accel Y
    axs6[1].plot(timestamps, gt_acc[:, 1], color="green", label="Accel Y")
    axs6[1].set_ylabel("Accel Y (m/s^2)")
    axs6[1].legend()

    # Accel Z
    axs6[2].plot(timestamps, gt_acc[:, 2], color="green", label="Accel Z")
    axs6[2].set_ylabel("Accel Z (m/s^2)")
    axs6[2].legend()

    # Gyro X
    axs6[3].plot(timestamps, gt_gyr[:, 0], color="red", label="Gyro X")
    axs6[3].set_ylabel("Gyro X (rad/s)")
    axs6[3].legend()

    # Gyro Y
    axs6[4].plot(timestamps, gt_gyr[:, 1], color="red", label="Gyro Y")
    axs6[4].set_ylabel("Gyro Y (rad/s)")
    axs6[4].legend()

    # Gyro Z
    axs6[5].plot(timestamps, gt_gyr[:, 2], color="red", label="Gyro Z")
    axs6[5].set_ylabel("Gyro Z (rad/s)")
    axs6[5].set_xlabel("Timestamp")
    axs6[5].legend()

    plt.tight_layout()

    if display_plots:
        plt.show()
