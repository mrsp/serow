import matplotlib.pyplot as plt
import numpy as np
import os
import json
from mcap.reader import make_reader
from scipy.spatial.transform import Rotation as R

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
DISPLAY_PLOTS = True
CONFIG_FILE = "test_config.json"

if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Configuration file {CONFIG_FILE} not found.")

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

serow_path = os.environ.get("SEROW_PATH")

if not serow_path:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    serow_path = os.path.abspath(os.path.join(script_dir, "..", ".."))

base_path = config["Paths"]["base_path"]
experiment_type = config["Experiment"]["type"]

def fix_extension(path):
    if path.endswith(".h5"):
        return path.replace(".h5", ".mcap")
    return path

raw_meas_path = config["Paths"]["data_file"].replace("{base_path}", base_path).replace("{type}", experiment_type)
raw_pred_path = config["Paths"]["prediction_file"].replace("{base_path}", base_path).replace("{type}", experiment_type)

MEASUREMENT_FILE = os.path.join(serow_path, fix_extension(raw_meas_path).lstrip("/"))
PREDICTION_FILE = os.path.join(serow_path, fix_extension(raw_pred_path).lstrip("/"))

# -------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# -------------------------------------------------------------------------

def load_gt_data(mcap_file):
    """Loads Ground Truth data from the input MCAP."""
    data_store = {
        "pos": [], "rot": [], "lin_vel": [], "ts": [], "com": [],
        "acc": [], "gyr": [], "f_fl": [], "f_fr": [], "f_rl": [], "f_rr": []
    }

    print(f"Loading Ground Truth from: {mcap_file}")
    
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=["/robot_state"]):
            d = json.loads(message.data)
            
            data_store["ts"].append(d["timestamp"])
            
            # Base State
            gt = d["base_ground_truth"]
            data_store["pos"].append([gt["position"]["x"], gt["position"]["y"], gt["position"]["z"]])
            # Quaternion stored as [w, x, y, z]
            data_store["rot"].append([gt["orientation"]["w"], gt["orientation"]["x"], gt["orientation"]["y"], gt["orientation"]["z"]])
            data_store["lin_vel"].append([gt["linear_velocity"]["x"], gt["linear_velocity"]["y"], gt["linear_velocity"]["z"]])
            data_store["com"].append([gt["com_position"]["x"], gt["com_position"]["y"], gt["com_position"]["z"]])
            
            # IMU
            data_store["acc"].append([d["imu"]["linear_acceleration"]["x"], d["imu"]["linear_acceleration"]["y"], d["imu"]["linear_acceleration"]["z"]])
            data_store["gyr"].append([d["imu"]["angular_velocity"]["x"], d["imu"]["angular_velocity"]["y"], d["imu"]["angular_velocity"]["z"]])
            
            # Forces
            ff = d["feet_forces"]
            data_store["f_fl"].append([ff["FL"]["x"], ff["FL"]["y"], ff["FL"]["z"]])
            data_store["f_fr"].append([ff["FR"]["x"], ff["FR"]["y"], ff["FR"]["z"]])
            data_store["f_rl"].append([ff["RL"]["x"], ff["RL"]["y"], ff["RL"]["z"]])
            data_store["f_rr"].append([ff["RR"]["x"], ff["RR"]["y"], ff["RR"]["z"]])

    # Convert all lists to numpy arrays
    return {k: np.array(v) for k, v in data_store.items()}


def load_serow_preds(mcap_file):
    """Loads Estimated data from the output MCAP."""
    data_store = {
        "pos": [], "rot": [], "lin_vel": [], "ts": [], "com_pos": [], "com_vel": [],
        "b_acc": [], "b_gyr": []
    }

    print(f"Loading Predictions from: {mcap_file}")

    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=["serow_predictions"]):
            d = json.loads(message.data)

            # Important: Get the timestamp!
            data_store["ts"].append(d["timestamp"])

            # Base Pose
            bp = d["base_pose"]
            data_store["pos"].append([bp["position"]["x"], bp["position"]["y"], bp["position"]["z"]])
            data_store["rot"].append([bp["rotation"]["w"], bp["rotation"]["x"], bp["rotation"]["y"], bp["rotation"]["z"]])
            
            if "linear_velocity" in bp:
                data_store["lin_vel"].append([bp["linear_velocity"]["x"], bp["linear_velocity"]["y"], bp["linear_velocity"]["z"]])
            else:
                data_store["lin_vel"].append([0.0, 0.0, 0.0])

            # CoM
            com = d["CoM_state"]
            data_store["com_pos"].append([com["position"]["x"], com["position"]["y"], com["position"]["z"]])
            data_store["com_vel"].append([com["velocity"]["x"], com["velocity"]["y"], com["velocity"]["z"]])

            # Bias
            bias = d["imu_bias"]
            data_store["b_acc"].append([bias["accel"]["x"], bias["accel"]["y"], bias["accel"]["z"]])
            data_store["b_gyr"].append([bias["angVel"]["x"], bias["angVel"]["y"], bias["angVel"]["z"]])

    return {k: np.array(v) for k, v in data_store.items()}

# -------------------------------------------------------------------------
# ALIGNMENT & METRICS
# -------------------------------------------------------------------------

def align_gt_to_estimation_frame(gt_pos, gt_rot):
    """
    Transforms the Ground Truth trajectory so that its initial pose
    matches the Estimator's initial pose (0,0,0 and Identity Rot).
    
    Args:
        gt_pos: (N, 3) array of positions
        gt_rot: (N, 4) array of quaternions [w, x, y, z]
    """
    # 1. Get Initial GT Pose
    p0 = gt_pos[0]
    q0 = gt_rot[0] # [w, x, y, z]
    
    # Create Rotation object for q0
    r0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])
    r0_inv = r0.inv()

    # 2. Align Positions
    # Formula: P_aligned = R0_inv * (P_current - P0)
    # This rotates the "world vector" into the "initial body frame"
    pos_centered = gt_pos - p0
    pos_aligned = r0_inv.apply(pos_centered)

    # 3. Align Rotations
    # Formula: Q_aligned = Q0_inv * Q_current
    r_current = R.from_quat(gt_rot[:, [1, 2, 3, 0]]) # Convert all to xyzw
    r_aligned = r0_inv * r_current
    
    # Convert back to [w, x, y, z] for plotting consistency
    rot_aligned_xyzw = r_aligned.as_quat()
    rot_aligned = rot_aligned_xyzw[:, [3, 0, 1, 2]]
    
    return pos_aligned, rot_aligned

def compute_ATE(gt_ts, gt_pos, est_ts, est_pos):
    """
    Computes Absolute Trajectory Error (RMSE) after temporal alignment.
    Interpolates GT to match Estimation timestamps.
    """
    # Only evaluate during the overlapping time period
    t_start = max(gt_ts[0], est_ts[0])
    t_end = min(gt_ts[-1], est_ts[-1])
    
    # Filter indices
    idx_est = np.where((est_ts >= t_start) & (est_ts <= t_end))
    eval_ts = est_ts[idx_est]
    eval_est = est_pos[idx_est]
    
    # Interpolate GT to these timestamps
    eval_gt_x = np.interp(eval_ts, gt_ts, gt_pos[:, 0])
    eval_gt_y = np.interp(eval_ts, gt_ts, gt_pos[:, 1])
    eval_gt_z = np.interp(eval_ts, gt_ts, gt_pos[:, 2])
    eval_gt = np.column_stack((eval_gt_x, eval_gt_y, eval_gt_z))
    
    # Compute RMSE
    error = np.linalg.norm(eval_gt - eval_est, axis=1)
    rmse = np.sqrt(np.mean(error**2))
    return rmse

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load Data
    gt = load_gt_data(MEASUREMENT_FILE)
    est = load_serow_preds(PREDICTION_FILE)

    if len(gt["ts"]) == 0 or len(est["ts"]) == 0:
        print("Error: Empty data arrays.")
        exit()

    # 2. Align Ground Truth to "Zero Start"
    print("Aligning Ground Truth to Estimation Frame (0,0,0)...")
    gt_pos_aligned, gt_rot_aligned = align_gt_to_estimation_frame(gt["pos"], gt["rot"])

    # 3. Compute Metrics
    ate_pos = compute_ATE(gt["ts"], gt_pos_aligned, est["ts"], est["pos"])
    print(f"========================================")
    print(f"Absolute Trajectory Error (ATE): {ate_pos:.4f} m")
    print(f"========================================")

    # 4. Plotting
    if DISPLAY_PLOTS:
        # --- Figure 1: Base Position ---
        fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig1.suptitle(f"Base Position (Aligned)\nATE: {ate_pos:.4f}m")

        labels = ["X", "Y", "Z"]
        for i in range(3):
            axs1[i].plot(gt["ts"], gt_pos_aligned[:, i], label="Ground Truth (Aligned)", color="blue")
            axs1[i].plot(est["ts"], est["pos"][:, i], label="Estimated", color="orange", linestyle="--")
            axs1[i].set_ylabel(f"Pos {labels[i]} (m)")
            axs1[i].grid(True, alpha=0.3)
            axs1[i].legend(loc="upper left")
        axs1[2].set_xlabel("Time (s)")

        # --- Figure 2: Base Orientation ---
        fig2, axs2 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        fig2.suptitle("Base Orientation (Aligned)")

        quat_labels = ["W", "X", "Y", "Z"]
        for i in range(4):
            axs2[i].plot(gt["ts"], gt_rot_aligned[:, i], label="GT (Aligned)", color="blue")
            axs2[i].plot(est["ts"], est["rot"][:, i], label="Est", color="orange", linestyle="--")
            axs2[i].set_ylabel(f"Quat {quat_labels[i]}")
            axs2[i].grid(True, alpha=0.3)
        axs2[0].legend()
        axs2[3].set_xlabel("Time (s)")

        # --- Figure 3: Linear Velocity (Body Frame approximation for simple comparison) ---
        fig3, axs3 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig3.suptitle("Linear Velocity (World Frame)")


        q0 = gt["rot"][0]
        r0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])
        r0_inv = r0.inv()
        gt_vel_aligned = r0_inv.apply(gt["lin_vel"])

        for i in range(3):
            axs3[i].plot(gt["ts"], gt_vel_aligned[:, i], label="GT (Aligned)", color="blue")
            axs3[i].plot(est["ts"], est["lin_vel"][:, i], label="Est", color="orange", linestyle="--")
            axs3[i].set_ylabel(f"Vel {labels[i]} (m/s)")
            axs3[i].grid(True, alpha=0.3)
        axs3[0].legend()
        axs3[2].set_xlabel("Time (s)")

        # --- Figure 4: Biases ---
        fig4, axs4 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig4.suptitle("Estimated IMU Biases")
        
        axs4[0].plot(est["ts"], est["b_acc"], label=["ax", "ay", "az"])
        axs4[0].set_ylabel("Accel Bias (m/s^2)")
        axs4[0].legend()
        axs4[0].grid(True)
        
        axs4[1].plot(est["ts"], est["b_gyr"], label=["wx", "wy", "wz"])
        axs4[1].set_ylabel("Gyro Bias (rad/s)")
        axs4[1].legend()
        axs4[1].grid(True)
        axs4[1].set_xlabel("Time (s)")

        # --- Figure 5: Forces (Diagnostic) ---
        fig5, axs5 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig5.suptitle("Feet Forces Z (GT)")
        
        axs5[0].plot(gt["ts"], gt["f_fl"][:, 2], color="blue", label="FL")
        axs5[1].plot(gt["ts"], gt["f_fr"][:, 2], color="blue", label="FR")
        axs5[2].plot(gt["ts"], gt["f_rl"][:, 2], color="blue", label="RL")
        axs5[3].plot(gt["ts"], gt["f_rr"][:, 2], color="blue", label="RR")
        
        for ax in axs5:
            ax.set_ylabel("Force Z")
            ax.grid(True)
            ax.legend(loc="upper right")
        axs5[3].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()
