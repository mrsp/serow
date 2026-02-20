import matplotlib.pyplot as plt
import numpy as np
import os
import json
import re
from mcap.reader import make_reader
from scipy.spatial.transform import Rotation as R


CONFIG_FILE = "test_config.json"

if not os.path.exists(CONFIG_FILE):
    if os.path.exists(f"../{CONFIG_FILE}"):
        CONFIG_FILE = f"../{CONFIG_FILE}"
    else:
        raise FileNotFoundError(f"Configuration file {CONFIG_FILE} not found.")

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

script_dir = os.path.dirname(os.path.abspath(__file__))

robot_name = config["Target"].get("robot", "g1")
exp_name   = config["Target"].get("experiment", "straight")
base_path_cfg = config["Paths"].get("base_path", ".")

full_exp_name = f"{robot_name}_{exp_name}"
gt_topic_name = f"/robot_state"      # Data MCAP Topic
est_topic_name = "serow_predictions" # Estimator Topic

def resolve_template(raw_path):
    s = raw_path.replace("{base_path}", base_path_cfg)
    s = s.replace("{robot}", robot_name)
    s = s.replace("{experiment}", exp_name)
    s = s.replace("{full_exp}", full_exp_name)
    return s.replace("//", "/")

def get_absolute_path(rel_path):
    if rel_path.startswith("/"):
        return rel_path
    
    project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
    path_from_root = os.path.join(project_root, rel_path)
    path_from_script = os.path.join(script_dir, rel_path)
    
    if os.path.exists(path_from_root):
        return path_from_root
    elif os.path.exists(path_from_script):
        return path_from_script
    else:
        return path_from_root

MEASUREMENT_FILE = get_absolute_path(resolve_template(config["Paths"]["data_file"]))
PREDICTION_FILE = get_absolute_path(resolve_template(config["Paths"]["prediction_file"]))

print(f"[CONFIG] Robot:      {robot_name}")
print(f"[CONFIG] Experiment: {exp_name}")
print(f"[CONFIG] GT File:    {MEASUREMENT_FILE}")
print(f"[CONFIG] Est File:   {PREDICTION_FILE}")


def load_gt(mcap_file, topic):
    """Loads Ground Truth data (Pos, Rot, Vel, IMU, Forces)"""
    data_store = {
        "ts": [], "pos": [], "rot": [], "lin_vel": [],
        "acc": [], "gyr": [], "f_left": [], "f_right": []
    }
    
    if not os.path.exists(mcap_file):
        print(f"Error: GT File not found at {mcap_file}")
        return None

    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=[topic]):
            d = json.loads(message.data)
            
            data_store["ts"].append(d["timestamp"])
            
            gt = d["base_ground_truth"]
            data_store["pos"].append([gt["position"]["x"], gt["position"]["y"], gt["position"]["z"]])
            # Store Quaternion as [w, x, y, z]
            data_store["rot"].append([gt["orientation"]["w"], gt["orientation"]["x"], gt["orientation"]["y"], gt["orientation"]["z"]])
            data_store["lin_vel"].append([gt["linear_velocity"]["x"], gt["linear_velocity"]["y"], gt["linear_velocity"]["z"]])
            
            data_store["acc"].append([d["imu"]["linear_acceleration"]["x"], d["imu"]["linear_acceleration"]["y"], d["imu"]["linear_acceleration"]["z"]])
            data_store["gyr"].append([d["imu"]["angular_velocity"]["x"], d["imu"]["angular_velocity"]["y"], d["imu"]["angular_velocity"]["z"]])
            
            forces = d.get("feet_forces", {})
            def get_f(key):
                if key in forces: return [forces[key]["x"], forces[key]["y"], forces[key]["z"]]
                return [0.0, 0.0, 0.0]

            if "left" in forces:
                data_store["f_left"].append(get_f("left"))
                data_store["f_right"].append(get_f("right"))
            elif "FL" in forces: 
                fl = np.array(get_f("FL"))
                rl = np.array(get_f("RL"))
                fr = np.array(get_f("FR"))
                rr = np.array(get_f("RR"))
                data_store["f_left"].append(fl + rl)
                data_store["f_right"].append(fr + rr)

    return {k: np.array(v) for k, v in data_store.items()}

def load_est(mcap_file, topic):
    """Loads Serow Predictions"""
    data_store = {
        "ts": [], "pos": [], "rot": [], "lin_vel": [],
        "b_acc": [], "b_gyr": []
    }

    if not os.path.exists(mcap_file):
        print(f"Error: Prediction File not found at {mcap_file}")
        return None

    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=[topic]):
            d = json.loads(message.data)

            data_store["ts"].append(d["timestamp"])

            # Base Pose
            bp = d["base_pose"]
            data_store["pos"].append([bp["position"]["x"], bp["position"]["y"], bp["position"]["z"]])
            # Store Quaternion as [w, x, y, z]
            data_store["rot"].append([bp["rotation"]["w"], bp["rotation"]["x"], bp["rotation"]["y"], bp["rotation"]["z"]])
            
            if "linear_velocity" in bp:
                data_store["lin_vel"].append([bp["linear_velocity"]["x"], bp["linear_velocity"]["y"], bp["linear_velocity"]["z"]])
            else:
                data_store["lin_vel"].append([0.0, 0.0, 0.0])

            # Bias
            if "imu_bias" in d:
                bias = d["imu_bias"]
                data_store["b_acc"].append([bias["accel"]["x"], bias["accel"]["y"], bias["accel"]["z"]])
                data_store["b_gyr"].append([bias["angVel"]["x"], bias["angVel"]["y"], bias["angVel"]["z"]])

    return {k: np.array(v) for k, v in data_store.items()}

def align_gt_to_est(gt_pos, gt_rot):
    """
    Transforms GT trajectory to start at (0,0,0) with Identity rotation,
    matching the typical startup state of the Estimator.
    """
    if len(gt_pos) == 0: return gt_pos, gt_rot

    # Get Initial GT Pose
    p0 = gt_pos[0]
    q0 = gt_rot[0] # [w, x, y, z]
    
    # Create Rotation object for q0 (scipy expects [x, y, z, w])
    r0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])
    r0_inv = r0.inv()

    # Align Positions: P_aligned = R0_inv * (P - P0)
    pos_centered = gt_pos - p0
    pos_aligned = r0_inv.apply(pos_centered)

    # Align Rotations: R_aligned = R0_inv * R
    r_current = R.from_quat(gt_rot[:, [1, 2, 3, 0]]) # Convert input [w,x,y,z] to [x,y,z,w]
    r_aligned = r0_inv * r_current
    
    # Convert back to [w, x, y, z]
    rot_aligned_xyzw = r_aligned.as_quat()
    rot_aligned = rot_aligned_xyzw[:, [3, 0, 1, 2]]
    
    return pos_aligned, rot_aligned

def compute_ATE(gt_ts, gt_pos, est_ts, est_pos):
    """Computes RMSE between aligned GT and Est."""
    if len(gt_ts) == 0 or len(est_ts) == 0: return 0.0

    t_start = max(gt_ts[0], est_ts[0])
    t_end = min(gt_ts[-1], est_ts[-1])
    
    idx_est = np.where((est_ts >= t_start) & (est_ts <= t_end))
    eval_ts = est_ts[idx_est]
    eval_est = est_pos[idx_est]
    
    if len(eval_ts) == 0: return 0.0

    # Interpolate GT to Est timestamps
    eval_gt_x = np.interp(eval_ts, gt_ts, gt_pos[:, 0])
    eval_gt_y = np.interp(eval_ts, gt_ts, gt_pos[:, 1])
    eval_gt_z = np.interp(eval_ts, gt_ts, gt_pos[:, 2])
    eval_gt = np.column_stack((eval_gt_x, eval_gt_y, eval_gt_z))
    
    error = np.linalg.norm(eval_gt - eval_est, axis=1)
    rmse = np.sqrt(np.mean(error**2))
    return rmse


gt_data = load_gt(MEASUREMENT_FILE, gt_topic_name)
est_data = load_est(PREDICTION_FILE, est_topic_name)

if not gt_data or len(gt_data["ts"]) == 0:
    print("FATAL: Ground Truth data is empty or missing.")
    exit(1)

if not est_data or len(est_data["ts"]) == 0:
    print("FATAL: Prediction data is empty or missing. Check your C++ writer.")
    exit(1)

# --- Alignment ---
print("Aligning Ground Truth to Estimation Start Frame...")
gt_pos_aligned, gt_rot_aligned = align_gt_to_est(gt_data["pos"], gt_data["rot"])

# --- Metrics ---
ate = compute_ATE(gt_data["ts"], gt_pos_aligned, est_data["ts"], est_data["pos"])
print(f"=========================================")
print(f" ATE (Position RMSE): {ate:.4f} m")
print(f"=========================================")

# --- Plotting ---
ts_gt = gt_data["ts"]
ts_est = est_data["ts"]

# Figure 1: Position & Velocity
fig1, axs1 = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
fig1.suptitle(f"Serow State Estimation - {robot_name}\nATE: {ate:.4f} m")
labels = ["X", "Y", "Z"]

# Col 1: Position
for i in range(3):
    axs1[i, 0].plot(ts_gt, gt_pos_aligned[:, i], label="GT (Aligned)", color="black", alpha=0.6)
    axs1[i, 0].plot(ts_est, est_data["pos"][:, i], label="Estimated", color="red", linestyle="--")
    axs1[i, 0].set_ylabel(f"Pos {labels[i]} (m)")
    axs1[i, 0].grid(True, alpha=0.3)
axs1[0, 0].legend()
axs1[0, 0].set_title("Base Position")

# Col 2: Linear Velocity
q0 = gt_data["rot"][0]
r0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])
r0_inv = r0.inv()
gt_vel_aligned = r0_inv.apply(gt_data["lin_vel"])

for i in range(3):
    axs1[i, 1].plot(ts_gt, gt_vel_aligned[:, i], label="GT (Aligned)", color="black", alpha=0.6)
    axs1[i, 1].plot(ts_est, est_data["lin_vel"][:, i], label="Estimated", color="red", linestyle="--")
    axs1[i, 1].set_ylabel(f"Vel {labels[i]} (m/s)")
    axs1[i, 1].grid(True, alpha=0.3)
axs1[0, 1].set_title("Base Linear Velocity")
axs1[2, 0].set_xlabel("Time (s)")
axs1[2, 1].set_xlabel("Time (s)")

# Figure 2: Orientation & Biases
fig2, axs2 = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
fig2.suptitle("Orientation & IMU Biases")

# Col 1: Orientation (Quaternions)
quat_labels = ["W", "X", "Y", "Z"]
for i in range(4):
    axs2[i, 0].plot(ts_gt, gt_rot_aligned[:, i], label="GT (Aligned)", color="black", alpha=0.6)
    axs2[i, 0].plot(ts_est, est_data["rot"][:, i], label="Est", color="red", linestyle="--")
    axs2[i, 0].set_ylabel(f"Q {quat_labels[i]}")
    axs2[i, 0].grid(True, alpha=0.3)
axs2[0, 0].set_title("Orientation (Quaternion)")

# Col 2: Biases & Forces (Context)
# Plot Accel Bias
axs2[0, 1].plot(ts_est, est_data["b_acc"], label=["ax", "ay", "az"])
axs2[0, 1].set_title("Est. Accel Bias")
axs2[0, 1].grid(True)
axs2[0, 1].legend()

# Plot Gyro Bias
axs2[1, 1].plot(ts_est, est_data["b_gyr"], label=["wx", "wy", "wz"])
axs2[1, 1].set_title("Est. Gyro Bias")
axs2[1, 1].grid(True)
axs2[1, 1].legend()

# Plot GT Forces (Context for slip)
axs2[2, 1].plot(ts_gt, gt_data["f_left"][:, 2], label="Left/FL Z", color="blue", alpha=0.5)
axs2[2, 1].set_title("GT Force Left (Z)")
axs2[2, 1].grid(True)

axs2[3, 1].plot(ts_gt, gt_data["f_right"][:, 2], label="Right/FR Z", color="orange", alpha=0.5)
axs2[3, 1].set_title("GT Force Right (Z)")
axs2[3, 1].grid(True)
axs2[3, 1].set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

