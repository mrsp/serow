import matplotlib.pyplot as plt
import numpy as np
import os
import json
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
gt_topic_name = f"/robot_state"      
est_topic_name = "serow_predictions" 

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
    data_store = {
        "ts": [], "pos": [], "rot": [], "lin_vel": [],
        "prob_left": [], "prob_right": []
    }

    if not os.path.exists(mcap_file):
        print(f"Error: Prediction File not found at {mcap_file}")
        return None

    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=[topic]):
            d = json.loads(message.data)

            data_store["ts"].append(d["timestamp"])

            bp = d["base_pose"]
            data_store["pos"].append([bp["position"]["x"], bp["position"]["y"], bp["position"]["z"]])
            data_store["rot"].append([bp["rotation"]["w"], bp["rotation"]["x"], bp["rotation"]["y"], bp["rotation"]["z"]])
            
            if "linear_velocity" in bp:
                data_store["lin_vel"].append([bp["linear_velocity"]["x"], bp["linear_velocity"]["y"], bp["linear_velocity"]["z"]])
            else:
                data_store["lin_vel"].append([0.0, 0.0, 0.0])

            # Extract Contact Probabilities
            probs = d.get("contact_probabilities", {})
            p_left = 0.0
            p_right = 0.0
            
            if probs:
                # Group by left/right using keyword matching in the link names
                l_vals = [v for k, v in probs.items() if 'left' in k.lower() or 'fl' in k.lower() or 'rl' in k.lower()]
                r_vals = [v for k, v in probs.items() if 'right' in k.lower() or 'fr' in k.lower() or 'rr' in k.lower()]
                
                # If a quadruped has two left feet, take the max probability to show the "left side" is in contact
                if l_vals: p_left = max(l_vals)
                if r_vals: p_right = max(r_vals)
                
            data_store["prob_left"].append(p_left)
            data_store["prob_right"].append(p_right)

    return {k: np.array(v) for k, v in data_store.items()}

def align_gt_to_est(gt_pos, gt_rot):
    if len(gt_pos) == 0: return gt_pos, gt_rot

    p0 = gt_pos[0]
    q0 = gt_rot[0] 
    r0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])
    r0_inv = r0.inv()

    pos_centered = gt_pos - p0
    pos_aligned = r0_inv.apply(pos_centered)

    r_current = R.from_quat(gt_rot[:, [1, 2, 3, 0]])
    r_aligned = r0_inv * r_current
    
    rot_aligned_xyzw = r_aligned.as_quat()
    rot_aligned = rot_aligned_xyzw[:, [3, 0, 1, 2]]
    
    return pos_aligned, rot_aligned

def compute_ATE(gt_ts, gt_pos, est_ts, est_pos):
    if len(gt_ts) == 0 or len(est_ts) == 0: return 0.0

    t_start = max(gt_ts[0], est_ts[0])
    t_end = min(gt_ts[-1], est_ts[-1])
    
    idx_est = np.where((est_ts >= t_start) & (est_ts <= t_end))
    eval_ts = est_ts[idx_est]
    eval_est = est_pos[idx_est]
    
    if len(eval_ts) == 0: return 0.0

    eval_gt_x = np.interp(eval_ts, gt_ts, gt_pos[:, 0])
    eval_gt_y = np.interp(eval_ts, gt_ts, gt_pos[:, 1])
    eval_gt_z = np.interp(eval_ts, gt_ts, gt_pos[:, 2])
    eval_gt = np.column_stack((eval_gt_x, eval_gt_y, eval_gt_z))
    
    error = np.linalg.norm(eval_gt - eval_est, axis=1)
    return np.sqrt(np.mean(error**2))

gt_data = load_gt(MEASUREMENT_FILE, gt_topic_name)
est_data = load_est(PREDICTION_FILE, est_topic_name)

if not gt_data or len(gt_data["ts"]) == 0:
    print("FATAL: Ground Truth data is empty.")
    exit(1)
if not est_data or len(est_data["ts"]) == 0:
    print("FATAL: Prediction data is empty.")
    exit(1)

print("Aligning Ground Truth to Estimation Start Frame...")
gt_pos_aligned, gt_rot_aligned = align_gt_to_est(gt_data["pos"], gt_data["rot"])

ate = compute_ATE(gt_data["ts"], gt_pos_aligned, est_data["ts"], est_data["pos"])
print(f"=========================================")
print(f" ATE (Position RMSE): {ate:.4f} m")
print(f"=========================================")

ts_gt = gt_data["ts"]
ts_est = est_data["ts"]

r_gt = R.from_quat(gt_rot_aligned[:, [1, 2, 3, 0]])
r_est = R.from_quat(est_data["rot"][:, [1, 2, 3, 0]])

rpy_gt = np.unwrap(r_gt.as_euler('xyz', degrees=True), period=360, axis=0)
rpy_est = np.unwrap(r_est.as_euler('xyz', degrees=True), period=360, axis=0)

q0 = gt_data["rot"][0]
r0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])
gt_vel_aligned = r0.inv().apply(gt_data["lin_vel"])

# ---------------------------------------------------------
# Figure 1: Position, Velocity, and RPY Orientation
# ---------------------------------------------------------
fig1, axs1 = plt.subplots(3, 3, figsize=(18, 10), sharex=True)
fig1.suptitle(f"Serow State Estimation - {robot_name}\nATE: {ate:.4f} m")

# Column 1: Position
labels_pos = ["X", "Y", "Z"]
for i in range(3):
    axs1[i, 0].plot(ts_gt, gt_pos_aligned[:, i], label="GT", color="black", alpha=0.6)
    axs1[i, 0].plot(ts_est, est_data["pos"][:, i], label="Estimated", color="red", linestyle="--")
    axs1[i, 0].set_ylabel(f"Pos {labels_pos[i]} (m)")
    axs1[i, 0].grid(True, alpha=0.3)
axs1[0, 0].legend()
axs1[0, 0].set_title("Base Position")

# Column 2: Velocity
labels_vel = ["X", "Y", "Z"]
for i in range(3):
    axs1[i, 1].plot(ts_gt, gt_vel_aligned[:, i], label="GT", color="black", alpha=0.6)
    axs1[i, 1].plot(ts_est, est_data["lin_vel"][:, i], label="Estimated", color="red", linestyle="--")
    axs1[i, 1].set_ylabel(f"Vel {labels_vel[i]} (m/s)")
    axs1[i, 1].grid(True, alpha=0.3)
axs1[0, 1].set_title("Base Velocity")

# Column 3: Orientation (RPY)
labels_rpy = ["Roll", "Pitch", "Yaw"]
for i in range(3):
    axs1[i, 2].plot(ts_gt, rpy_gt[:, i], label="GT", color="black", alpha=0.6)
    axs1[i, 2].plot(ts_est, rpy_est[:, i], label="Estimated", color="red", linestyle="--")
    axs1[i, 2].set_ylabel(f"{labels_rpy[i]} (deg)")
    axs1[i, 2].grid(True, alpha=0.3)
axs1[0, 2].set_title("Orientation (Euler)")

for j in range(3):
    axs1[2, j].set_xlabel("Time (s)")

fig1.tight_layout()

# ---------------------------------------------------------
# Figure 2: IMU Measurements & Feet Forces + Probabilities
# ---------------------------------------------------------
fig2, axs2 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig2.suptitle("Raw IMU, Forces, and Est. Contact Probability")

# Accel
axs2[0].plot(ts_gt, gt_data["acc"][:, 0], label="Acc X")
axs2[0].plot(ts_gt, gt_data["acc"][:, 1], label="Acc Y")
axs2[0].plot(ts_gt, gt_data["acc"][:, 2], label="Acc Z")
axs2[0].set_ylabel("Accel (m/s²)")
axs2[0].grid(True, alpha=0.4)
axs2[0].legend(loc="upper right")

# Gyro
axs2[1].plot(ts_gt, gt_data["gyr"][:, 0], label="Gyr X")
axs2[1].plot(ts_gt, gt_data["gyr"][:, 1], label="Gyr Y")
axs2[1].plot(ts_gt, gt_data["gyr"][:, 2], label="Gyr Z")
axs2[1].set_ylabel("Gyro (rad/s)")
axs2[1].grid(True, alpha=0.4)
axs2[1].legend(loc="upper right")

# Left Force & Probability
axs2[2].plot(ts_gt, gt_data["f_left"][:, 2], color="blue", alpha=0.5, label="GT Force Z")
axs2[2].set_ylabel("Left Force Z (N)", color="blue")
axs2[2].tick_params(axis='y', labelcolor="blue")
axs2[2].grid(True, alpha=0.4)

ax_prob_l = axs2[2].twinx()  # Create a twin Y-axis
ax_prob_l.plot(ts_est, est_data["prob_left"], color="red", linestyle=":", linewidth=2, label="Est Prob")
ax_prob_l.set_ylabel("Est Probability", color="red")
ax_prob_l.set_ylim(-0.1, 1.1)  # Bound it cleanly
ax_prob_l.tick_params(axis='y', labelcolor="red")

# Right Force & Probability
axs2[3].plot(ts_gt, gt_data["f_right"][:, 2], color="orange", alpha=0.5, label="GT Force Z")
axs2[3].set_ylabel("Right Force Z (N)", color="orange")
axs2[3].tick_params(axis='y', labelcolor="orange")
axs2[3].grid(True, alpha=0.4)

ax_prob_r = axs2[3].twinx()  # Create a twin Y-axis
ax_prob_r.plot(ts_est, est_data["prob_right"], color="red", linestyle=":", linewidth=2, label="Est Prob")
ax_prob_r.set_ylabel("Est Probability", color="red")
ax_prob_r.set_ylim(-0.1, 1.1)
ax_prob_r.tick_params(axis='y', labelcolor="red")
axs2[3].set_xlabel("Time (s)")

fig2.tight_layout()
plt.show()
