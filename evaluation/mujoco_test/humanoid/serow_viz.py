import matplotlib.pyplot as plt
import numpy as np
import os
import json
from mcap.reader import make_reader
from scipy.spatial.transform import Rotation as R

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy <2.0 compatibility

# ---------------------------------------------------------------------------
# Frame convention
# ---------------------------------------------------------------------------
# Serow starts at position (0,0,0) and orientation (0,0,0,1). Its world frame is
# therefore NOT the simulator world frame
# ALIGNMENT = "yaw"  : correct. Removes initial position + initial yaw only.
#                      Ground-truth roll/pitch and z stay physical, so a 5 deg
#                      slope reads as 5 deg and any levelling error in Serow's
#                      initial attitude shows up as a constant roll/pitch offset.
# ALIGNMENT = "full" : the old behaviour. Rotates GT by the full initial
#                      orientation, which silently tilts the GT trajectory by the
#                      initial roll/pitch and rotates part of the terrain slope
#                      out of the z axis. Kept for comparison only.
# ALIGNMENT = "none" : raw simulator frame (only useful if Serow is initialised
#                      from the true initial pose).
ALIGNMENT = "yaw"

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
print(f"[CONFIG] Alignment:  {ALIGNMENT}")

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
                fl = np.array(get_f("FL")); rl = np.array(get_f("RL"))
                fr = np.array(get_f("FR")); rr = np.array(get_f("RR"))
                data_store["f_left"].append(fl + rl)
                data_store["f_right"].append(fr + rr)

    return {k: np.array(v) for k, v in data_store.items()}

def load_est(mcap_file, topic):
    data_store = {
        "ts": [], "pos": [], "rot": [], "lin_vel": [],
        "prob_left": [], "prob_right": [],
        "bias_acc": [], "bias_gyr": []
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

            probs = d.get("contact_probabilities", {})
            p_left = 0.0
            p_right = 0.0
            if probs:
                l_vals = [v for k, v in probs.items() if 'left' in k.lower() or 'fl' in k.lower() or 'rl' in k.lower()]
                r_vals = [v for k, v in probs.items() if 'right' in k.lower() or 'fr' in k.lower() or 'rr' in k.lower()]
                if l_vals: p_left = max(l_vals)
                if r_vals: p_right = max(r_vals)
            data_store["prob_left"].append(p_left)
            data_store["prob_right"].append(p_right)

            if "imu_bias" in d:
                acc_b = d["imu_bias"].get("accel", {"x": 0.0, "y": 0.0, "z": 0.0})
                gyr_b = d["imu_bias"].get("angVel", {"x": 0.0, "y": 0.0, "z": 0.0})
                data_store["bias_acc"].append([acc_b["x"], acc_b["y"], acc_b["z"]])
                data_store["bias_gyr"].append([gyr_b["x"], gyr_b["y"], gyr_b["z"]])
            else:
                data_store["bias_acc"].append([0.0, 0.0, 0.0])
                data_store["bias_gyr"].append([0.0, 0.0, 0.0])

    return {k: np.array(v) for k, v in data_store.items()}

# ---------------------------------------------------------------------------
# Alignment: map ground truth into Serow's frame (starts at 0 / identity)
# ---------------------------------------------------------------------------
def alignment_rotation(gt_rot, mode):
    """Rotation that takes simulator-world vectors into Serow's world frame."""
    q0 = gt_rot[0]                                  # stored as (w,x,y,z)
    r0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])
    if mode == "full":
        return r0.inv()
    if mode == "yaw":
        yaw0 = r0.as_euler("ZYX")[0]                # intrinsic Z-Y-X -> [yaw, pitch, roll]
        return R.from_euler("Z", -yaw0)
    if mode == "none":
        return R.identity()
    raise ValueError(f"unknown ALIGNMENT '{mode}'")

def align_gt(gt_pos, gt_rot, gt_vel, mode):
    if len(gt_pos) == 0:
        return gt_pos, gt_rot, gt_vel
    R_align = alignment_rotation(gt_rot, mode)
    pos = R_align.apply(gt_pos - gt_pos[0])          # Serow starts at (0,0,0)
    rot = (R_align * R.from_quat(gt_rot[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    vel = R_align.apply(gt_vel)
    return pos, rot, vel

def compute_ATE(gt_ts, gt_pos, est_ts, est_pos, per_axis=False):
    if len(gt_ts) == 0 or len(est_ts) == 0:
        return (0.0, np.zeros(3)) if per_axis else 0.0

    t_start = max(gt_ts[0], est_ts[0])
    t_end = min(gt_ts[-1], est_ts[-1])

    idx_est = np.where((est_ts >= t_start) & (est_ts <= t_end))
    eval_ts = est_ts[idx_est]
    eval_est = est_pos[idx_est]

    if len(eval_ts) == 0:
        return (0.0, np.zeros(3)) if per_axis else 0.0

    eval_gt = np.column_stack([np.interp(eval_ts, gt_ts, gt_pos[:, i]) for i in range(3)])
    err = eval_gt - eval_est
    ate = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    if per_axis:
        return ate, np.sqrt(np.mean(err**2, axis=0))
    return ate

gt_data = load_gt(MEASUREMENT_FILE, gt_topic_name)
est_data = load_est(PREDICTION_FILE, est_topic_name)

if not gt_data or len(gt_data["ts"]) == 0:
    print("FATAL: Ground Truth data is empty."); exit(1)
if not est_data or len(est_data["ts"]) == 0:
    print("FATAL: Prediction data is empty."); exit(1)

gt_pos_aligned, gt_rot_aligned, gt_vel_aligned = align_gt(
    gt_data["pos"], gt_data["rot"], gt_data["lin_vel"], ALIGNMENT)

ate, ate_axis = compute_ATE(gt_data["ts"], gt_pos_aligned, est_data["ts"], est_data["pos"], per_axis=True)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
q0 = gt_data["rot"][0]
r0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])
rpy0 = r0.as_euler("xyz", degrees=True)

print("\n" + "=" * 62)
print(f" ATE (Position RMSE), alignment='{ALIGNMENT}' : {ate:.4f} m")
print(f"   per axis  x {ate_axis[0]:.4f}   y {ate_axis[1]:.4f}   z {ate_axis[2]:.4f}   m")
print("=" * 62)

print(f"\n initial GT base attitude: roll {rpy0[0]:+.2f}  pitch {rpy0[1]:+.2f}  yaw {rpy0[2]:+.2f}  deg")
print(" Serow starts level (identity), so the roll/pitch above is a levelling")
print(" error it never sees. With alignment='full' it is hidden; with 'yaw' it")
print(" appears as a constant offset in the roll/pitch panels. That is intended.")

print("\n ATE under each alignment (same data, different frame convention):")
for mode in ("yaw", "full", "none"):
    p_m, _, _ = align_gt(gt_data["pos"], gt_data["rot"], gt_data["lin_vel"], mode)
    a_m, ax_m = compute_ATE(gt_data["ts"], p_m, est_data["ts"], est_data["pos"], per_axis=True)
    tag = "  <-- in use" if mode == ALIGNMENT else ""
    print(f"   {mode:5s} : {a_m:.4f} m   (x {ax_m[0]:.4f}  y {ax_m[1]:.4f}  z {ax_m[2]:.4f}){tag}")

# final drift and path inclination in the aligned frame
d_gt = gt_pos_aligned[-1] - gt_pos_aligned[0]
d_es = est_data["pos"][-1] - est_data["pos"][0]
print("\n final displacement            x        y        z")
print(f"   ground truth            {d_gt[0]:+8.3f} {d_gt[1]:+8.3f} {d_gt[2]:+8.3f}")
print(f"   estimated               {d_es[0]:+8.3f} {d_es[1]:+8.3f} {d_es[2]:+8.3f}")
print(f"   error                   {d_es[0]-d_gt[0]:+8.3f} {d_es[1]-d_gt[1]:+8.3f} {d_es[2]-d_gt[2]:+8.3f}")
hz_gt = np.linalg.norm(d_gt[:2]); hz_es = np.linalg.norm(d_es[:2])
if hz_gt > 1e-6 and hz_es > 1e-6:
    print(f"\n path inclination : GT {np.degrees(np.arctan2(d_gt[2], hz_gt)):+.2f} deg"
          f"   est {np.degrees(np.arctan2(d_es[2], hz_es)):+.2f} deg")
    print(f" horizontal distance : GT {hz_gt:.3f} m   est {hz_es:.3f} m   ({100*hz_es/hz_gt:.1f} %)")

# is the reported velocity consistent with the reported position?
tv = est_data["ts"]; ve = est_data["lin_vel"]; pe = est_data["pos"]
print("\n reported velocity vs reported position (integral of one against the other):")
for i, a in enumerate("xyz"):
    iv = _trapz(ve[:, i], tv); dp = pe[-1, i] - pe[0, i]
    ratio = iv / dp if abs(dp) > 1e-6 else float("nan")
    print(f"   {a}   integral(v) {iv:+8.3f} m   delta(p) {dp:+8.3f} m   ratio {ratio:7.3f}")
print("   (in the LI-EKF position and velocity are separate states and the velocity")
print("    update corrects both, so these need not match exactly - but a large gap")
print("    means the panels are telling you different stories.)\n")

ts_gt = gt_data["ts"]
ts_est = est_data["ts"]

r_gt = R.from_quat(gt_rot_aligned[:, [1, 2, 3, 0]])
r_est = R.from_quat(est_data["rot"][:, [1, 2, 3, 0]])

rpy_gt = np.unwrap(r_gt.as_euler('xyz', degrees=True), period=360, axis=0)
rpy_est = np.unwrap(r_est.as_euler('xyz', degrees=True), period=360, axis=0)

# ---------------------------------------------------------
# Figure 1: Position, Velocity, and RPY Orientation
# ---------------------------------------------------------
fig1, axs1 = plt.subplots(3, 3, figsize=(18, 10), sharex=True)
fig1.suptitle(f"Serow State Estimation - {robot_name}   [GT aligned: {ALIGNMENT}]\n"
              f"ATE: {ate:.4f} m   (x {ate_axis[0]:.3f}  y {ate_axis[1]:.3f}  z {ate_axis[2]:.3f})")

labels_pos = ["X", "Y", "Z"]
for i in range(3):
    axs1[i, 0].plot(ts_gt, gt_pos_aligned[:, i], label="GT", color="black", alpha=0.6)
    axs1[i, 0].plot(ts_est, est_data["pos"][:, i], label="Estimated", color="red", linestyle="--")
    axs1[i, 0].set_ylabel(f"Pos {labels_pos[i]} (m)")
    axs1[i, 0].grid(True, alpha=0.3)
axs1[0, 0].legend()
axs1[0, 0].set_title("Base Position")

for i in range(3):
    axs1[i, 1].plot(ts_gt, gt_vel_aligned[:, i], label="GT", color="black", alpha=0.6)
    axs1[i, 1].plot(ts_est, est_data["lin_vel"][:, i], label="Estimated", color="red", linestyle="--")
    axs1[i, 1].set_ylabel(f"Vel {labels_pos[i]} (m/s)")
    axs1[i, 1].grid(True, alpha=0.3)
axs1[0, 1].set_title("Base Velocity")

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

for k, lbl in enumerate("XYZ"):
    axs2[0].plot(ts_gt, gt_data["acc"][:, k], label=f"Acc {lbl}")
axs2[0].set_ylabel("Accel (m/s²)"); axs2[0].grid(True, alpha=0.4); axs2[0].legend(loc="upper right")

for k, lbl in enumerate("XYZ"):
    axs2[1].plot(ts_gt, gt_data["gyr"][:, k], label=f"Gyr {lbl}")
axs2[1].set_ylabel("Gyro (rad/s)"); axs2[1].grid(True, alpha=0.4); axs2[1].legend(loc="upper right")

axs2[2].plot(ts_gt, gt_data["f_left"][:, 2], color="blue", alpha=0.5, label="GT Force Z")
axs2[2].set_ylabel("Left Force Z (N)", color="blue")
axs2[2].tick_params(axis='y', labelcolor="blue"); axs2[2].grid(True, alpha=0.4)
ax_prob_l = axs2[2].twinx()
ax_prob_l.plot(ts_est, est_data["prob_left"], color="red", linestyle=":", linewidth=2, label="Est Prob")
ax_prob_l.set_ylabel("Est Probability", color="red"); ax_prob_l.set_ylim(-0.1, 1.1)
ax_prob_l.tick_params(axis='y', labelcolor="red")

axs2[3].plot(ts_gt, gt_data["f_right"][:, 2], color="orange", alpha=0.5, label="GT Force Z")
axs2[3].set_ylabel("Right Force Z (N)", color="orange")
axs2[3].tick_params(axis='y', labelcolor="orange"); axs2[3].grid(True, alpha=0.4)
ax_prob_r = axs2[3].twinx()
ax_prob_r.plot(ts_est, est_data["prob_right"], color="red", linestyle=":", linewidth=2, label="Est Prob")
ax_prob_r.set_ylabel("Est Probability", color="red"); ax_prob_r.set_ylim(-0.1, 1.1)
ax_prob_r.tick_params(axis='y', labelcolor="red")
axs2[3].set_xlabel("Time (s)")

fig2.tight_layout()

# ---------------------------------------------------------
# Figure 3: Estimated IMU Biases
# ---------------------------------------------------------
fig3, axs3 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig3.suptitle("Estimated IMU Biases over Time")

for k, lbl in enumerate("XYZ"):
    axs3[0].plot(ts_est, est_data["bias_acc"][:, k], label=f"Bias Acc {lbl}")
axs3[0].set_ylabel("Accel Bias (m/s²)"); axs3[0].grid(True, alpha=0.4); axs3[0].legend(loc="upper right")

for k, lbl in enumerate("XYZ"):
    axs3[1].plot(ts_est, est_data["bias_gyr"][:, k], label=f"Bias Gyr {lbl}")
axs3[1].set_ylabel("Gyro Bias (rad/s)"); axs3[1].set_xlabel("Time (s)")
axs3[1].grid(True, alpha=0.4); axs3[1].legend(loc="upper right")

fig3.tight_layout()

plt.show()
