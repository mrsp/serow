import matplotlib.pyplot as plt
import numpy as np
import os
import json
import sys
from mcap.reader import make_reader
from scipy.spatial.transform import Rotation as R

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
DISPLAY_PLOTS = True
CONFIG_FILE = "test_config.json"
PERCENTAGE = 100.0

if len(sys.argv) > 1:
    try:
        PERCENTAGE = float(sys.argv[1])
    except ValueError:
        pass

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

# Resolve Paths
serow_path = os.environ.get("SEROW_PATH", "")
base_path = config["Paths"]["base_path"]
experiment_type = config["Target"]["experiment"] if "Target" in config else config["Experiment"]["type"]

raw_meas_path = config["Paths"]["data_file"].replace("{base_path}", base_path).replace("{type}", experiment_type)
raw_pred_path = config["Paths"]["prediction_file"].replace("{base_path}", base_path).replace("{type}", experiment_type)

def fix_extension(path):
    if path.endswith(".h5"): return path.replace(".h5", ".mcap")
    return path

MEASUREMENT_FILE = os.path.join(serow_path, fix_extension(raw_meas_path).lstrip("/"))
PREDICTION_FILE = os.path.join(serow_path, fix_extension(raw_pred_path).lstrip("/"))

# -------------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------------

def load_gt_data(mcap_file, percentage):
    data = {"pos": [], "rot": [], "lin_vel": [], "ts": [], "acc": [], "gyr": [], "f_lf": [], "f_rf": [], "f_lh": [], "f_rh": []}
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        messages = list(reader.iter_messages(topics=["/robot_state"]))
        limit = int(len(messages) * (percentage / 100.0))
        for schema, channel, message in messages[:limit]:
            d = json.loads(message.data)
            data["ts"].append(d["timestamp"])
            
            if "base_ground_truth" in d:
                gt = d["base_ground_truth"]
                data["pos"].append([gt["position"]["x"], gt["position"]["y"], gt["position"]["z"]])
                data["rot"].append([gt["orientation"]["w"], gt["orientation"]["x"], gt["orientation"]["y"], gt["orientation"]["z"]])
                if "linear_velocity" in gt:
                    data["lin_vel"].append([gt["linear_velocity"]["x"], gt["linear_velocity"]["y"], gt["linear_velocity"]["z"]])

            data["acc"].append([d["imu"]["linear_acceleration"]["x"], d["imu"]["linear_acceleration"]["y"], d["imu"]["linear_acceleration"]["z"]])
            data["gyr"].append([d["imu"]["angular_velocity"]["x"], d["imu"]["angular_velocity"]["y"], d["imu"]["angular_velocity"]["z"]])
            
            ff = d.get("feet_forces", {})
            data["f_lf"].append([ff.get("LF", {}).get("z", 0.0)])
            data["f_rf"].append([ff.get("RF", {}).get("z", 0.0)])
            data["f_lh"].append([ff.get("LH", {}).get("z", 0.0)])
            data["f_rh"].append([ff.get("RH", {}).get("z", 0.0)])
            
    return {k: np.array(v) for k, v in data.items() if len(v) > 0}

def load_serow_preds(mcap_file, percentage):
    data = {"pos": [], "rot": [], "lin_vel": [], "ts": [], "b_acc": [], "b_gyr": []}
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        messages = list(reader.iter_messages(topics=["serow_predictions"]))
        limit = int(len(messages) * (percentage / 100.0))
        for schema, channel, message in messages[:limit]:
            d = json.loads(message.data)
            data["ts"].append(d["timestamp"])
            bp = d["base_pose"]
            data["pos"].append([bp["position"]["x"], bp["position"]["y"], bp["position"]["z"]])
            data["rot"].append([bp["rotation"]["w"], bp["rotation"]["x"], bp["rotation"]["y"], bp["rotation"]["z"]])
            if "linear_velocity" in bp:
                data["lin_vel"].append([bp["linear_velocity"]["x"], bp["linear_velocity"]["y"], bp["linear_velocity"]["z"]])
            
            bias = d.get("imu_bias", {})
            data["b_acc"].append([bias.get("accel",{}).get("x",0), bias.get("accel",{}).get("y",0), bias.get("accel",{}).get("z",0)])
            data["b_gyr"].append([bias.get("angVel",{}).get("x",0), bias.get("angVel",{}).get("y",0), bias.get("angVel",{}).get("z",0)])
    return {k: np.array(v) for k, v in data.items() if len(v) > 0}

# -------------------------------------------------------------------------
# DEBUG & ALIGNMENT
# -------------------------------------------------------------------------

def analyze_timestamps(ts_array, name="Data"):
    """Scans timestamps to calculate refresh rate and identify drops/jumps."""
    print(f"\n--- Timestamp Analysis ({name}) ---")
    if len(ts_array) < 2:
        print("Not enough data to analyze.")
        return
        
    deltas = np.diff(ts_array)
    mean_dt = np.mean(deltas)
    hz = 1.0 / mean_dt if mean_dt > 0 else 0
    
    print(f"Total Frames:  {len(ts_array)}")
    print(f"Total Time:    {ts_array[-1] - ts_array[0]:.2f} seconds")
    print(f"Average Rate:  {hz:.2f} Hz (dt = {mean_dt:.6f}s)")
    print(f"Max gap:       {np.max(deltas):.6f}s")
    print(f"Min gap:       {np.min(deltas):.6f}s")
    
    # Check for backward time jumps
    negative_dts = np.where(deltas <= 0)[0]
    if len(negative_dts) > 0:
        print(f"[ERROR] Found {len(negative_dts)} frames where time went backwards or stood still!")
        for idx in negative_dts[:5]:
            print(f"  -> Index {idx}: ts1={ts_array[idx]:.6f}, ts2={ts_array[idx+1]:.6f}")

def align_trajectory_to_zero(pos_arr, rot_arr):
    p0 = pos_arr[0]
    q0 = R.from_quat([rot_arr[0,1], rot_arr[0,2], rot_arr[0,3], rot_arr[0,0]])
    q0_inv = q0.inv()
    pos_aligned = q0_inv.apply(pos_arr - p0)
    all_rot = R.from_quat(rot_arr[:, [1, 2, 3, 0]])
    rot_aligned = (q0_inv * all_rot).as_quat()[:, [3, 0, 1, 2]]
    return pos_aligned, rot_aligned

# -------------------------------------------------------------------------
# EXECUTION
# -------------------------------------------------------------------------

if __name__ == "__main__":
    gt = load_gt_data(MEASUREMENT_FILE, PERCENTAGE)
    est = load_serow_preds(PREDICTION_FILE, PERCENTAGE)
    
    # Run the debug analysis
    analyze_timestamps(gt["ts"], "Ground Truth")
    analyze_timestamps(est["ts"], "Estimator Output")

    # Shift timelines
    gt_ts_norm = gt["ts"] - gt["ts"][0]
    est_ts_norm = est["ts"] - est["ts"][0]

    # Align trajectories
    gt_pos_al, gt_rot_al = align_trajectory_to_zero(gt["pos"], gt["rot"])

    # Make arrays same size for direct comparison plotting (assuming est is subset of gt)
    timesteps = np.arange(0, est["pos"].shape[0])
    
    # RMSE Math
    gt_interp = np.column_stack([np.interp(est_ts_norm, gt_ts_norm, gt_pos_al[:, i]) for i in range(3)])
    rmse = np.sqrt(np.mean(np.linalg.norm(gt_interp - est["pos"], axis=1)**2))

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    if DISPLAY_PLOTS:
        # Figure 1: Position
        fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig1.suptitle(f"Base Position (RMSE: {rmse:.4f} m)")
        labels = ["X", "Y", "Z"]
        for i in range(3):
            axs1[i].plot(gt_ts_norm[:len(timesteps)], gt_pos_al[:len(timesteps), i], label="GT", color="blue", lw=2)
            axs1[i].plot(est_ts_norm, est["pos"][:, i], label="Est", color="orange", ls="--", lw=2)
            axs1[i].set_ylabel(f"Pos {labels[i]} (m)")
            axs1[i].grid(True, alpha=0.3)
        axs1[0].legend(loc="upper left")

        # Figure 2: Orientation
        fig2, axs2 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        fig2.suptitle("Base Orientation")
        q_labels = ["W", "X", "Y", "Z"]
        for i in range(4):
            axs2[i].plot(gt_ts_norm[:len(timesteps)], gt_rot_al[:len(timesteps), i], label="GT", color="blue")
            axs2[i].plot(est_ts_norm, est["rot"][:, i], label="Est", color="orange", ls="--")
            axs2[i].set_ylabel(f"Quat {q_labels[i]}")
            axs2[i].grid(True, alpha=0.3)

        # Figure 3: Linear Velocity
        if "lin_vel" in gt and "lin_vel" in est:
            fig3, axs3 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig3.suptitle("Linear Velocity")
            q0_gt = R.from_quat([gt["rot"][0,1], gt["rot"][0,2], gt["rot"][0,3], gt["rot"][0,0]]).inv()
            gt_vel_aligned = q0_gt.apply(gt["lin_vel"])
            
            for i in range(3):
                axs3[i].plot(gt_ts_norm[:len(timesteps)], gt_vel_aligned[:len(timesteps), i], label="GT", color="blue")
                axs3[i].plot(est_ts_norm, est["lin_vel"][:, i], label="Est", color="orange", ls="--")
                axs3[i].set_ylabel(f"Vel {labels[i]} (m/s)")
                axs3[i].grid(True, alpha=0.3)

        # Figure 4: Biases
        if "b_acc" in est:
            fig4, axs4 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            fig4.suptitle("Estimated IMU Biases")
            axs4[0].plot(est_ts_norm, est["b_acc"], label=["ax", "ay", "az"])
            axs4[0].set_ylabel("Accel Bias (m/s^2)")
            axs4[1].plot(est_ts_norm, est["b_gyr"], label=["wx", "wy", "wz"])
            axs4[1].set_ylabel("Gyro Bias (rad/s)")
            for ax in axs4: ax.legend(); ax.grid(True)

        # Figure 5: Vertical Forces
        fig5, axs5 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig5.suptitle("Feet Forces Z (GT)")
        keys = ["f_lf", "f_rf", "f_lh", "f_rh"]
        legs = ["LF", "RF", "LH", "RH"]
        for i in range(4):
            axs5[i].plot(gt_ts_norm[:len(timesteps)], gt[keys[i]][:len(timesteps)], color="blue", label=legs[i])
            axs5[i].set_ylabel("Force Z")
            axs5[i].grid(True)
            axs5[i].legend()

        # Figure 6: Raw IMU
        fig6, axs6 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig6.suptitle("Raw IMU Measurements")
        axs6[0].plot(gt_ts_norm[:len(timesteps)], gt["acc"][:len(timesteps)], label=["X","Y","Z"], alpha=0.7)
        axs6[0].set_ylabel("Linear Acc (m/s²)")
        axs6[1].plot(gt_ts_norm[:len(timesteps)], gt["gyr"][:len(timesteps)], label=["X","Y","Z"], alpha=0.7)
        axs6[1].set_ylabel("Angular Vel (rad/s)")
        for ax in axs6: ax.legend(); ax.grid(True, alpha=0.3)



        # --- Figure 7: Timestep Deltas (Refresh Rate Check) ---
        fig7, axs7 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig7.suptitle("Timestep Differences (\u0394t) - Refresh Rate Stability")

        # Calculate the differences between consecutive timestamps
        gt_dt = np.diff(gt["ts"])
        est_dt = np.diff(est["ts"])

        # Ground Truth Timesteps
        axs7[0].plot(gt_ts_norm[1:], gt_dt, color="blue", label="GT \u0394t", alpha=0.8)
        axs7[0].axhline(np.median(gt_dt), color="red", linestyle="--", label=f"Median: {np.median(gt_dt):.5f}s")
        axs7[0].set_ylabel("\u0394t (seconds)")
        axs7[0].set_title("Ground Truth (/robot_state)")
        axs7[0].grid(True, alpha=0.3)
        axs7[0].legend(loc="upper right")

        # Estimator Timesteps
        axs7[1].plot(est_ts_norm[1:], est_dt, color="orange", label="Est \u0394t", alpha=0.8)
        axs7[1].axhline(np.median(est_dt), color="red", linestyle="--", label=f"Median: {np.median(est_dt):.5f}s")
        axs7[1].set_ylabel("\u0394t (seconds)")
        axs7[1].set_title("Estimator (serow_predictions)")
        axs7[1].set_xlabel("Time (s)")
        axs7[1].grid(True, alpha=0.3)
        axs7[1].legend(loc="upper right")

        fig7.tight_layout()
        
        # Finally, show all plots
        plt.show()
