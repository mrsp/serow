import matplotlib.pyplot as plt
import numpy as np
import os
import json
from mcap.reader import make_reader
from scipy.spatial.transform import Rotation as R, Slerp

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
# ALIGNMENT = "full" : rotates the GROUND TRUTH ONLY by its full initial
#                      orientation. This tilts the GT trajectory by the initial
#                      roll/pitch and rotates part of the terrain slope out of
#                      the z axis, so it silently changes the z error it is
#                      supposed to measure. Kept for comparison only, never for
#                      the paper. Use ZERO_INITIAL_ATTITUDE below instead.
# ALIGNMENT = "none" : raw simulator frame (only useful if Serow is initialised
#                      from the true initial pose).
ALIGNMENT = "yaw"

# ZERO_INITIAL_ATTITUDE rotates GROUND TRUTH AND ESTIMATE BY THE SAME rotation,
# the inverse of the ground-truth initial roll/pitch, so every curve starts at
# identity orientation and (0,0,0) position. Because the rotation is common to
# both trajectories the error between them, and hence the ATE, is unchanged -
# this is a change of world frame, not an alignment of the estimate to the
# ground truth, and it cannot hide an initialisation error in Serow.
#
# The cost is that the plotted z axis is no longer gravity. A tilt of a deg
# leaks sin(a) of the travelled horizontal distance into z: 0.88 deg over 8 m is
# 0.12 m, which is the same order as the vertical drift being reported. Leave
# this False for the elevation results and fix the initial attitude in Serow
# instead, the initialisation check printed below says whether it is fixed.
ZERO_INITIAL_ATTITUDE = True

# The static window used for the initial attitude: leading samples whose
# ground-truth speed is below STATIC_SPEED_EPS, capped at STATIC_MAX_SECONDS.
# Averaging over it instead of taking sample 0 matches what Serow's
# levelBaseAttitude does, so the two numbers are directly comparable.
STATIC_SPEED_EPS = 1e-3
STATIC_MAX_SECONDS = 5.0
STATIC_MIN_SAMPLES = 50

CONFIG_FILE = "test_config.json"

# ---------------------------------------------------------------------------
# Paper figures
# ---------------------------------------------------------------------------
# SAVE_PAPER_FIGURES writes <FIGURE_DIR>/<prefix>_{position,linear_velocity,
# orientation}.pdf with the style used in the paper. SHOW_DIAGNOSTICS opens the
# raw IMU / force / bias panels on screen, those are not paper figures.
SAVE_PAPER_FIGURES = True
SHOW_DIAGNOSTICS = True

# Table I of the paper (terrain-correction ablation). PRINT_PAPER_TABLE prints
# the On / Off / Improvement block for this experiment to the terminal, and
# SAVE_PAPER_TABLE additionally writes <FIGURE_DIR>/<prefix>_table.tex with the
# same numbers as the LaTeX rows of the table (better value in bold).
PRINT_PAPER_TABLE = True
SAVE_PAPER_TABLE = True

# PLOT_FROM_ESTIMATOR_START puts t = 0 at the first published estimate and crops
# the leading ground-truth-only stretch, so both curves start together. Serow
# publishes nothing until levelBaseAttitude has filled its window and the IMU
# biases are calibrated, which is attitude_leveling_samples +
# max_imu_calibration_cycles cycles. The ground truth is zeroed at the same
# instant. The ATE is unaffected either way, it is already computed over the
# overlap of the two time ranges.
PLOT_FROM_ESTIMATOR_START = True

# Terrain ablation. With AUTO_TERRAIN_ABLATION the script looks next to the
# configured prediction file for the pair
#     <stem>_with_terrain.mcap  and  <stem>_without_terrain.mcap
# and when both are there it plots them together: the terrain run in the
# estimate style, the no-terrain run in the ablation style, with the labels and
# the figure prefix below. Either file on its own is ignored and the configured
# prediction file is used as usual. Point the config at either member of the
# pair, or at the bare <stem>.mcap, the suffixes are stripped before searching.
AUTO_TERRAIN_ABLATION = True
WITH_TERRAIN_SUFFIX = "_with_terrain"
WITHOUT_TERRAIN_SUFFIX = "_without_terrain"

# Second estimate, drawn in the ablation style. Setting it by hand overrides the
# search above, leave it None for a single-estimate figure when no pair is found.
ABLATION_FILE = None

# Legend entries and figure stem. None picks the default for whichever case the
# run turns out to be, single estimate or terrain ablation.
GT_LABEL = "Ground truth"
EST_LABEL = None               # None -> "SEROW", or "With terrain" for the pair
ABLATION_LABEL = None          # None -> "Without terrain"
FIGURE_PREFIX = None           # None -> "<robot>_<experiment>", or "terrain_ablation"

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

def find_terrain_pair(prediction_file):
    """The *_with_terrain / *_without_terrain mcaps next to prediction_file."""
    directory, name = os.path.split(prediction_file)
    for suffix in (WITH_TERRAIN_SUFFIX, WITHOUT_TERRAIN_SUFFIX):
        name = name.replace(suffix, "")
    stem, ext = os.path.splitext(name)
    with_terrain = os.path.join(directory, f"{stem}{WITH_TERRAIN_SUFFIX}{ext}")
    without_terrain = os.path.join(directory, f"{stem}{WITHOUT_TERRAIN_SUFFIX}{ext}")
    if os.path.exists(with_terrain) and os.path.exists(without_terrain):
        return with_terrain, without_terrain
    return None, None

MEASUREMENT_FILE = get_absolute_path(resolve_template(config["Paths"]["data_file"]))
PREDICTION_FILE = get_absolute_path(resolve_template(config["Paths"]["prediction_file"]))

if AUTO_TERRAIN_ABLATION and ABLATION_FILE is None:
    # The configured prediction file need not exist, it is often just the stem
    # of the pair, so search every directory it could resolve to plus the one
    # holding the ground truth.
    raw_prediction = resolve_template(config["Paths"]["prediction_file"])
    if os.path.isabs(raw_prediction):
        candidates = [raw_prediction]
    else:
        project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
        candidates = [os.path.join(project_root, raw_prediction),
                      os.path.join(script_dir, raw_prediction)]
    candidates.append(os.path.join(os.path.dirname(MEASUREMENT_FILE),
                                   os.path.basename(raw_prediction)))
    for candidate in candidates:
        with_terrain, without_terrain = find_terrain_pair(candidate)
        if with_terrain:
            PREDICTION_FILE = with_terrain
            ABLATION_FILE = without_terrain
            break

is_ablation = ABLATION_FILE is not None
EST_LABEL = EST_LABEL or ("With terrain" if is_ablation else "SEROW")
ABLATION_LABEL = ABLATION_LABEL or "Without terrain"
FIGURE_DIR = resolve_template(config["Paths"].get("figure_dir", "figures"))
if not os.path.isabs(FIGURE_DIR):
    FIGURE_DIR = os.path.join(script_dir, FIGURE_DIR)
FIGURE_PREFIX = FIGURE_PREFIX or ("terrain_ablation" if is_ablation else full_exp_name)

print(f"[CONFIG] Robot:      {robot_name}")
print(f"[CONFIG] Experiment: {exp_name}")
print(f"[CONFIG] GT File:    {MEASUREMENT_FILE}")
print(f"[CONFIG] Est File:   {PREDICTION_FILE}   ({EST_LABEL})")
if is_ablation:
    print(f"[CONFIG] Abl File:   {ABLATION_FILE}   ({ABLATION_LABEL})")
print(f"[CONFIG] Alignment:  {ALIGNMENT}   zero initial attitude: {ZERO_INITIAL_ATTITUDE}")
print(f"[CONFIG] Figures:    {FIGURE_DIR}   prefix {FIGURE_PREFIX}")

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
def static_window(ts, vel):
    """Leading samples over which the robot is standing still."""
    speed = np.linalg.norm(vel, axis=1)
    moving = np.flatnonzero(speed > STATIC_SPEED_EPS)
    n = int(moving[0]) if moving.size else len(speed)
    n = min(n, int(np.searchsorted(ts, ts[0] + STATIC_MAX_SECONDS)))
    return max(n, min(STATIC_MIN_SAMPLES, len(speed)))

def initial_attitude(rot, n):
    """Mean attitude over the first n samples, quaternions stored as (w,x,y,z)."""
    return R.from_quat(rot[:max(n, 1), [1, 2, 3, 0]]).mean()

def alignment_rotation(gt_rot, n_static, mode):
    """Rotation that takes simulator-world vectors into Serow's world frame."""
    r0 = initial_attitude(gt_rot, n_static)
    if mode == "full":
        return r0.inv()
    if mode == "yaw":
        yaw0 = r0.as_euler("ZYX")[0]                # intrinsic Z-Y-X -> [yaw, pitch, roll]
        return R.from_euler("Z", -yaw0)
    if mode == "none":
        return R.identity()
    raise ValueError(f"unknown ALIGNMENT '{mode}'")

def rotate_trajectory(R_align, pos, rot, vel, origin=None):
    """Apply a rigid rotation to a trajectory, optionally re-zeroing the origin."""
    if len(pos) == 0:
        return pos, rot, vel
    p = R_align.apply(pos if origin is None else pos - origin)
    q = (R_align * R.from_quat(rot[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    v = R_align.apply(vel)
    return p, q, v

def align_gt(gt_pos, gt_rot, gt_vel, n_static, mode, origin_index=0):
    if len(gt_pos) == 0:
        return gt_pos, gt_rot, gt_vel
    R_align = alignment_rotation(gt_rot, n_static, mode)
    return rotate_trajectory(R_align, gt_pos, gt_rot, gt_vel, origin=gt_pos[origin_index])

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
abl_data = load_est(ABLATION_FILE, est_topic_name) if ABLATION_FILE else None

if not gt_data or len(gt_data["ts"]) == 0:
    print("FATAL: Ground Truth data is empty."); exit(1)
if not est_data or len(est_data["ts"]) == 0:
    print("FATAL: Prediction data is empty."); exit(1)

n_static = static_window(gt_data["ts"], gt_data["lin_vel"])

# Serow's origin is its own first published sample, not the first logged one: it
# publishes nothing while levelBaseAttitude fills its window and the IMU biases
# are calibrated. Zero the ground truth at the same instant.
origin_index = min(int(np.searchsorted(gt_data["ts"], est_data["ts"][0])),
                   len(gt_data["pos"]) - 1)

gt_pos_aligned, gt_rot_aligned, gt_vel_aligned = align_gt(
    gt_data["pos"], gt_data["rot"], gt_data["lin_vel"], n_static, ALIGNMENT, origin_index)

est_pos, est_rot, est_vel = est_data["pos"], est_data["rot"], est_data["lin_vel"]
abl_pos, abl_rot, abl_vel = ((abl_data["pos"], abl_data["rot"], abl_data["lin_vel"])
                             if abl_data is not None else (None, None, None))

ate_before = compute_ATE(gt_data["ts"], gt_pos_aligned, est_data["ts"], est_pos)

# Table I is computed in this frame, before any ZERO_INITIAL_ATTITUDE rotation:
# the paper removes the initial position and yaw only, so that z stays gravity
# and the vertical errors are physical. RMSE_3D and RMSE_R are invariant to the
# common rotation below, RMSE_z and the mean vertical errors are not.
_tbl_gt = (gt_data["ts"], gt_pos_aligned.copy(), gt_rot_aligned.copy(), gt_vel_aligned.copy())
_tbl_est = (est_data["ts"], est_pos.copy(), est_rot.copy(), est_vel.copy())
_tbl_abl = ((abl_data["ts"], abl_pos.copy(), abl_rot.copy(), abl_vel.copy())
            if abl_data is not None and len(abl_data["ts"]) > 0 else None)

# ---------------------------------------------------------------------------
# Optional: rotate every trajectory so they all start at identity orientation
# ---------------------------------------------------------------------------
if ZERO_INITIAL_ATTITUDE:
    R_zero = initial_attitude(gt_rot_aligned, n_static).inv()
    gt_pos_aligned, gt_rot_aligned, gt_vel_aligned = rotate_trajectory(
        R_zero, gt_pos_aligned, gt_rot_aligned, gt_vel_aligned)
    est_pos, est_rot, est_vel = rotate_trajectory(R_zero, est_pos, est_rot, est_vel)
    if abl_data is not None:
        abl_pos, abl_rot, abl_vel = rotate_trajectory(R_zero, abl_pos, abl_rot, abl_vel)
    tilt = np.degrees(R_zero.magnitude())
    print(f"\n[FRAME] every trajectory rotated by {tilt:.2f} deg so they start at identity.")
    print(f"[FRAME] the plotted z is tilted by {tilt:.2f} deg from gravity, which leaks")
    print(f"[FRAME] {np.sin(np.radians(tilt)):.4f} m of z per metre travelled horizontally.")

ate, ate_axis = compute_ATE(gt_data["ts"], gt_pos_aligned, est_data["ts"], est_pos, per_axis=True)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print(f" ATE (Position RMSE), alignment='{ALIGNMENT}' : {ate:.4f} m")
print(f"   {EST_LABEL:<22s} {ate:.4f} m   "
      f"(x {ate_axis[0]:.4f}  y {ate_axis[1]:.4f}  z {ate_axis[2]:.4f})")
if abl_data is not None and len(abl_data["ts"]) > 0:
    ate_abl, ate_abl_axis = compute_ATE(gt_data["ts"], gt_pos_aligned, abl_data["ts"], abl_pos,
                                        per_axis=True)
    print(f"   {ABLATION_LABEL:<22s} {ate_abl:.4f} m   "
          f"(x {ate_abl_axis[0]:.4f}  y {ate_abl_axis[1]:.4f}  z {ate_abl_axis[2]:.4f})")
print("=" * 62)
if ZERO_INITIAL_ATTITUDE:
    print(f" ATE before the common rotation : {ate_before:.4f} m   (must be identical)")

# ---------------------------------------------------------------------------
# Paper Table I: terrain-correction ablation metrics, Eqs. (87), (88), (91)
# ---------------------------------------------------------------------------
# Protocol of Section VII-A: evaluate over the time interval common to the
# ground truth and BOTH estimator runs, with the ground truth interpolated at
# the estimator timestamps (positions and velocities linearly, orientations by
# slerp). No trajectory alignment beyond the initial position + yaw removal.
#
#   RMSE_3D = sqrt(mean ||x_hat - x||^2)                           (87)
#   RMSE_z  = sqrt(mean e_i^2),  e_bar_z = mean e_i,  e_i = e3^T (x_hat - x)   (88)
#   RMSE_R  = 180/pi sqrt(mean theta_i^2),  theta_i = angle(R^T R_hat)        (91)
#
# The improvement column is (Off - On) / Off; for the signed e_bar_z it is
# computed on magnitudes, as stated under Eq. (88). RMSE_xy is reported as well
# because the terrain update acts on z only, so it is the cleanest check that
# the horizontal estimate is untouched.
def _interp_gt(gt_ts, gt_pos, gt_rot, gt_vel, eval_ts):
    """Ground truth sampled at eval_ts: linear for pos/vel, slerp for rotation."""
    t, keep = np.unique(gt_ts, return_index=True)       # Slerp needs strictly increasing t
    pos = np.column_stack([np.interp(eval_ts, t, gt_pos[keep, i]) for i in range(3)])
    vel = np.column_stack([np.interp(eval_ts, t, gt_vel[keep, i]) for i in range(3)])
    rot = Slerp(t, R.from_quat(gt_rot[keep][:, [1, 2, 3, 0]]))(eval_ts)
    return pos, rot, vel

def table_metrics(gt, est, t_start, t_end):
    """Table I metrics of one run over [t_start, t_end]. gt/est are (ts, pos, rot, vel)."""
    gt_ts, gt_pos, gt_rot, gt_vel = gt
    est_ts, est_pos, est_rot, est_vel = est
    keep = (est_ts >= t_start) & (est_ts <= t_end)
    if keep.sum() < 2:
        return None
    eval_ts = est_ts[keep]
    g_pos, g_rot, g_vel = _interp_gt(gt_ts, gt_pos, gt_rot, gt_vel, eval_ts)
    e_rot = R.from_quat(est_rot[keep][:, [1, 2, 3, 0]])

    ep = est_pos[keep] - g_pos                           # x_hat - x, as in Eq. (88)
    ev = est_vel[keep] - g_vel
    theta = (g_rot.inv() * e_rot).magnitude()            # geodesic angle, Eq. (90)

    return {
        "n": int(keep.sum()),
        "t_start": float(eval_ts[0]), "t_end": float(eval_ts[-1]),
        "pos_rmse_3d": float(np.sqrt(np.mean(np.sum(ep**2, axis=1)))),
        "pos_rmse_xy": float(np.sqrt(np.mean(np.sum(ep[:, :2]**2, axis=1)))),
        "pos_rmse_z": float(np.sqrt(np.mean(ep[:, 2]**2))),
        "pos_mean_z": float(np.mean(ep[:, 2])),
        "pos_final_z": float(ep[-1, 2]),
        "vel_rmse_3d": float(np.sqrt(np.mean(np.sum(ev**2, axis=1)))),
        "vel_rmse_z": float(np.sqrt(np.mean(ev[:, 2]**2))),
        "vel_mean_z": float(np.mean(ev[:, 2])),
        "rot_rmse_deg": float(np.degrees(np.sqrt(np.mean(theta**2)))),
    }

# (key, label, unit, decimals, signed)   signed -> improvement on magnitudes
TABLE_ROWS = [
    ("pos_rmse_3d", "Position RMSE_3D",            "m",   4, False),
    ("pos_rmse_xy", "Position RMSE_xy",            "m",   4, False),
    ("pos_rmse_z",  "Position RMSE_z",             "m",   4, False),
    ("pos_mean_z",  "Mean vertical position error", "m",  4, True),
    ("pos_final_z", "Final vertical position error","m",  4, True),
    ("vel_rmse_3d", "Velocity RMSE_3D",            "m/s", 4, False),
    ("vel_rmse_z",  "Velocity RMSE_z",             "m/s", 4, False),
    ("vel_mean_z",  "Mean vertical velocity error", "m/s", 4, True),
    ("rot_rmse_deg","Orientation RMSE_R",          "deg", 4, False),
]

def _improvement(on, off, signed):
    a, b = (abs(on), abs(off)) if signed else (on, off)
    return float("nan") if abs(b) < 1e-12 else 100.0 * (b - a) / b

def print_paper_table(m_on, m_off, on_label, off_label):
    print("\n Terrain ablation metrics (common interval, no trajectory alignment)")
    print(f"   interval [{m_on['t_start']:.2f}, {m_on['t_end']:.2f}] s   "
          f"samples: {on_label} {m_on['n']}" +
          (f", {off_label} {m_off['n']}" if m_off else ""))
    hdr = f"   {'Metric':<34s} {on_label:>14s}" + (f" {off_label:>16s} {'Improvement':>12s}" if m_off else "")
    print("-" * len(hdr)); print(hdr); print("-" * len(hdr))
    for key, label, unit, dec, signed in TABLE_ROWS:
        line = f"   {label + ' [' + unit + ']':<34s} {m_on[key]:>14.{dec}f}"
        if m_off:
            line += f" {m_off[key]:>16.{dec}f} {_improvement(m_on[key], m_off[key], signed):>11.1f}%"
        print(line)
    print("-" * len(hdr))

def save_paper_table_tex(m_on, m_off, path):
    """LaTeX rows for one experiment column group of Table I (On / Off / Improvement)."""
    def cell(v, dec, best):
        s = f"{v:.{dec}f}"
        return rf"\textbf{{{s}}}" if best else s
    rows = [
        (r"\textit{Base position $^{w}\mathbf{p}_{b}$}", None),
        (r"\quad $\mathrm{RMSE}_{3D}$ (m)",  ("pos_rmse_3d", 4, False)),
        (r"\quad $\mathrm{RMSE}_{xy}$ (m)",  ("pos_rmse_xy", 4, False)),
        (r"\quad $\mathrm{RMSE}_{z}$ (m)",   ("pos_rmse_z",  4, False)),
        (r"\quad $\bar{e}_{z}$ (m)",         ("pos_mean_z",  4, True)),
        (r"\textit{Base velocity $^{w}\mathbf{v}_{b}$}", None),
        (r"\quad $\mathrm{RMSE}_{3D}$ (m/s)", ("vel_rmse_3d", 4, False)),
        (r"\quad $\mathrm{RMSE}_{z}$ (m/s)",  ("vel_rmse_z",  4, False)),
        (r"\quad $\bar{e}_{z}$ (m/s)",        ("vel_mean_z",  4, True)),
        (r"\textit{Base orientation $^{w}\mathbf{R}_{b}$}", None),
        (r"\quad $\mathrm{RMSE}_{R}$ ($^{\circ}$)", ("rot_rmse_deg", 4, False)),
    ]
    lines = ["% On / Off / Improvement for this experiment; paste into the Table I column group.",
             "% Improvement on magnitudes for the signed e_bar_z rows."]
    for label, spec in rows:
        if spec is None:
            lines.append(rf"{label} & & & \\")
            continue
        key, dec, signed = spec
        on, off = m_on[key], m_off[key]
        on_mag, off_mag = (abs(on), abs(off)) if signed else (on, off)
        if abs(on - off) < 1e-9:
            # Identical by construction (e.g. states the partial update never
            # touches): bold neither and mark the improvement as not applicable.
            lines.append(f"{label} & {cell(on, dec, False)} & {cell(off, dec, False)} & -- \\\\")
            continue
        imp = _improvement(on, off, signed)
        imp_s = (rf"$-${abs(imp):.1f}\%" if imp < 0 else rf"{imp:.1f}\%")
        lines.append(f"{label} & {cell(on, dec, on_mag < off_mag)} & "
                     f"{cell(off, dec, off_mag < on_mag)} & {imp_s} \\\\")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[TABLE] {path}")

if PRINT_PAPER_TABLE:
    # Common interval of the ground truth and every evaluated run, so On and
    # Off are scored over exactly the same stretch of the experiment.
    _runs = [_tbl_est] + ([_tbl_abl] if _tbl_abl is not None else [])
    _t_start = max([_tbl_gt[0][0]] + [r[0][0] for r in _runs])
    _t_end = min([_tbl_gt[0][-1]] + [r[0][-1] for r in _runs])
    m_on = table_metrics(_tbl_gt, _tbl_est, _t_start, _t_end)
    m_off = table_metrics(_tbl_gt, _tbl_abl, _t_start, _t_end) if _tbl_abl is not None else None
    if m_on is None:
        print("\n [TABLE] no overlap between ground truth and estimate, table skipped")
    else:
        print_paper_table(m_on, m_off, EST_LABEL, ABLATION_LABEL if m_off else "")
        if ZERO_INITIAL_ATTITUDE:
            print("   (computed in the yaw-aligned frame of the paper, not the rotated plot frame)")
        if m_off is not None and SAVE_PAPER_TABLE:
            os.makedirs(FIGURE_DIR, exist_ok=True)
            save_paper_table_tex(m_on, m_off, os.path.join(FIGURE_DIR, f"{FIGURE_PREFIX}_table.tex"))

# --- initialisation check -------------------------------------------------
# All three attitudes below are the base attitude with respect to gravity over
# the same static window, so they are directly comparable. Serow's
# levelBaseAttitude computes exactly the "gravity implied" row.
t_static = gt_data["ts"][min(n_static, len(gt_data["ts"])) - 1] - gt_data["ts"][0]

# The ground truth row is always taken in the yaw-aligned frame, independent of
# ALIGNMENT and ZERO_INITIAL_ATTITUDE, because that is the frame Serow is meant
# to be in: all three rows below are then directly comparable and the yaw of each
# reads near zero.
_, gt_rot_yaw, _ = align_gt(gt_data["pos"], gt_data["rot"], gt_data["lin_vel"], n_static, "yaw")
rpy_gt0 = initial_attitude(gt_rot_yaw, n_static).as_euler("xyz", degrees=True)

acc_static = gt_data["acc"][:n_static]
g_hat = acc_static.mean(axis=0)
g_hat = g_hat / np.linalg.norm(g_hat)
roll_g = np.degrees(np.arctan2(g_hat[1], g_hat[2]))
pitch_g = np.degrees(-np.arcsin(np.clip(g_hat[0], -1.0, 1.0)))
acc_std = acc_static.std(axis=0).max()

# Serow is averaged over whatever part of the static window it publishes in, a
# single sample is too noisy to compare against a multi second mean
n_est_static = int(np.searchsorted(est_data["ts"], gt_data["ts"][n_static - 1], side="right"))
if n_est_static > 0:
    rpy_est0 = initial_attitude(est_data["rot"], n_est_static).as_euler("xyz", degrees=True)
    est_row = f"SEROW over {n_est_static:5d} samples"
else:
    rpy_est0 = R.from_quat(est_data["rot"][0, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
    est_row = "SEROW first sample      "

print(f"\n initialisation check over {n_static} samples ({t_static:.2f} s of standing still)")
print(f"   ground truth attitude     roll {rpy_gt0[0]:+7.2f}  pitch {rpy_gt0[1]:+7.2f}  yaw {rpy_gt0[2]:+7.2f}  deg")
print(f"   gravity implied           roll {roll_g:+7.2f}  pitch {pitch_g:+7.2f}        -      deg"
      f"   (accel std {acc_std:.3f} m/s^2)")
print(f"   {est_row}  roll {rpy_est0[0]:+7.2f}  pitch {rpy_est0[1]:+7.2f}  yaw {rpy_est0[2]:+7.2f}  deg")

tilt_err = np.array([rpy_gt0[0] - rpy_est0[0], rpy_gt0[1] - rpy_est0[1]])
fake_acc = 9.81 * np.sin(np.radians(np.linalg.norm(tilt_err)))
print(f"   initial tilt error        roll {tilt_err[0]:+7.2f}  pitch {tilt_err[1]:+7.2f}        -      deg"
      f"   -> {fake_acc:.3f} m/s^2 of uncancelled gravity")
if np.linalg.norm(tilt_err) > 0.1:
    print("   SEROW is NOT starting at the measured gravity direction. Check that")
    print("   initialize_attitude_from_gravity is true in the robot config, that the")
    print("   leveling log line is printed, and that the EKF initialisation does not")
    print("   overwrite base_orientation afterwards.")
else:
    print("   SEROW starts level with gravity.")

print("\n ATE under each alignment (same data, different frame convention):")
for mode in ("yaw", "full", "none"):
    p_m, _, _ = align_gt(gt_data["pos"], gt_data["rot"], gt_data["lin_vel"], n_static, mode)
    a_m, ax_m = compute_ATE(gt_data["ts"], p_m, est_data["ts"], est_data["pos"], per_axis=True)
    tag = "  <-- in use" if mode == ALIGNMENT else ""
    print(f"   {mode:5s} : {a_m:.4f} m   (x {ax_m[0]:.4f}  y {ax_m[1]:.4f}  z {ax_m[2]:.4f}){tag}")

# final drift and path inclination in the aligned frame
d_gt = gt_pos_aligned[-1] - gt_pos_aligned[0]
d_es = est_pos[-1] - est_pos[0]
print("\n final displacement            x        y        z")
print(f"   ground truth            {d_gt[0]:+8.3f} {d_gt[1]:+8.3f} {d_gt[2]:+8.3f}")
print(f"   {EST_LABEL:<22s}  {d_es[0]:+8.3f} {d_es[1]:+8.3f} {d_es[2]:+8.3f}")
print(f"   {'error':<22s}  {d_es[0]-d_gt[0]:+8.3f} {d_es[1]-d_gt[1]:+8.3f} {d_es[2]-d_gt[2]:+8.3f}")
if abl_data is not None and len(abl_data["ts"]) > 0:
    d_ab = abl_pos[-1] - abl_pos[0]
    print(f"   {ABLATION_LABEL:<22s}  {d_ab[0]:+8.3f} {d_ab[1]:+8.3f} {d_ab[2]:+8.3f}")
    print(f"   {'error':<22s}  {d_ab[0]-d_gt[0]:+8.3f} {d_ab[1]-d_gt[1]:+8.3f}"
          f" {d_ab[2]-d_gt[2]:+8.3f}")
hz_gt = np.linalg.norm(d_gt[:2]); hz_es = np.linalg.norm(d_es[:2])
if hz_gt > 1e-6 and hz_es > 1e-6:
    print(f"\n path inclination : GT {np.degrees(np.arctan2(d_gt[2], hz_gt)):+.2f} deg"
          f"   {EST_LABEL} {np.degrees(np.arctan2(d_es[2], hz_es)):+.2f} deg")
    if abl_data is not None and len(abl_data["ts"]) > 0:
        hz_ab = np.linalg.norm(d_ab[:2])
        if hz_ab > 1e-6:
            print(f"                                     "
                  f"{ABLATION_LABEL} {np.degrees(np.arctan2(d_ab[2], hz_ab)):+.2f} deg")
    print(f" horizontal distance : GT {hz_gt:.3f} m   est {hz_es:.3f} m   ({100*hz_es/hz_gt:.1f} %)")

# is the reported velocity consistent with the reported position?
tv = est_data["ts"]; ve = est_vel; pe = est_pos
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
r_est = R.from_quat(est_rot[:, [1, 2, 3, 0]])

rpy_gt = np.unwrap(r_gt.as_euler('xyz', degrees=True), period=360, axis=0)
rpy_est = np.unwrap(r_est.as_euler('xyz', degrees=True), period=360, axis=0)

if abl_data is not None and len(abl_data["ts"]) > 0:
    rpy_abl = np.unwrap(R.from_quat(abl_rot[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True),
                        period=360, axis=0)
else:
    rpy_abl = None

# Time axis of the paper figures. With PLOT_FROM_ESTIMATOR_START, t = 0 is the
# first estimate and the leading ground-truth-only stretch is cropped away, so
# both curves begin together.
t0 = ts_est[0] if PLOT_FROM_ESTIMATOR_START else min(ts_gt[0], ts_est[0])

# ---------------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------------
# Font sizes are absolute (points) while the figure size is in inches, so what
# lands on the page is the ratio of the two. Every size below is calibrated for
# a REFERENCE_WIDTH_IN wide figure and then scaled by the actual width, so
# widening PAPER_FIGSIZE grows the text, the rules and the ticks with it and the
# figure reads the same once LaTeX scales it to a fixed column width. Without
# that, a wider figure silently shrinks its own text on the page.
#
# Effective point size on the page = size * (column width / PAPER_FIGSIZE[0]).
# At PAPER_FIGSIZE = (12, 7) the 15 pt axis labels are drawn at 26 pt, which
# comes out at 7.6 pt in a 3.5 in column and 15 pt across a 7.16 in two-column
# figure. FONT_SCALE is an extra manual multiplier on top of all that.
PAPER_FIGSIZE = (12.0, 7.0)
REFERENCE_WIDTH_IN = 7.0
FONT_SCALE = 1.0

_s = FONT_SCALE * PAPER_FIGSIZE[0] / REFERENCE_WIDTH_IN

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.titlesize": 18 * _s,
    "axes.labelsize": 15 * _s,
    "xtick.labelsize": 13 * _s,
    "ytick.labelsize": 13 * _s,
    "legend.fontsize": 13 * _s,
    "axes.linewidth": 0.8 * _s,
    "grid.linewidth": 0.8 * _s,
    "xtick.major.width": 0.8 * _s,
    "ytick.major.width": 0.8 * _s,
    "xtick.major.size": 3.5 * _s,
    "ytick.major.size": 3.5 * _s,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,          # TrueType, publishers reject the default Type 3
    "savefig.bbox": "tight",
})
# Axis labels. The units delimiter is one constant so the three figures cannot
# drift apart, set UNITS to "()" for "(m)" or "[]" for "[m]".
UNITS = "[]"
_u = lambda unit: f"{UNITS[0]}{unit}{UNITS[1]}"

POSITION_TITLE = r"Position $^{w}\mathbf{p}_{b}$"
VELOCITY_TITLE = r"Linear velocity $^{w}\mathbf{v}_{b}$"
ORIENTATION_TITLE = r"Orientation $^{w}\mathbf{R}_{b}$"

POSITION_YLABELS = [rf"$^{{w}}p_{{b}}^{{{a}}}$ {_u('m')}" for a in "xyz"]
VELOCITY_YLABELS = [rf"$^{{w}}v_{{b}}^{{{a}}}$ {_u('m/s')}" for a in "xyz"]
ORIENTATION_YLABELS = [f"{a} {_u(chr(176))}" for a in ("roll", "pitch", "yaw")]

STYLE_GT = dict(color="black", linestyle="--", linewidth=0.8 * _s, alpha=0.8, zorder=3)
STYLE_EST = dict(color="#1f77b4", linestyle="-", linewidth=1.0 * _s, zorder=2)
STYLE_ABL = dict(color="#d95f02", linestyle="-.", linewidth=1.0 * _s, zorder=1)

def paper_figure(series, title, ylabels, name):
    """series is a list of (t, y[N,3], style, label). Saves <prefix>_<name>.pdf"""
    fig, axs = plt.subplots(3, 1, figsize=PAPER_FIGSIZE, sharex=True)
    for i in range(3):
        for t, y, style, label in series:
            axs[i].plot(t, y[:, i], label=label if i == 0 else None, **style)
        axs[i].set_ylabel(ylabels[i])
        axs[i].grid(True)
    axs[0].set_title(title, pad=8 * _s)
    axs[2].set_xlabel("Time (s)")
    fig.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(labels), frameon=False, handlelength=2.6,
               columnspacing=2.5, handletextpad=0.6)
    if SAVE_PAPER_FIGURES:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        out = os.path.join(FIGURE_DIR, f"{FIGURE_PREFIX}_{name}.pdf")
        fig.savefig(out)
        print(f"[FIGURE] {out}")
    return fig

def _crop(ts, y, style, label):
    """Drop everything before the plot origin and re-zero the time axis."""
    keep = ts >= t0 if PLOT_FROM_ESTIMATOR_START else np.ones(len(ts), dtype=bool)
    return (ts[keep] - t0, y[keep], style, label)

def build_series(gt, est, abl):
    series = [_crop(ts_gt, gt, STYLE_GT, GT_LABEL),
              _crop(ts_est, est, STYLE_EST, EST_LABEL)]
    if abl is not None:
        series.append(_crop(abl_data["ts"], abl, STYLE_ABL, ABLATION_LABEL))
    return series

# ---------------------------------------------------------
# Figure 1: base position
# ---------------------------------------------------------
paper_figure(
    build_series(gt_pos_aligned, est_pos, abl_pos),
    POSITION_TITLE,
    POSITION_YLABELS,
    "position")

# ---------------------------------------------------------
# Figure 2: base linear velocity
# ---------------------------------------------------------
paper_figure(
    build_series(gt_vel_aligned, est_vel, abl_vel),
    VELOCITY_TITLE,
    VELOCITY_YLABELS,
    "linear_velocity")

# ---------------------------------------------------------
# Figure 3: base orientation
# ---------------------------------------------------------
paper_figure(
    build_series(rpy_gt, rpy_est, rpy_abl),
    ORIENTATION_TITLE,
    ORIENTATION_YLABELS,
    "orientation")

# ---------------------------------------------------------
# Diagnostics: IMU, feet forces, contact probabilities, biases (not paper figures)
# ---------------------------------------------------------
if SHOW_DIAGNOSTICS:
    with plt.rc_context({"axes.titlesize": "large", "axes.labelsize": "medium",
                         "xtick.labelsize": "medium", "ytick.labelsize": "medium",
                         "legend.fontsize": "medium"}):
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
