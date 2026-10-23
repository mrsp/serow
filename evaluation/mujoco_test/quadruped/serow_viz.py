import matplotlib.pyplot as plt
import numpy as np
import os
import json
from mcap.reader import make_reader
from scipy.spatial.transform import Rotation as R, Slerp

# =============================================================================
# COMMON EVALUATION BLOCK
# -----------------------------------------------------------------------------
# This block is duplicated verbatim in serow_viz_1.py and serow_viz_2.py so that
# each script stands alone. It is byte-for-byte identical in both files: if you
# change one, diff the two blocks afterwards to confirm they still match.
#
# Plot convention: ground truth is BLACK DASHED, the estimate is SOLID BLUE.
#
# Metrics, all computed over the GT/estimator time overlap with GT linearly
# interpolated (SLERP for rotations) onto the estimator timestamps. Estimator
# samples farther than MAX_GT_GAP from the nearest GT sample are dropped, so
# interpolation never bridges a GT dropout.
#
#   ATE      Umeyama alignment without scale (same as `evo_ape -a`), then RMSE
#            of the translational residual norms.
#   ATEvel   RMSE of the velocity residual, with the estimate rotated into the
#            GT frame by the rotation from the position alignment.
#   RPE      Non-overlapping index pairs taken from the GT trajectory, stepping
#            either a fixed number of frames or a fixed path length.
#            Translation uses evo's point_distance relation:
#                e = ||(p_est_j - p_est_i) - (p_gt_j - p_gt_i)||
#            Rotation is reported twice, because these are NOT the same number:
#              rot_geodesic_rmse_deg  true angle of the relative rotation error
#                                     in degrees; the physical quantity.
#              rot_evo_style_rmse     ||R_err - I||_F scaled by 180/pi, which is
#                                     what `evo_rpe --pose_relation rot_part`
#                                     gives after the conversion serow_viz_1
#                                     applies. Since ||R-I||_F ~= sqrt(2)*theta,
#                                     this sits ~41% above the geodesic angle.
#                                     Kept because it reproduces the published
#                                     benchmark table.
#   Attitude Geodesic angle of R_gt^T R_est, plus a per-axis roll/pitch/yaw
#            breakdown taken from the relative rotation (so it is free of the
#            +/-180 deg wrapping that plagues raw RPY differencing).
# =============================================================================

# Ground truth (or the measured reference) is always black dashed.
# Anything estimated is always solid blue.
GT_STYLE = dict(color="black", linestyle="--", linewidth=1.4, alpha=0.9)
EST_STYLE = dict(color="#1f77b4", linestyle="-", linewidth=1.3, alpha=0.75)

# Orientation is reported as roll-pitch-yaw. "xyz" in scipy means extrinsic
# rotations about the fixed x, y, z axes, i.e. R = Rz(yaw) Ry(pitch) Rx(roll),
# the standard RPY convention used in ROS and most robotics code.
EULER_SEQ = "xyz"
EULER_LABELS = ["Roll", "Pitch", "Yaw"]

# Maximum distance [s] from an estimator sample to the nearest GT sample for
# that sample to be used.
MAX_GT_GAP = 0.1


def plot_gt_vs_est(ax, gt_t, gt_y, est_t, est_y, ylabel=None,
                   gt_label="GT", est_label="Estimated", legend=False):
    """Draw one GT/estimate pair on an axis using the standard convention."""
    # Draw the translucent estimate first, then the dashed reference on top,
    # so both curves remain visible where they coincide.
    ax.plot(est_t, est_y, label=est_label, **EST_STYLE)
    ax.plot(gt_t, gt_y, label=gt_label, **GT_STYLE)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="best")


def add_comparison_legend(ax, loc="upper left"):
    """Add the single GT/estimate legend used by a stacked figure.

    The paper legend presents the curves in the requested order: estimate,
    ground truth.
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = ("Estimated", "Ground truth")
    ax.legend(
        [by_label[label] for label in order],
        list(order),
        loc=loc,
    )


def wxyz_to_xyzw(q_wxyz):
    """MCAP-style [w, x, y, z] -> scipy-style [x, y, z, w]."""
    q = np.asarray(q_wxyz, dtype=float)
    return q[:, [1, 2, 3, 0]]


def unit_quaternions(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=1)
    n = np.where(n > 1e-12, n, 1.0)
    return q / n[:, None]


def quat_to_euler_deg(q_xyzw, unwrap=True, relative_to_first=False):
    """
    Convert (N, 4) [x, y, z, w] quaternions to roll/pitch/yaw in degrees.

    Quaternion sign flips (q and -q are the same rotation) cancel in the
    conversion, so no pre-processing is required. With unwrap=True the
    +/-180 deg discontinuity is removed, so yaw accumulates continuously
    instead of sawtoothing.

    RPY is singular at pitch = +/-90 deg; near that attitude prefer the
    geodesic error from attitude_metrics() over the per-axis curves.
    """
    rot = R.from_quat(unit_quaternions(q_xyzw))
    if relative_to_first:
        rot = rot[0].inv() * rot
    ang = rot.as_euler(EULER_SEQ)
    if unwrap:
        ang = np.unwrap(ang, axis=0)
    return np.degrees(ang)


def umeyama_no_scale(source, target):
    """Rigid alignment without scale: target ~= R_align @ source + t_align."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    H = (source - mu_s).T @ (target - mu_t)
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1.0
        R_align = Vt.T @ U.T
    return R_align, mu_t - R_align @ mu_s


def interp_vec(t_query, t_src, v_src):
    """Component-wise linear interpolation of an (N, D) array onto t_query."""
    v_src = np.asarray(v_src, dtype=float)
    return np.column_stack([
        np.interp(t_query, t_src, v_src[:, k]) for k in range(v_src.shape[1])
    ])


def associate(gt_t, est_t, max_gt_gap=MAX_GT_GAP):
    """
    Restrict estimator samples to the GT time span and drop any that sit
    farther than max_gt_gap from the nearest GT sample.

    Returns (eval_ts, indices_into_est, n_excluded).
    """
    gt_t = np.asarray(gt_t, dtype=float)
    est_t = np.asarray(est_t, dtype=float)
    inside = np.flatnonzero((est_t >= gt_t[0]) & (est_t <= gt_t[-1]))
    if inside.size == 0:
        raise RuntimeError(
            f"No temporal overlap: GT spans [{gt_t[0]:.3f}, {gt_t[-1]:.3f}], "
            f"estimate spans [{est_t[0]:.3f}, {est_t[-1]:.3f}]. "
            "Check that both files are stamped on the same clock."
        )
    j = np.searchsorted(gt_t, est_t[inside])
    lo = np.clip(j - 1, 0, len(gt_t) - 1)
    hi = np.clip(j, 0, len(gt_t) - 1)
    gap = np.minimum(np.abs(est_t[inside] - gt_t[lo]), np.abs(gt_t[hi] - est_t[inside]))
    idx = inside[gap <= max_gt_gap]
    if idx.size < 10:
        raise RuntimeError(
            f"Only {idx.size} usable GT/estimate pairs after association "
            f"(max_gt_gap={max_gt_gap:g}s). GT is probably too sparse."
        )
    return est_t[idx], idx, int(np.count_nonzero(gap > max_gt_gap))


def _error_stats(err):
    err = np.asarray(err, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "std": float(np.std(err)),
        "max": float(np.max(err)),
    }


def ate_metrics(gt_t, gt_p, est_t, est_p, align=True, max_gt_gap=MAX_GT_GAP):
    """
    Absolute Trajectory Error, mirroring `evo_ape tum ... -a` when align=True.
    Returns (metrics, R_align, t_align); metrics["rmse"] is the ATE.
    """
    gt_p = np.asarray(gt_p, dtype=float)
    est_p = np.asarray(est_p, dtype=float)

    t, idx, n_excluded = associate(gt_t, est_t, max_gt_gap)
    est_eval = est_p[idx]
    gt_eval = interp_vec(t, gt_t, gt_p)

    if align:
        R_align, t_align = umeyama_no_scale(est_eval, gt_eval)
    else:
        R_align, t_align = np.eye(3), np.zeros(3)

    diff = (R_align @ est_eval.T).T + t_align - gt_eval
    err = np.linalg.norm(diff, axis=1)

    m = _error_stats(err)
    m["final"] = float(err[-1])
    m["n_pairs"] = int(err.size)
    m["n_excluded"] = n_excluded
    m["duration"] = float(t[-1] - t[0])
    for k, name in enumerate("xyz"):
        m[f"rmse_{name}"] = float(np.sqrt(np.mean(diff[:, k] ** 2)))
        m[f"bias_{name}"] = float(np.mean(diff[:, k]))
    return m, R_align, t_align


def velocity_metrics(gt_t, gt_v, est_t, est_v, R_vel=None, max_gt_gap=MAX_GT_GAP):
    """
    Time-correct velocity RMSE (ATEvel). GT velocity is interpolated onto the
    estimator timestamps; the estimate is rotated into the GT frame by R_vel,
    normally the rotation returned by ate_metrics().
    """
    gt_v = np.asarray(gt_v, dtype=float)
    est_v = np.asarray(est_v, dtype=float)

    t, idx, n_excluded = associate(gt_t, est_t, max_gt_gap)
    est_eval = est_v[idx]
    if R_vel is not None:
        est_eval = (R_vel @ est_eval.T).T
    diff = est_eval - interp_vec(t, gt_t, gt_v)
    err = np.linalg.norm(diff, axis=1)

    m = _error_stats(err)
    m["n_pairs"] = int(err.size)
    m["n_excluded"] = n_excluded
    for k, name in enumerate("xyz"):
        m[f"rmse_{name}"] = float(np.sqrt(np.mean(diff[:, k] ** 2)))
        m[f"bias_{name}"] = float(np.mean(diff[:, k]))
    return m


def attitude_metrics(gt_t, gt_q_xyzw, est_t, est_q_xyzw,
                     max_gt_gap=MAX_GT_GAP, R_align=None):
    """
    Attitude error in degrees: the geodesic angle of R_gt^T R_est.

    No frame alignment is applied unless R_align is given. If GT and the
    estimate live in different world frames, the constant frame offset will
    otherwise dominate -- pass the rotation from ate_metrics().
    """
    t, idx, n_excluded = associate(gt_t, est_t, max_gt_gap)
    r_gt = Slerp(gt_t, R.from_quat(unit_quaternions(gt_q_xyzw)))(t)
    r_est = R.from_quat(unit_quaternions(est_q_xyzw)[idx])
    if R_align is not None:
        r_est = R.from_matrix(R_align) * r_est

    r_err = r_gt.inv() * r_est
    ang_deg = np.degrees(r_err.magnitude())
    rpy_err_deg = np.degrees(r_err.as_euler(EULER_SEQ))

    m = {f"{k}_deg": v for k, v in _error_stats(ang_deg).items()}
    m["final_deg"] = float(ang_deg[-1])
    m["n_pairs"] = int(ang_deg.size)
    m["n_excluded"] = n_excluded
    for k, name in enumerate(EULER_LABELS):
        key = name.lower()
        m[f"rmse_{key}_deg"] = float(np.sqrt(np.mean(rpy_err_deg[:, k] ** 2)))
        m[f"bias_{key}_deg"] = float(np.mean(rpy_err_deg[:, k]))
    return m


def _pairs_by_frames(n, delta):
    """Non-overlapping index pairs separated by `delta` frames."""
    delta = int(delta)
    return [(i, i + delta) for i in range(0, n - delta, delta)]


def _pairs_by_path_distance(p, delta):
    """Non-overlapping index pairs separated by `delta` metres along the path."""
    accum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    pairs = []
    i = 0
    for j in range(1, len(accum)):
        if accum[j] - accum[i] >= delta:
            pairs.append((i, j))
            i = j
    return pairs


def rpe_metrics(gt_t, gt_p, gt_q_xyzw, est_t, est_p, est_q_xyzw,
                delta=1.0, delta_unit="m", align=True, max_gt_gap=MAX_GT_GAP):
    """
    Relative Pose Error, mirroring evo_rpe with all_pairs=False.
    See the block header for the translation and rotation conventions.
    """
    if delta_unit not in ("m", "f"):
        raise ValueError("delta_unit must be 'm' (metres) or 'f' (frames)")

    gt_p = np.asarray(gt_p, dtype=float)
    est_p = np.asarray(est_p, dtype=float)

    t, idx, _ = associate(gt_t, est_t, max_gt_gap)
    p_est = est_p[idx]
    p_gt = interp_vec(t, gt_t, gt_p)
    r_gt = Slerp(gt_t, R.from_quat(unit_quaternions(gt_q_xyzw)))(t)
    r_est = R.from_quat(unit_quaternions(est_q_xyzw)[idx])

    if align:
        R_align, t_align = umeyama_no_scale(p_est, p_gt)
        p_est = (R_align @ p_est.T).T + t_align
        r_est = R.from_matrix(R_align) * r_est

    # evo derives the index pairs from the reference trajectory.
    if delta_unit == "f":
        pairs = _pairs_by_frames(len(t), int(delta))
    else:
        pairs = _pairs_by_path_distance(p_gt, float(delta))

    if not pairs:
        raise RuntimeError(
            f"No RPE pairs at delta={delta}{delta_unit}. The trajectory is "
            "shorter than one delta, or the estimate barely moves."
        )

    i_idx = np.fromiter((a for a, _ in pairs), dtype=int)
    j_idx = np.fromiter((b for _, b in pairs), dtype=int)

    trans_err = np.linalg.norm(
        (p_est[j_idx] - p_est[i_idx]) - (p_gt[j_idx] - p_gt[i_idx]), axis=1
    )

    rel_gt = r_gt[i_idx].inv() * r_gt[j_idx]
    rel_est = r_est[i_idx].inv() * r_est[j_idx]
    r_err = rel_gt.inv() * rel_est

    geo_deg = np.degrees(r_err.magnitude())
    evo_style = np.degrees(np.linalg.norm(r_err.as_matrix() - np.eye(3), axis=(1, 2)))

    return {
        "trans_rmse": float(np.sqrt(np.mean(trans_err ** 2))),
        "trans_mean": float(np.mean(trans_err)),
        "trans_max": float(np.max(trans_err)),
        "rot_geodesic_rmse_deg": float(np.sqrt(np.mean(geo_deg ** 2))),
        "rot_evo_style_rmse": float(np.sqrt(np.mean(evo_style ** 2))),
        "n_pairs": len(pairs),
    }


def evaluate(gt_t, gt_p, gt_q_xyzw, gt_v, est_t, est_p, est_q_xyzw, est_v,
             max_gt_gap=MAX_GT_GAP, with_rpe=True):
    """
    Full metric suite. Returns a flat dict of headline numbers plus the
    detailed sub-dictionaries under 'detail_*', and the alignment rotation
    under 'R_align'.
    """
    ate, R_align, _ = ate_metrics(gt_t, gt_p, est_t, est_p, True, max_gt_gap)
    vel = velocity_metrics(gt_t, gt_v, est_t, est_v, R_align, max_gt_gap)
    att = attitude_metrics(gt_t, gt_q_xyzw, est_t, est_q_xyzw, max_gt_gap, R_align)
    ate_raw, _, _ = ate_metrics(gt_t, gt_p, est_t, est_p, False, max_gt_gap)

    out = {
        "ATE": ate["rmse"],
        "ATE_unaligned": ate_raw["rmse"],
        "ATEvel": vel["rmse"],
        "ATT_deg": att["rmse_deg"],
        "R_align": R_align,
        "detail_ATE": ate,
        "detail_ATE_unaligned": ate_raw,
        "detail_ATEvel": vel,
        "detail_attitude": att,
    }

    if with_rpe:
        rpe_m = rpe_metrics(gt_t, gt_p, gt_q_xyzw, est_t, est_p, est_q_xyzw,
                            1.0, "m", True, max_gt_gap)
        rpe_f = rpe_metrics(gt_t, gt_p, gt_q_xyzw, est_t, est_p, est_q_xyzw,
                            1, "f", True, max_gt_gap)
        out.update({
            "RPE_1m_trans": rpe_m["trans_rmse"],
            "RPE_1f_trans": rpe_f["trans_rmse"],
            "RPE_1m_rot": rpe_m["rot_evo_style_rmse"],
            "RPE_1f_rot": rpe_f["rot_evo_style_rmse"],
            "RPE_1m_rot_geodesic": rpe_m["rot_geodesic_rmse_deg"],
            "RPE_1f_rot_geodesic": rpe_f["rot_geodesic_rmse_deg"],
            "detail_RPE_1m": rpe_m,
            "detail_RPE_1f": rpe_f,
        })
    return out


def headline(m, unit, keys=("rmse", "mean", "max")):
    """One-line summary for a figure suptitle."""
    parts = []
    for k in keys:
        if k in m:
            parts.append(f"{k.upper()} {m[k]:.4f} {unit}")
        elif f"{k}_deg" in m:
            parts.append(f"{k.upper()} {m[f'{k}_deg']:.4f} deg")
    return "   |   ".join(parts)


def print_error_metrics(metrics, label="SEROW", width=68):
    """Headline table plus the per-signal breakdowns. Same layout in both scripts."""
    print("\n" + "=" * width)
    print(f"{label} error metrics".center(width))
    print("=" * width)
    rows = [
        ("ATE [m]  (Umeyama-aligned, evo_ape -a)", "ATE"),
        ("ATE [m]  (no alignment)", "ATE_unaligned"),
        ("ATEvel [m/s]", "ATEvel"),
        ("Attitude [deg]  (geodesic)", "ATT_deg"),
        ("RPE trans, delta = 1 m [m]", "RPE_1m_trans"),
        ("RPE trans, delta = 1 frame [m]", "RPE_1f_trans"),
        ("RPE rot, delta = 1 m  [evo rot_part convention]", "RPE_1m_rot"),
        ("RPE rot, delta = 1 frame  [evo rot_part convention]", "RPE_1f_rot"),
        ("RPE rot, delta = 1 m  [deg, geodesic]", "RPE_1m_rot_geodesic"),
        ("RPE rot, delta = 1 frame  [deg, geodesic]", "RPE_1f_rot_geodesic"),
    ]
    for row_label, key in rows:
        if key in metrics:
            print(f"  {row_label:<52} {metrics[key]:.6f}")
    print("=" * width)
    print("  The evo rot_part rows are ||R-I||_F scaled by 180/pi, matching the")
    print("  published benchmark table; they are NOT degrees. The geodesic rows are.")
    print("=" * width)

    sections = [
        ("Position (ATE, Umeyama-aligned)", "m", "detail_ATE"),
        ("Position (no alignment)", "m", "detail_ATE_unaligned"),
        ("Linear velocity (ATEvel)", "m/s", "detail_ATEvel"),
        ("Attitude", "deg", "detail_attitude"),
    ]
    for title, unit, key in sections:
        if key not in metrics:
            continue
        print(f"\n{title}")
        print("-" * width)
        for k, v in metrics[key].items():
            if k in ("n_pairs", "n_excluded"):
                print(f"  {k:<22} {int(v)}")
            elif k == "duration":
                print(f"  {k:<22} {v:.3f} s")
            else:
                suffix = " deg" if k.endswith("_deg") else f" {unit}"
                print(f"  {k:<22} {v:.6f}{suffix}")
    print("=" * width)


# =============================================================================
# END COMMON EVALUATION BLOCK
# =============================================================================


# -------------------------------------------------------------------------
# FIGURE LAYOUT HELPER
# -------------------------------------------------------------------------
# Every figure in this script is a vertical stack of shared-x axes under a
# suptitle. Constrained layout measures the suptitle text and reserves exactly
# the space it needs, so do NOT call tight_layout(rect=[...]) afterwards: a
# hand-picked rect top such as 0.93 reserves a fixed 7% of the figure height
# regardless of how tall the title actually is, which is what left the large
# empty band under one-line titles. Calling tight_layout() on a
# constrained-layout figure also makes matplotlib warn and disable the
# constrained engine, so the two must not be mixed.
#
# If you want the same behaviour in serow_viz_1.py, copy this helper there too.


def make_stack(nrows, title, figsize=(10, 8)):
    """Vertically stacked shared-x axes with a snugly fitted suptitle."""
    fig, axs = plt.subplots(nrows, 1, figsize=figsize, sharex=True,
                            layout="constrained")
    fig.suptitle(title)
    return fig, axs


def save_paper_figure(fig, stem):
    """Save a vector PDF and a high-resolution PNG for the manuscript."""
    os.makedirs(PAPER_PLOT_DIR, exist_ok=True)
    fig.savefig(
        os.path.join(PAPER_PLOT_DIR, f"{stem}.pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(PAPER_PLOT_DIR, f"{stem}.svg"),
        format="svg",
        bbox_inches="tight",
    )


def trim_x(axs, t0, t1, pad=0.02):
    """
    Clip a stack of axes to [t0, t1] and rescale y to what is actually visible.

    Matplotlib keeps the y limits it derived from the full data when x is
    narrowed, which is why a zoomed-in view can show a flat trace against a
    y-axis spanning the whole run. Only pass axes whose lines are all real data
    series -- an axhline stores x = [0, 1] in axes coordinates and would be
    misread as a data point here.
    """
    for ax in np.atleast_1d(axs):
        ax.set_xlim(t0, t1)
        lo, hi = np.inf, -np.inf
        for line in ax.get_lines():
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            if x.size == 0 or x.size != y.size:
                continue
            visible = (x >= t0) & (x <= t1) & np.isfinite(y)
            if visible.any():
                lo = min(lo, float(y[visible].min()))
                hi = max(hi, float(y[visible].max()))
        if np.isfinite(lo) and np.isfinite(hi):
            margin = pad * (hi - lo) if hi > lo else max(abs(hi), 1.0) * pad
            ax.set_ylim(lo - margin, hi + margin)


# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
DISPLAY_PLOTS = True
CONFIG_FILE = "test_config.json"
SAVE_PAPER_PLOTS = True
PAPER_PLOT_DIR = "paper_plots"
PAPER_PLOT_DPI = 300

# The prediction MCAP starts later than the input MCAP: the runner feeds every
# input frame to the filter, but writes nothing until filter() succeeds and
# getState(true) returns a valid state, which takes SEROW's whole initialization
# (IMU bias calibration / initial attitude settling). Ground truth therefore
# extends before the estimate. With this True the GT/estimate figures are clipped
# to the estimator's own time span; set it False to see the warm-up stretch and
# how long the filter takes to come up. Metrics are unaffected either way --
# associate() only ever pairs samples inside the overlap.
TRIM_X_TO_ESTIMATE = True

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
        "b_acc": [], "b_gyr": [],
        "f_meas_fl_local": [], "f_meas_fr_local": [],
        "f_meas_rl_local": [], "f_meas_rr_local": [],
        "f_est_fl_local": [], "f_est_fr_local": [],
        "f_est_rl_local": [], "f_est_rr_local": []
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

            required_force_fields = (
                "measured_contact_forces_local",
                "estimated_contact_forces_local",
            )
            missing = [name for name in required_force_fields if name not in d]
            if missing:
                raise KeyError(
                    f"Prediction MCAP is missing force field(s) {missing}. "
                    "Re-run the modified go2_mujoco_test executable."
                )

            measured = d["measured_contact_forces_local"]
            estimated = d["estimated_contact_forces_local"]
            availability = d.get("estimated_contact_force_available", {})

            force_keys = {
                "FL_foot": ("f_meas_fl_local", "f_est_fl_local"),
                "FR_foot": ("f_meas_fr_local", "f_est_fr_local"),
                "RL_foot": ("f_meas_rl_local", "f_est_rl_local"),
                "RR_foot": ("f_meas_rr_local", "f_est_rr_local"),
            }

            for frame_name, (meas_key, est_key) in force_keys.items():
                fm = measured[frame_name]
                data_store[meas_key].append([fm["x"], fm["y"], fm["z"]])

                if availability.get(frame_name, True):
                    fe = estimated[frame_name]
                    data_store[est_key].append([fe["x"], fe["y"], fe["z"]])
                else:
                    data_store[est_key].append([np.nan, np.nan, np.nan])

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
    """Backwards-compatible scalar ATE (position RMSE, metres), no alignment."""
    m, _, _ = ate_metrics(gt_ts, gt_pos, est_ts, est_pos, align=False)
    return m["rmse"]


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

    # Warm-up report. A non-zero lead means the estimator had not produced a
    # valid state yet, not that the plot or the clocks are wrong.
    print(f"GT   {gt['ts'][0]:.3f} -> {gt['ts'][-1]:.3f} s  ({len(gt['ts'])} samples)")
    print(f"EST  {est['ts'][0]:.3f} -> {est['ts'][-1]:.3f} s  ({len(est['ts'])} samples)")
    print(f"Estimator warm-up: {est['ts'][0] - gt['ts'][0]:.3f} s "
          f"({len(gt['ts']) - len(est['ts'])} input frames without an estimate)")

    # 2. Align Ground Truth to "Zero Start"
    print("Aligning Ground Truth to Estimation Frame (0,0,0)...")
    gt_pos_aligned, gt_rot_aligned = align_gt_to_estimation_frame(gt["pos"], gt["rot"])

    q0 = gt["rot"][0]
    r0_inv = R.from_quat([q0[1], q0[2], q0[3], q0[0]]).inv()
    gt_vel_aligned = r0_inv.apply(gt["lin_vel"])

    # 3. Compute Metrics -- same suite and same table as serow_viz_1.py
    eval_metrics = evaluate(
        gt["ts"], gt_pos_aligned, wxyz_to_xyzw(gt_rot_aligned), gt_vel_aligned,
        est["ts"], est["pos"], wxyz_to_xyzw(est["rot"]), est["lin_vel"],
    )
    print_error_metrics(eval_metrics, label="SEROW")

    ate_pos = eval_metrics["ATE"]
    R_align = eval_metrics["R_align"]

    # Velocity is compared in the GT frame, using the same rotation ATEvel uses.
    est_vel_plot = (R_align @ est["lin_vel"].T).T

    # 4. Plotting
    if DISPLAY_PLOTS:
        components = ("x", "y", "z")

        # --- Figure 1: Base Position ---
        fig1, axs1 = make_stack(3, r"Position ${}^{w}\mathbf{p}_b$")
        for i, component in enumerate(components):
            plot_gt_vs_est(
                axs1[i], gt["ts"], gt_pos_aligned[:, i], est["ts"], est["pos"][:, i],
                ylabel=rf"${{}}^{{w}}p_b^{{{component}}}$ (m)",
                gt_label="Ground truth", est_label="Estimated",
            )
        add_comparison_legend(axs1[0])
        axs1[2].set_xlabel(r"Time (s)")

        # --- Figure 2: Base Orientation (roll / pitch / yaw) ---
        gt_rpy = quat_to_euler_deg(wxyz_to_xyzw(gt_rot_aligned))
        est_rpy = quat_to_euler_deg(wxyz_to_xyzw(est["rot"]))

        fig2, axs2 = make_stack(
            3, r"Orientation ${}^{w}\mathbf{R}_b$"
        )
        for i, name in enumerate(EULER_LABELS):
            plot_gt_vs_est(
                axs2[i], gt["ts"], gt_rpy[:, i], est["ts"], est_rpy[:, i],
                ylabel=f"{name} (deg)",
                gt_label="Ground truth", est_label="Estimated",
            )
        add_comparison_legend(axs2[0], loc="upper right")
        axs2[2].set_xlabel(r"Time (s)")

        # --- Figure 3: Linear Velocity ---
        fig3, axs3 = make_stack(3, r"Linear velocity ${}^{w}\mathbf{v}_b$")
        for i, component in enumerate(components):
            plot_gt_vs_est(
                axs3[i], gt["ts"], gt_vel_aligned[:, i], est["ts"], est_vel_plot[:, i],
                ylabel=rf"${{}}^{{w}}v_b^{{{component}}}$ (m/s)",
                gt_label="Ground truth", est_label="Estimated",
            )
        add_comparison_legend(axs3[0])
        axs3[2].set_xlabel(r"Time (s)")

        # Drop the warm-up stretch where only ground truth exists, so the three
        # comparison figures start where the estimator does.
        if TRIM_X_TO_ESTIMATE:
            for axs in (axs1, axs2, axs3):
                trim_x(axs, est["ts"][0], est["ts"][-1])

        if SAVE_PAPER_PLOTS:
            save_paper_figure(fig1, "position")
            save_paper_figure(fig2, "orientation")
            save_paper_figure(fig3, "linear_velocity")

        # --- Figure 4: Biases (not a GT/estimate pair -> per-axis colours) ---
        fig4, axs4 = make_stack(2, "Estimated IMU biases")

        axs4[0].plot(est["ts"], est["b_acc"], label=["ax", "ay", "az"])
        axs4[0].axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
        axs4[0].set_ylabel("Accel bias (m/s^2)")
        axs4[0].legend()
        axs4[0].grid(True, alpha=0.3)

        axs4[1].plot(est["ts"], est["b_gyr"], label=["wx", "wy", "wz"])
        axs4[1].axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
        axs4[1].set_ylabel("Gyro bias (rad/s)")
        axs4[1].legend()
        axs4[1].grid(True, alpha=0.3)
        axs4[1].set_xlabel("Time (s)")

        # --- Figures 5-8: Contact forces in each local foot frame ---
        # The measured force is the reference here, so it takes the GT style.
        force_series = {
            "FL": (est["f_meas_fl_local"], est["f_est_fl_local"]),
            "FR": (est["f_meas_fr_local"], est["f_est_fr_local"]),
            "RL": (est["f_meas_rl_local"], est["f_est_rl_local"]),
            "RR": (est["f_meas_rr_local"], est["f_est_rr_local"]),
        }
        force_axes = ["Fx", "Fy", "Fz"]

        for leg, (measured_local, estimated_local) in force_series.items():
            valid = np.isfinite(estimated_local).all(axis=1)
            if valid.any():
                err = np.linalg.norm(estimated_local[valid] - measured_local[valid], axis=1)
                subtitle = (
                    f"\nRMSE {np.sqrt(np.mean(err ** 2)):.4f} N   |   "
                    f"MAX {err.max():.4f} N   |   "
                    f"{int(valid.sum())}/{len(valid)} samples with an estimate"
                )
            else:
                subtitle = "\n(no valid GMO estimate samples)"

            # The two-line suptitle needs no special handling: constrained
            # layout measures the rendered text height itself.
            fig_force, axs_force = make_stack(
                3, f"{leg} contact force in local foot frame{subtitle}",
                figsize=(12, 8),
            )

            for axis_idx, axis_name in enumerate(force_axes):
                plot_gt_vs_est(
                    axs_force[axis_idx],
                    est["ts"], measured_local[:, axis_idx],
                    est["ts"], estimated_local[:, axis_idx],
                    ylabel=f"{axis_name} [N]",
                    gt_label="Measured (local, unchanged)",
                    est_label="GMO estimate (world -> local foot)",
                )
            axs_force[0].legend(loc="upper right")
            axs_force[-1].set_xlabel("Time (s)")

        # --- Figure 9: Raw IMU Measurements (diagnostic, per-axis colours) ---
        fig6, axs6 = make_stack(2, "Raw IMU measurements")

        imu_acc = np.asarray(gt["acc"])
        for i, name in enumerate(["Acc X", "Acc Y", "Acc Z"]):
            axs6[0].plot(gt["ts"], imu_acc[:, i], label=name, alpha=0.8)
        axs6[0].set_ylabel("Linear acc (m/s^2)")
        axs6[0].legend(loc="upper right")
        axs6[0].grid(True, alpha=0.3)

        imu_gyr = np.asarray(gt["gyr"])
        for i, name in enumerate(["Gyr X", "Gyr Y", "Gyr Z"]):
            axs6[1].plot(gt["ts"], imu_gyr[:, i], label=name, alpha=0.8)
        axs6[1].set_ylabel("Angular vel (rad/s)")
        axs6[1].set_xlabel("Time (s)")
        axs6[1].legend(loc="upper right")
        axs6[1].grid(True, alpha=0.3)

        plt.show()
