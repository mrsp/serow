#!/usr/bin/env python3
"""
SEROW / ANYmal CYN-1 visualizer and benchmark-style evaluator.

Expected CSV columns for both GT and SEROW estimate:
    t, x, y, z, qx, qy, qz, qw, vx, vy, vz

Important conventions used here:
    1. SEROW starts writing later than GT, after initialization/calibration.
       Therefore GT is interpolated/cropped to SEROW timestamps.

    2. Positions are rebased at the common start time:
           p_gt  <- p_gt(t)  - p_gt(t_serow_start)
           p_est <- p_est(t) - p_est(first SEROW sample)
       This makes both plotted trajectories start at (0,0,0).

    3. For the orientation plot only, GT uses the benchmark frame fix
       that made the GT RPY curves match the benchmark paper.
       SEROW does not use this fix.

    4. Metrics:
       - ATE: SE(3) Umeyama rigid alignment, no scale, translation RMSE.
       - ATEvel: raw velocity RMSE. The velocity is NOT rotated by the ATE
         alignment, because that made ATEvel artificially optimistic.
       - Translational RPE: RMSE of relative position increments over the
         selected windows after applying the ATE rigid alignment to positions.
         This avoids contaminating translational RPE with quaternion frame
         plotting conventions.
       - Rotational RPE: relative rotation RMSE in degrees.

    5. Plots are visual/debug plots. By default, SEROW position and velocity
       are yaw-aligned for plotting only, so the curves visually overlap well.
       This does not change the computed metrics.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp


DEFAULT_GT = "/home/michael/github/serow/evaluation/anymal_test/anymal_data/test/cyn-1/groundtruth.csv"
DEFAULT_EST = "/home/michael/github/serow/evaluation/anymal_test/anymal_data/test/cyn-1/serow/fused_state.csv"


# -----------------------------------------------------------------------------
# CSV loading
# -----------------------------------------------------------------------------

def read_csv_auto(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns)
        )


def load_pose_csv(path: str):
    df = read_csv_auto(path)

    required = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz"]
    require_columns(df, required)

    t_abs = df["t"].to_numpy(dtype=float)
    pos = df[["x", "y", "z"]].to_numpy(dtype=float)
    quat_xyzw = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
    vel = df[["vx", "vy", "vz"]].to_numpy(dtype=float)

    valid = (
        np.isfinite(t_abs)
        & np.all(np.isfinite(pos), axis=1)
        & np.all(np.isfinite(quat_xyzw), axis=1)
        & np.all(np.isfinite(vel), axis=1)
    )

    t_abs = t_abs[valid]
    pos = pos[valid]
    quat_xyzw = quat_xyzw[valid]
    vel = vel[valid]

    order = np.argsort(t_abs)
    t_abs = t_abs[order]
    pos = pos[order]
    quat_xyzw = quat_xyzw[order]
    vel = vel[order]

    # Remove duplicate timestamps while preserving chronological order.
    _, unique_idx = np.unique(t_abs, return_index=True)
    unique_idx = np.sort(unique_idx)
    t_abs = t_abs[unique_idx]
    pos = pos[unique_idx]
    quat_xyzw = quat_xyzw[unique_idx]
    vel = vel[unique_idx]

    # Normalize quaternions defensively.
    qnorm = np.linalg.norm(quat_xyzw, axis=1)
    good = qnorm > 1e-12
    t_abs = t_abs[good]
    pos = pos[good]
    quat_xyzw = quat_xyzw[good] / qnorm[good, None]
    vel = vel[good]

    t = t_abs - t_abs[0]
    return {
        "t_abs": t_abs,
        "t": t,
        "pos_raw": pos,
        "pos_zero": pos - pos[0],
        "quat_xyzw": quat_xyzw,
        "vel": vel,
    }


# -----------------------------------------------------------------------------
# Synchronization / cropping
# -----------------------------------------------------------------------------

def interp_columns(t_query, t_source, values):
    return np.column_stack([
        np.interp(t_query, t_source, values[:, 0]),
        np.interp(t_query, t_source, values[:, 1]),
        np.interp(t_query, t_source, values[:, 2]),
    ])


def sync_and_crop_gt_to_est(gt_raw, est_raw):
    """
    Return GT and SEROW sampled at SEROW timestamps, only over the common interval.

    Both positions are zero-started at the common start. This is the key fix for
    plotting after SEROW starts writing later than GT.
    """
    t_gt_abs = gt_raw["t_abs"]
    t_est_abs = est_raw["t_abs"]

    overlap_start = max(t_gt_abs[0], t_est_abs[0])
    overlap_end = min(t_gt_abs[-1], t_est_abs[-1])

    est_mask = (t_est_abs >= overlap_start) & (t_est_abs <= overlap_end)
    if np.count_nonzero(est_mask) < 2:
        raise RuntimeError("No sufficient GT/SEROW timestamp overlap.")

    t_sync_abs = t_est_abs[est_mask]
    t_sync = t_sync_abs - t_sync_abs[0]

    # Interpolate GT to SEROW timestamps.
    gt_pos = interp_columns(t_sync_abs, t_gt_abs, gt_raw["pos_raw"])
    gt_vel = interp_columns(t_sync_abs, t_gt_abs, gt_raw["vel"])

    gt_rot_all = R.from_quat(gt_raw["quat_xyzw"])
    gt_slerp = Slerp(t_gt_abs, gt_rot_all)
    gt_rot = gt_slerp(t_sync_abs)
    gt_quat = gt_rot.as_quat()

    # SEROW samples over the same interval.
    est_pos = est_raw["pos_raw"][est_mask]
    est_vel = est_raw["vel"][est_mask]
    est_quat = est_raw["quat_xyzw"][est_mask]

    # Rebase positions so both trajectories start at the same common origin.
    gt_pos = gt_pos - gt_pos[0]
    est_pos = est_pos - est_pos[0]

    gt = {
        "t_abs": t_sync_abs,
        "t": t_sync,
        "pos_raw": gt_pos,
        "pos_zero": gt_pos - gt_pos[0],
        "quat_xyzw": gt_quat,
        "vel": gt_vel,
    }
    est = {
        "t_abs": t_sync_abs,
        "t": t_sync,
        "pos_raw": est_pos,
        "pos_zero": est_pos - est_pos[0],
        "quat_xyzw": est_quat,
        "vel": est_vel,
    }

    return gt, est, overlap_start, overlap_end


# -----------------------------------------------------------------------------
# Orientation helpers
# -----------------------------------------------------------------------------

def rotations_from_quat_xyzw(quat_xyzw: np.ndarray, benchmark_frame_fix: bool = False):
    rot = R.from_quat(quat_xyzw)
    if benchmark_frame_fix:
        frame_fix = R.from_euler("x", np.pi)
        rot = frame_fix * rot * frame_fix.inv()
    return rot


def rpy_from_quat_xyzw(
    quat_xyzw: np.ndarray,
    unwrap: bool = True,
    benchmark_frame_fix: bool = False,
):
    """Convert qx,qy,qz,qw to RPY relative to the first orientation."""
    rot_abs = rotations_from_quat_xyzw(
        quat_xyzw,
        benchmark_frame_fix=benchmark_frame_fix,
    )
    rot_rel = rot_abs[0].inv() * rot_abs
    rpy_rad = rot_rel.as_euler("xyz", degrees=False)

    if unwrap:
        rpy_rad = np.unwrap(rpy_rad, axis=0)

    return np.rad2deg(rpy_rad)


# -----------------------------------------------------------------------------
# Alignment helpers
# -----------------------------------------------------------------------------

def se3_umeyama_no_scale(source: np.ndarray, target: np.ndarray):
    """Rigid SE(3) alignment without scale: target ~= R_align * source + t_align."""
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    A = source - mu_s
    B = target - mu_t

    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T

    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1.0
        R_align = Vt.T @ U.T

    t_align = mu_t - R_align @ mu_s
    return R_align, t_align


def yaw_alignment_matrix_2d(source_xy: np.ndarray, target_xy: np.ndarray):
    """Best constant yaw rotation, no scale. Source and target should share origin."""
    A = source_xy - source_xy[0]
    B = target_xy - target_xy[0]

    c = np.sum(A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1])
    s = np.sum(A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0])
    theta = np.arctan2(s, c)

    R2 = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])
    return R2, theta


def apply_yaw_alignment_for_plot(gt, est):
    """Yaw-align SEROW to GT for visual plotting only. Metrics are unaffected."""
    R2, theta = yaw_alignment_matrix_2d(est["pos_raw"][:, :2], gt["pos_raw"][:, :2])

    est_plot = dict(est)
    est_plot["pos_raw"] = est["pos_raw"].copy()
    est_plot["pos_zero"] = est["pos_zero"].copy()
    est_plot["vel"] = est["vel"].copy()

    est_plot["pos_raw"][:, :2] = (R2 @ est["pos_raw"][:, :2].T).T
    est_plot["pos_zero"] = est_plot["pos_raw"] - est_plot["pos_raw"][0]
    est_plot["vel"][:, :2] = (R2 @ est["vel"][:, :2].T).T

    print(f"Plot-only XY yaw alignment angle [deg]: {np.rad2deg(theta):.6f}")
    return est_plot


def apply_se3_alignment_for_plot(gt, est, R_align, t_align):
    """SE(3)-align SEROW position for visual plotting only."""
    est_plot = dict(est)
    est_plot["pos_raw"] = (R_align @ est["pos_raw"].T).T + t_align
    est_plot["pos_zero"] = est_plot["pos_raw"] - est_plot["pos_raw"][0]
    est_plot["vel"] = est["vel"].copy()
    return est_plot


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def rmse_vec(x: np.ndarray):
    return float(np.sqrt(np.mean(np.sum(x * x, axis=1))))


def frame_pairs(n: int, delta_frames: int = 1):
    i = np.arange(0, n - delta_frames)
    j = i + delta_frames
    return np.column_stack([i, j])


def distance_pairs(gt_pos: np.ndarray, delta_m: float = 1.0):
    """
    Pairs separated by approximately delta_m along the reference trajectory arc length.
    This mirrors the idea of evo RPE with delta_unit=meter.
    """
    step = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])

    target = cum + delta_m
    j = np.searchsorted(cum, target, side="left")

    i = np.arange(len(cum))
    valid = j < len(cum)
    return np.column_stack([i[valid], j[valid]])


def translational_rpe_from_position_deltas(gt_pos, est_pos, pairs):
    """
    Translational RPE as relative position increment error.

    This is intentionally position-delta based, not orientation-frame based, because the
    GT Euler plotting convention requires a frame fix that should not leak into the
    translational RPE. This fixes the artificially huge RPE(1m) produced by mixing
    translational errors with the plotting rotation convention.
    """
    if len(pairs) == 0:
        return np.nan

    i = pairs[:, 0]
    j = pairs[:, 1]

    gt_delta = gt_pos[j] - gt_pos[i]
    est_delta = est_pos[j] - est_pos[i]
    err = est_delta - gt_delta
    return rmse_vec(err)


def rotational_rpe(gt_rot, est_rot, pairs):
    if len(pairs) == 0:
        return np.nan

    i = pairs[:, 0]
    j = pairs[:, 1]

    gt_rel = gt_rot[i].inv() * gt_rot[j]
    est_rel = est_rot[i].inv() * est_rot[j]
    rot_err = gt_rel.inv() * est_rel

    return float(np.sqrt(np.mean(np.rad2deg(rot_err.magnitude()) ** 2)))


def compute_metrics(gt, est):
    """
    Compute benchmark-style metrics on already synchronized/cropped trajectories.
    """
    gt_pos = gt["pos_raw"]
    est_pos = est["pos_raw"]
    gt_vel = gt["vel"]
    est_vel = est["vel"]

    # ATE alignment, no scale.
    R_align, t_align = se3_umeyama_no_scale(est_pos, gt_pos)
    est_pos_se3 = (R_align @ est_pos.T).T + t_align

    # For rotational RPE, use the same convention used for the orientation plot.
    gt_rot_metric = rotations_from_quat_xyzw(gt["quat_xyzw"], benchmark_frame_fix=True)
    est_rot_metric = rotations_from_quat_xyzw(est["quat_xyzw"], benchmark_frame_fix=False)

    ate = rmse_vec(est_pos_se3 - gt_pos)

    # Do NOT rotate velocity here. This is intentionally raw velocity RMSE.
    atevel = rmse_vec(est_vel - gt_vel)

    pairs_1frame = frame_pairs(len(gt_pos), delta_frames=1)
    pairs_1m = distance_pairs(gt_pos, delta_m=1.0)

    # Use SE(3)-aligned positions for translational RPE so that constant global pose
    # offsets do not dominate local-motion errors.
    rpe_1f_t = translational_rpe_from_position_deltas(gt_pos, est_pos_se3, pairs_1frame)
    rpe_1m_t = translational_rpe_from_position_deltas(gt_pos, est_pos_se3, pairs_1m)

    rpe_1f_r = rotational_rpe(gt_rot_metric, est_rot_metric, pairs_1frame)
    rpe_1m_r = rotational_rpe(gt_rot_metric, est_rot_metric, pairs_1m)

    metrics = {
        "ATE": ate,
        "ATEvel": atevel,
        "RPE_1m_trans": rpe_1m_t,
        "RPE_1frame_trans": rpe_1f_t,
        "RPE_1m_rot": rpe_1m_r,
        "RPE_1frame_rot": rpe_1f_r,
        "N_sync": len(gt_pos),
        "N_pairs_1m": len(pairs_1m),
        "N_pairs_1frame": len(pairs_1frame),
    }

    return metrics, R_align, t_align, est_pos_se3


def format_metrics_for_title(metrics):
    return (
        f"ATE={metrics['ATE']:.3f} m | "
        f"ATEvel={metrics['ATEvel']:.3f} m/s | "
        f"RPE Δ=1m: {metrics['RPE_1m_trans']:.3f} m, {metrics['RPE_1m_rot']:.3f}° | "
        f"RPE Δ=1fr: {metrics['RPE_1frame_trans']:.4f} m, {metrics['RPE_1frame_rot']:.4f}°"
    )


def print_metrics(metrics):
    print("================ Benchmark-style metrics ================")
    print(f"Synchronized samples:       {metrics['N_sync']}")
    print(f"RPE Δ=1m pairs:            {metrics['N_pairs_1m']}")
    print(f"RPE Δ=1frame pairs:        {metrics['N_pairs_1frame']}")
    print(f"ATE [m]:                   {metrics['ATE']:.6f}")
    print(f"ATEvel raw [m/s]:          {metrics['ATEvel']:.6f}")
    print(f"RPE Δ=1m trans [m]:        {metrics['RPE_1m_trans']:.6f}")
    print(f"RPE Δ=1m rot [deg]:        {metrics['RPE_1m_rot']:.6f}")
    print(f"RPE Δ=1frame trans [m]:    {metrics['RPE_1frame_trans']:.6f}")
    print(f"RPE Δ=1frame rot [deg]:    {metrics['RPE_1frame_rot']:.6f}")
    print("=========================================================")


def green_if_best(text, value, best_value):
    GREEN = "\033[92m"
    END = "\033[0m"
    if np.isclose(value, best_value, rtol=1e-12, atol=1e-12):
        return f"{GREEN}{text}{END}"
    return text


def print_benchmark_comparison_table(metrics):
    rows = [
        ("ATE [m]", {
            "MUSE": 2.269461,
            "IEKF": 1.405668,
            "IS": 1.363114,
            "SEROW": metrics["ATE"],
        }),
        ("ATEvel [m/s]", {
            "MUSE": 0.876147,
            "IEKF": 0.869383,
            "IS": 0.869481,
            "SEROW": metrics["ATEvel"],
        }),
        ("RPE (Δ = 1 meter) [m]", {
            "MUSE": 0.072223,
            "IEKF": 0.043187,
            "IS": 0.042526,
            "SEROW": metrics["RPE_1m_trans"],
        }),
        ("RPE (Δ = 1 frame) [m]", {
            "MUSE": 0.000670,
            "IEKF": 0.000544,
            "IS": 0.001965,
            "SEROW": metrics["RPE_1frame_trans"],
        }),
        ("RPE (Δ = 1 meter) [°]", {
            "MUSE": 0.578469,
            "IEKF": 0.568987,
            "IS": 0.565644,
            "SEROW": metrics["RPE_1m_rot"],
        }),
        ("RPE (Δ = 1 frame) [°]", {
            "MUSE": 0.002605,
            "IEKF": 0.002611,
            "IS": 0.002614,
            "SEROW": metrics["RPE_1frame_rot"],
        }),
    ]

    methods = ["MUSE", "IEKF", "IS", "SEROW"]
    label_w = 28
    value_w = 12
    total_w = label_w + (value_w + 1) * len(methods)

    print("\n" + "=" * total_w)
    print("Benchmark comparison table")
    print("=" * total_w)

    header = f"{'RMSE':<{label_w}}"
    for method in methods:
        header += f" {method:>{value_w}}"
    print(header)
    print("-" * total_w)

    for label, values in rows:
        best_value = min(values.values())
        line = f"{label:<{label_w}}"
        for method in methods:
            value = values[method]
            txt_plain = f"{value:.6f}"
            txt_padded = f"{txt_plain:>{value_w}}"
            line += " " + green_if_best(txt_padded, value, best_value)
        print(line)

    print("=" * total_w + "\n")


# -----------------------------------------------------------------------------
# Diagnostics and plotting
# -----------------------------------------------------------------------------

def print_diagnostics(name, traj, rpy_deg):
    t = traj["t"]
    pos = traj["pos_raw"]
    vel = traj["vel"]

    print(f"================ {name} diagnostics ================")
    print(f"Samples:       {len(t)}")
    print(f"Duration:      {t[-1]:.3f} s")
    print(f"Timestamp abs: {traj['t_abs'][0]:.9f} -> {traj['t_abs'][-1]:.9f}")
    print("Position ranges:")
    print(f"  x: {pos[:,0].min(): .6f} -> {pos[:,0].max(): .6f} m")
    print(f"  y: {pos[:,1].min(): .6f} -> {pos[:,1].max(): .6f} m")
    print(f"  z: {pos[:,2].min(): .6f} -> {pos[:,2].max(): .6f} m")
    print("Velocity ranges:")
    print(f"  vx: {vel[:,0].min(): .6f} -> {vel[:,0].max(): .6f} m/s")
    print(f"  vy: {vel[:,1].min(): .6f} -> {vel[:,1].max(): .6f} m/s")
    print(f"  vz: {vel[:,2].min(): .6f} -> {vel[:,2].max(): .6f} m/s")
    print("Orientation ranges:")
    print(f"  roll:  {rpy_deg[:,0].min(): .6f} -> {rpy_deg[:,0].max(): .6f} deg")
    print(f"  pitch: {rpy_deg[:,1].min(): .6f} -> {rpy_deg[:,1].max(): .6f} deg")
    print(f"  yaw:   {rpy_deg[:,2].min(): .6f} -> {rpy_deg[:,2].max(): .6f} deg")
    print("=====================================================")


def plot_pose(gt, gt_rpy_deg, est, est_rpy_deg, metrics_text=None):
    fig = plt.figure(figsize=(15, 6))
    title = "CYN-1 pose estimate vs ground truth"
    if metrics_text is not None:
        title += "\n" + metrics_text
    fig.suptitle(title)

    gs = fig.add_gridspec(3, 3)

    t_gt = gt["t"]
    p_gt = gt["pos_raw"]
    t_est = est["t"]
    p_est = est["pos_raw"]

    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.plot(p_gt[:, 0], p_gt[:, 1], "--", label="GT")
    ax_traj.plot(p_est[:, 0], p_est[:, 1], label="SEROW")
    ax_traj.set_title("(a) XY trajectory")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.axis("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc="best")

    pos_labels = [r"$p_x$ [m]", r"$p_y$ [m]", r"$p_z$ [m]"]
    pos_axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for i, ax in enumerate(pos_axes):
        ax.plot(t_gt, p_gt[:, i], "--", label="GT")
        ax.plot(t_est, p_est[:, i], label="SEROW")
        ax.set_ylabel(pos_labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("(b) Position")
            ax.legend(loc="best")
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time since SEROW start [s]")

    ori_labels = ["roll [deg]", "pitch [deg]", "yaw [deg]"]
    ori_axes = [fig.add_subplot(gs[i, 2]) for i in range(3)]
    for i, ax in enumerate(ori_axes):
        ax.plot(t_gt, gt_rpy_deg[:, i], "--", label="GT")
        ax.plot(t_est, est_rpy_deg[:, i], label="SEROW")
        ax.set_ylabel(ori_labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("(c) Orientation")
            ax.legend(loc="best")
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time since SEROW start [s]")

    plt.tight_layout(rect=[0, 0, 1, 0.90])


def plot_velocity(gt, est, metrics_text=None):
    fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
    title = "CYN-1 velocity estimate vs ground truth"
    if metrics_text is not None:
        title += "\n" + metrics_text
    fig.suptitle(title)

    labels = [r"$v_x$ [m/s]", r"$v_y$ [m/s]", r"$v_z$ [m/s]"]
    for i, ax in enumerate(axes):
        ax.plot(gt["t"], gt["vel"][:, i], "--", label="GT")
        ax.plot(est["t"], est["vel"][:, i], label="SEROW")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time since SEROW start [s]")
    plt.tight_layout(rect=[0, 0, 1, 0.90])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default=DEFAULT_GT)
    parser.add_argument("--est", default=DEFAULT_EST)
    parser.add_argument("--no-unwrap", action="store_true")
    parser.add_argument(
        "--plot-pose-alignment",
        choices=["yaw", "se3", "none"],
        default="yaw",
        help="Pose plotting only. Metrics are unaffected.",
    )
    parser.add_argument(
        "--plot-velocity-alignment",
        choices=["yaw", "raw"],
        default="yaw",
        help="Velocity plotting only. Metrics always use raw velocity.",
    )
    args = parser.parse_args()

    gt_raw = load_pose_csv(args.gt)
    est_raw = load_pose_csv(args.est)

    gt, est, overlap_start, overlap_end = sync_and_crop_gt_to_est(gt_raw, est_raw)

    gt_rpy_deg = rpy_from_quat_xyzw(
        gt["quat_xyzw"],
        unwrap=not args.no_unwrap,
        benchmark_frame_fix=True,
    )
    est_rpy_deg = rpy_from_quat_xyzw(
        est["quat_xyzw"],
        unwrap=not args.no_unwrap,
        benchmark_frame_fix=False,
    )

    metrics, R_align, t_align, _ = compute_metrics(gt, est)
    metrics_text = format_metrics_for_title(metrics)

    print(f"GT/SEROW overlap abs time: {overlap_start:.9f} -> {overlap_end:.9f}")
    print_diagnostics("cropped GT", gt, gt_rpy_deg)
    print_diagnostics("SEROW", est, est_rpy_deg)
    print_metrics(metrics)
    print_benchmark_comparison_table(metrics)

    # Plotting-only alignment. This does not modify metrics.
    if args.plot_pose_alignment == "yaw":
        est_pose_plot = apply_yaw_alignment_for_plot(gt, est)
    elif args.plot_pose_alignment == "se3":
        est_pose_plot = apply_se3_alignment_for_plot(gt, est, R_align, t_align)
    else:
        est_pose_plot = est

    if args.plot_velocity_alignment == "yaw":
        est_vel_plot = apply_yaw_alignment_for_plot(gt, est)
    else:
        est_vel_plot = est

    plot_pose(gt, gt_rpy_deg, est_pose_plot, est_rpy_deg, metrics_text)
    plot_velocity(gt, est_vel_plot, metrics_text)
    plt.show()


if __name__ == "__main__":
    main()
