#!/usr/bin/env python3
"""Compare SEROW with and without terrain correction against ground truth.

Input locations are taken from ``test_config.json`` next to this script:

    {
      "Experiment": {"type": "slope_large"},
      "Paths": {
        "base_path": "/evaluation/mujoco_test/quadruped/mujoco_data",
        "data_file": "{base_path}/{type}/go2_data.mcap",
        "prediction_file": "{base_path}/{type}/serow_predictions.mcap",
        ...
      }
    }

``base_path`` is interpreted relative to the repository root (the directory
that contains ``.git``).  The two ablation runs are derived from
``prediction_file`` by inserting ``_with_terrain`` / ``_without_terrain``
before the extension, e.g.

    <repo>/evaluation/mujoco_test/quadruped/mujoco_data/slope_large/go2_data.mcap
    <repo>/.../slope_large/serow_predictions_with_terrain.mcap
    <repo>/.../slope_large/serow_predictions_without_terrain.mcap

Any of these can still be overridden on the command line.

The script produces three stacked-component figures for base position,
orientation (roll, pitch, yaw), and world-frame base linear velocity.  The
ground truth is transformed once into the estimator's initial frame.  The two
estimates are not independently trajectory-aligned, because doing so could hide
the drift difference that this terrain ablation is intended to show.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mcap.reader import make_reader
from scipy.spatial.transform import Rotation as R, Slerp


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIRECTORY / "test_config.json"

EULER_SEQUENCE = "xyz"
COMPONENTS = ("x", "y", "z")
EULER_LABELS = ("Roll", "Pitch", "Yaw")
VERTICAL_AXIS = COMPONENTS.index("z")

# Sized for figures that will be reduced to one IEEE/TRO column.  The larger
# source fonts remain legible after LaTeX scales the 7-inch figure down to
# \columnwidth.
FIGURE_SIZE = (7.0, 6.2)
TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 15
TICK_LABEL_FONT_SIZE = 13
LEGEND_FONT_SIZE = 13

# The legend is drawn in its own band underneath the bottom subplot instead of
# inside the top axes, where it used to cover the roll/x traces.  The fraction
# below is the share of the figure height reserved for that band.
LEGEND_BAND_HEIGHT = 0.06
LEGEND_COLUMNS = 3

# Draw estimates first and the dashed reference last.  This keeps the ground
# truth visible where all three curves coincide.
WITH_TERRAIN_STYLE = {
    "color": "#1f77b4",
    "linestyle": "-",
    "linewidth": 1.0,
    "alpha": 0.8,
    "zorder": 2,
}
WITHOUT_TERRAIN_STYLE = {
    "color": "#d95f02",
    "linestyle": "-.",
    "linewidth": 1.0,
    "alpha": 0.80,
    "zorder": 1,
}
GROUND_TRUTH_STYLE = {
    "color": "black",
    "linestyle": "--",
    "linewidth": 0.8,
    "alpha": 0.60,
    "zorder": 3,
}


# --------------------------------------------------------------------------- #
# Configuration / path resolution
# --------------------------------------------------------------------------- #
def find_repo_root(start: Path) -> Path:
    """Walk upwards from ``start`` until a directory containing ``.git``."""
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    # Fallback for a checkout without .git: this script lives at
    # <repo>/evaluation/mujoco_test/quadruped/, i.e. three levels down.
    parents = start.parents
    return parents[2] if len(parents) > 2 else start


def load_test_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Test config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _with_suffix_tag(path: Path, tag: str) -> Path:
    """``serow_predictions.mcap`` -> ``serow_predictions_<tag>.mcap``."""
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def resolve_config_paths(
    config: dict, repo_root: Path, experiment_type: str | None = None
) -> dict[str, Path]:
    """Expand the ``Paths`` templates of ``test_config.json`` into real paths."""
    experiment_type = experiment_type or config["Experiment"]["type"]
    templates = config["Paths"]

    raw_base = Path(templates["base_path"]).expanduser()
    if raw_base.is_absolute() and raw_base.is_dir():
        # A genuine absolute path on this machine.
        base_path = raw_base
    else:
        # Repository-relative path, written with a leading slash in the config.
        base_path = repo_root / str(raw_base).lstrip("/\\")

    def expand(template: str) -> Path:
        return Path(
            template.format(base_path=str(base_path), type=experiment_type)
        )

    data_file = expand(templates["data_file"])
    prediction_file = expand(templates["prediction_file"])

    return {
        "experiment_type": experiment_type,
        "base_path": base_path,
        "data_dir": data_file.parent,
        "gt": data_file,
        "with_terrain": _with_suffix_tag(prediction_file, "with_terrain"),
        "without_terrain": _with_suffix_tag(prediction_file, "without_terrain"),
    }


def resolve_input_path(data_directory: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else data_directory / path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _sorted_arrays(data: dict[str, list]) -> dict[str, np.ndarray]:
    """Convert to arrays, sort by time, and discard duplicate timestamps."""
    arrays = {key: np.asarray(value, dtype=float) for key, value in data.items()}
    if arrays["ts"].size == 0:
        return arrays

    order = np.argsort(arrays["ts"], kind="stable")
    sorted_time = arrays["ts"][order]
    _, unique_indices = np.unique(sorted_time, return_index=True)
    order = order[np.sort(unique_indices)]
    return {key: value[order] for key, value in arrays.items()}


def load_ground_truth(mcap_path: Path) -> dict[str, np.ndarray]:
    """Load base ground truth from ``go2_data.mcap``."""
    data = {"ts": [], "pos": [], "rot": [], "lin_vel": []}

    print(f"Loading ground truth:       {mcap_path}")
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream)
        for _, channel, message in reader.iter_messages(
            topics=["/robot_state", "robot_state"]
        ):
            if channel.topic.lstrip("/") != "robot_state":
                continue

            payload = json.loads(message.data)
            gt = payload["base_ground_truth"]

            data["ts"].append(payload["timestamp"])
            data["pos"].append(
                [gt["position"][axis] for axis in COMPONENTS]
            )
            # MCAP quaternion order: [w, x, y, z].
            data["rot"].append(
                [gt["orientation"][axis] for axis in ("w", "x", "y", "z")]
            )
            data["lin_vel"].append(
                [gt["linear_velocity"][axis] for axis in COMPONENTS]
            )

    return _sorted_arrays(data)


def load_prediction(mcap_path: Path) -> dict[str, np.ndarray]:
    """Load the base pose and linear velocity from a SEROW prediction MCAP."""
    data = {"ts": [], "pos": [], "rot": [], "lin_vel": []}

    print(f"Loading prediction:         {mcap_path}")
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream)
        for _, channel, message in reader.iter_messages(
            topics=["serow_predictions", "/serow_predictions"]
        ):
            if channel.topic.lstrip("/") != "serow_predictions":
                continue

            payload = json.loads(message.data)
            base = payload["base_pose"]

            data["ts"].append(payload["timestamp"])
            data["pos"].append(
                [base["position"][axis] for axis in COMPONENTS]
            )
            # MCAP quaternion order: [w, x, y, z].
            data["rot"].append(
                [base["rotation"][axis] for axis in ("w", "x", "y", "z")]
            )

            if "linear_velocity" not in base:
                raise KeyError(
                    f"{mcap_path.name} has no base_pose.linear_velocity field."
                )
            data["lin_vel"].append(
                [base["linear_velocity"][axis] for axis in COMPONENTS]
            )

    return _sorted_arrays(data)


def _validate_stream(name: str, data: dict[str, np.ndarray]) -> None:
    if data["ts"].size == 0:
        raise RuntimeError(f"No samples were loaded for {name}.")
    if np.any(np.diff(data["ts"]) < 0):
        raise RuntimeError(f"Timestamps for {name} are not monotonically increasing.")
    for key in ("pos", "lin_vel"):
        if data[key].ndim != 2 or data[key].shape[1] != 3:
            raise RuntimeError(f"Unexpected {name} {key} shape: {data[key].shape}")
    if data["rot"].ndim != 2 or data["rot"].shape[1] != 4:
        raise RuntimeError(f"Unexpected {name} rot shape: {data['rot'].shape}")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def wxyz_to_xyzw(quaternions: np.ndarray) -> np.ndarray:
    """Convert an ``(N, 4)`` quaternion array from wxyz to scipy's xyzw."""
    q = np.asarray(quaternions, dtype=float)
    return q[:, (1, 2, 3, 0)]


def normalized_xyzw(quaternions_wxyz: np.ndarray) -> np.ndarray:
    q = wxyz_to_xyzw(quaternions_wxyz)
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("A zero-norm quaternion was found in an MCAP stream.")
    return q / norms[:, None]


def align_ground_truth_to_initial_frame(
    gt: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Express GT position, attitude, and velocity in the initial GT frame."""
    q0_wxyz = gt["rot"][0]
    # Use a list for fancy indexing.  A tuple on a 1-D NumPy array is treated
    # as multidimensional indexing and raises "too many indices".
    initial_rotation = R.from_quat(q0_wxyz[[1, 2, 3, 0]])
    inverse_initial = initial_rotation.inv()

    aligned = dict(gt)
    aligned["pos"] = inverse_initial.apply(gt["pos"] - gt["pos"][0])
    aligned["lin_vel"] = inverse_initial.apply(gt["lin_vel"])

    current_rotation = R.from_quat(normalized_xyzw(gt["rot"]))
    aligned_xyzw = (inverse_initial * current_rotation).as_quat()
    aligned["rot"] = aligned_xyzw[:, (3, 0, 1, 2)]
    return aligned


def crop_stream(
    data: dict[str, np.ndarray], start_time: float, end_time: float
) -> dict[str, np.ndarray]:
    mask = (data["ts"] >= start_time) & (data["ts"] <= end_time)
    cropped = {key: value[mask] for key, value in data.items()}
    if cropped["ts"].size < 2:
        raise RuntimeError(
            f"Only {cropped['ts'].size} samples remain in the common time interval."
        )
    return cropped


def common_time_interval(*streams: dict[str, np.ndarray]) -> tuple[float, float]:
    start_time = max(float(stream["ts"][0]) for stream in streams)
    end_time = min(float(stream["ts"][-1]) for stream in streams)
    if end_time <= start_time:
        spans = ", ".join(
            f"[{stream['ts'][0]:.6f}, {stream['ts'][-1]:.6f}]"
            for stream in streams
        )
        raise RuntimeError(f"The three MCAP streams do not overlap: {spans}")
    return start_time, end_time


def quaternion_to_euler_deg(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions to continuous roll-pitch-yaw in degrees."""
    angles_rad = R.from_quat(normalized_xyzw(quaternions_wxyz)).as_euler(
        EULER_SEQUENCE
    )
    return np.degrees(np.unwrap(angles_rad, axis=0))


def _relative_time(data: dict[str, np.ndarray], origin: float) -> np.ndarray:
    return data["ts"] - origin


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def make_stack(title: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE, sharex=True)
    # Put the title on the top axes instead of using a figure-level suptitle.
    # This avoids reserving a large empty band above the actual plots.
    axes[0].set_title(title, fontsize=TITLE_FONT_SIZE, pad=5)
    return fig, axes


def plot_three_way_stack(
    *,
    title: str,
    ylabels: tuple[str, str, str],
    gt_time: np.ndarray,
    gt_values: np.ndarray,
    with_time: np.ndarray,
    with_values: np.ndarray,
    without_time: np.ndarray,
    without_values: np.ndarray,
) -> plt.Figure:
    """Create one three-component GT/terrain-on/terrain-off comparison."""
    fig, axes = make_stack(title)

    legend_handles: list[plt.Line2D] = []
    for component, axis in enumerate(axes):
        with_line = axis.plot(
            with_time,
            with_values[:, component],
            label="With terrain",
            **WITH_TERRAIN_STYLE,
        )[0]
        without_line = axis.plot(
            without_time,
            without_values[:, component],
            label="Without terrain",
            **WITHOUT_TERRAIN_STYLE,
        )[0]
        gt_line = axis.plot(
            gt_time,
            gt_values[:, component],
            label="Ground truth",
            **GROUND_TRUTH_STYLE,
        )[0]
        if not legend_handles:
            legend_handles = [gt_line, with_line, without_line]

        axis.set_ylabel(ylabels[component], fontsize=AXIS_LABEL_FONT_SIZE)
        axis.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONT_SIZE)

    # Lay the subplots out first, leaving an empty band at the bottom, then
    # place a single-row figure legend in that band.  Keeping the legend
    # outside the data area means it can never occlude a trace, whatever the
    # signal happens to do near the top-right corner of the first subplot.
    fig.tight_layout(
        pad=0.45, h_pad=0.35, rect=(0.0, LEGEND_BAND_HEIGHT, 1.0, 1.0)
    )
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=LEGEND_COLUMNS,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.4,
        columnspacing=1.6,
        borderaxespad=0.0,
    )
    return fig


def save_figure(
    figure: plt.Figure,
    output_directory: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output_path = output_directory / f"{stem}.{extension}"
        save_kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            save_kwargs["dpi"] = dpi
        figure.savefig(output_path, format=extension, **save_kwargs)
        print(f"Saved: {output_path}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def interpolate_vectors(
    target_time: np.ndarray, source_time: np.ndarray, source_values: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_time, source_time, source_values[:, component])
            for component in range(source_values.shape[1])
        ]
    )


def vector_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    residual_norm = np.linalg.norm(estimate - reference, axis=1)
    return float(np.sqrt(np.mean(residual_norm**2)))


def component_rmse(
    reference: np.ndarray, estimate: np.ndarray, component: int
) -> float:
    """RMSE of a single Cartesian component, e.g. the vertical axis alone."""
    residual = estimate[:, component] - reference[:, component]
    return float(np.sqrt(np.mean(residual**2)))


def component_mean_error(
    reference: np.ndarray, estimate: np.ndarray, component: int
) -> float:
    """Signed mean error of a single component, i.e. the accumulated bias."""
    return float(np.mean(estimate[:, component] - reference[:, component]))


def orientation_rmse_deg(
    gt: dict[str, np.ndarray], estimate: dict[str, np.ndarray]
) -> float:
    gt_rotation = R.from_quat(normalized_xyzw(gt["rot"]))
    gt_at_estimate_time = Slerp(gt["ts"], gt_rotation)(estimate["ts"])
    estimate_rotation = R.from_quat(normalized_xyzw(estimate["rot"]))
    error_rotation = gt_at_estimate_time.inv() * estimate_rotation
    errors_deg = np.degrees(error_rotation.magnitude())
    return float(np.sqrt(np.mean(errors_deg**2)))


def print_ablation_summary(
    gt: dict[str, np.ndarray],
    with_terrain: dict[str, np.ndarray],
    without_terrain: dict[str, np.ndarray],
) -> None:
    """Print unaligned RMSE values over the common interval.

    Full-vector norms are reported alongside vertical-axis-only figures.  The
    terrain update acts exclusively on the vertical channel, so the 3-D norms
    dilute its effect with horizontal error that the map never touches.
    """
    print("\nTerrain ablation RMSE (common interval, no trajectory alignment)")
    print("-" * 88)
    print(
        f"{'Metric':<36}{'With terrain':>16}{'Without terrain':>18}"
        f"{'Improvement':>16}"
    )
    print("-" * 88)

    rows = []
    for estimate in (with_terrain, without_terrain):
        gt_pos = interpolate_vectors(estimate["ts"], gt["ts"], gt["pos"])
        gt_vel = interpolate_vectors(estimate["ts"], gt["ts"], gt["lin_vel"])
        rows.append(
            {
                "Position RMSE [m]": vector_rmse(gt_pos, estimate["pos"]),
                "Vertical position RMSE [m]": component_rmse(
                    gt_pos, estimate["pos"], VERTICAL_AXIS
                ),
                "Mean vertical position error [m]": component_mean_error(
                    gt_pos, estimate["pos"], VERTICAL_AXIS
                ),
                "Final vertical position error [m]": float(
                    estimate["pos"][-1, VERTICAL_AXIS] - gt_pos[-1, VERTICAL_AXIS]
                ),
                "Linear velocity RMSE [m/s]": vector_rmse(
                    gt_vel, estimate["lin_vel"]
                ),
                "Vertical velocity RMSE [m/s]": component_rmse(
                    gt_vel, estimate["lin_vel"], VERTICAL_AXIS
                ),
                "Mean vertical velocity error [m/s]": component_mean_error(
                    gt_vel, estimate["lin_vel"], VERTICAL_AXIS
                ),
                "Orientation RMSE [deg]": orientation_rmse_deg(gt, estimate),
            }
        )

    with_metrics, without_metrics = rows
    for label, with_value in with_metrics.items():
        without_value = without_metrics[label]
        # Signed metrics are compared by magnitude: a bias of -0.30 m is worse
        # than one of +0.02 m, whatever the sign.
        if abs(without_value) > 1e-12:
            improvement = (
                (abs(without_value) - abs(with_value)) / abs(without_value) * 100.0
            )
            improvement_text = f"{improvement:>15.1f}%"
        else:
            improvement_text = f"{'n/a':>16}"
        print(
            f"{label:<36}{with_value:>16.6f}{without_value:>18.6f}"
            f"{improvement_text}"
        )
    print("-" * 88)

    # Position is the integral of velocity, so the vertical velocity bias of the
    # terrain-off run should account for the vertical drift it accumulates.  A
    # mismatch here means the drift is entering somewhere other than the
    # velocity channel and is worth investigating before it reaches the paper.
    duration = float(without_terrain["ts"][-1] - without_terrain["ts"][0])
    bias = without_metrics["Mean vertical velocity error [m/s]"]
    print(
        f"\nIntegral consistency check (terrain off): a mean vertical velocity "
        f"error of {bias * 1e3:.2f} mm/s over {duration:.1f} s integrates to "
        f"{bias * duration:+.3f} m of vertical drift, against a measured mean "
        f"vertical position error of "
        f"{without_metrics['Mean vertical position error [m]']:+.3f} m."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ground truth against SEROW predictions with and without "
            "terrain correction.  Input locations come from test_config.json."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Test config JSON (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--type",
        dest="experiment_type",
        default=None,
        help="Override Experiment.type from the config (e.g. slope, trot).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Repository root that Paths.base_path is relative to.  Detected "
            "automatically from the .git directory if omitted."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the directory containing the three MCAP files.",
    )
    parser.add_argument(
        "--gt", default=None, help="Override the ground-truth MCAP path."
    )
    parser.add_argument(
        "--with-terrain", default=None, help="Override the terrain-on MCAP path."
    )
    parser.add_argument(
        "--without-terrain",
        default=None,
        help="Override the terrain-off MCAP path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Plot output directory. Defaults to "
            "<data-dir>/terrain_ablation_plots."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg", "png"),
        default=("pdf", "svg"),
        help="Output formats (default: pdf svg).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    parser.add_argument(
        "--no-show", action="store_true", help="Save plots without opening GUI windows."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config_path = args.config.expanduser().resolve()
    config = load_test_config(config_path)
    repo_root = (
        args.repo_root.expanduser().resolve()
        if args.repo_root is not None
        else find_repo_root(SCRIPT_DIRECTORY)
    )
    resolved = resolve_config_paths(config, repo_root, args.experiment_type)

    print(f"Config:                     {config_path}")
    print(f"Repository root:            {repo_root}")
    print(f"Experiment type:            {resolved['experiment_type']}")

    data_directory = (
        args.data_dir.expanduser().resolve()
        if args.data_dir is not None
        else resolved["data_dir"]
    )
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_directory / "terrain_ablation_plots"
    )

    # Explicit CLI paths win; otherwise use the config-derived ones.  When only
    # --data-dir is overridden, keep the config file names but in that folder.
    def pick(override: str | None, default: Path) -> Path:
        if override is not None:
            return resolve_input_path(data_directory, override)
        if args.data_dir is not None:
            return data_directory / default.name
        return default

    gt_path = pick(args.gt, resolved["gt"])
    with_terrain_path = pick(args.with_terrain, resolved["with_terrain"])
    without_terrain_path = pick(args.without_terrain, resolved["without_terrain"])

    for path in (gt_path, with_terrain_path, without_terrain_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required MCAP file not found: {path}")

    gt = load_ground_truth(gt_path)
    with_terrain = load_prediction(with_terrain_path)
    without_terrain = load_prediction(without_terrain_path)

    _validate_stream("ground truth", gt)
    _validate_stream("with-terrain estimate", with_terrain)
    _validate_stream("without-terrain estimate", without_terrain)

    # Both prediction files are assumed to use the same estimator initial
    # frame.  Only the simulator ground truth needs the common origin change.
    gt_reference = align_ground_truth_to_initial_frame(gt)

    start_time, end_time = common_time_interval(
        gt_reference, with_terrain, without_terrain
    )
    gt = crop_stream(gt_reference, start_time, end_time)
    with_terrain = crop_stream(with_terrain, start_time, end_time)
    without_terrain = crop_stream(without_terrain, start_time, end_time)

    print(
        f"Common interval: {start_time:.6f} -> {end_time:.6f} "
        f"({end_time - start_time:.3f} s)"
    )
    print(
        "Samples: "
        f"GT={len(gt['ts'])}, with terrain={len(with_terrain['ts'])}, "
        f"without terrain={len(without_terrain['ts'])}"
    )

    # Use the complete GT stream for interpolation so estimates at the exact
    # common-interval boundaries remain inside the Slerp domain.
    print_ablation_summary(gt_reference, with_terrain, without_terrain)

    gt_time = _relative_time(gt, start_time)
    with_time = _relative_time(with_terrain, start_time)
    without_time = _relative_time(without_terrain, start_time)

    position_labels = tuple(
        rf"${{}}^{{w}}p_b^{{{component}}}$ (m)" for component in COMPONENTS
    )
    velocity_labels = tuple(
        rf"${{}}^{{w}}v_b^{{{component}}}$ (m/s)" for component in COMPONENTS
    )
    orientation_labels = tuple(f"{name} (deg)" for name in EULER_LABELS)

    position_figure = plot_three_way_stack(
        title=r"Position ${}^{w}\mathbf{p}_b$",
        ylabels=position_labels,
        gt_time=gt_time,
        gt_values=gt["pos"],
        with_time=with_time,
        with_values=with_terrain["pos"],
        without_time=without_time,
        without_values=without_terrain["pos"],
    )

    orientation_figure = plot_three_way_stack(
        title=r"Orientation ${}^{w}\mathbf{R}_b$",
        ylabels=orientation_labels,
        gt_time=gt_time,
        gt_values=quaternion_to_euler_deg(gt["rot"]),
        with_time=with_time,
        with_values=quaternion_to_euler_deg(with_terrain["rot"]),
        without_time=without_time,
        without_values=quaternion_to_euler_deg(without_terrain["rot"]),
    )

    velocity_figure = plot_three_way_stack(
        title=r"Linear velocity ${}^{w}\mathbf{v}_b$",
        ylabels=velocity_labels,
        gt_time=gt_time,
        gt_values=gt["lin_vel"],
        with_time=with_time,
        with_values=with_terrain["lin_vel"],
        without_time=without_time,
        without_values=without_terrain["lin_vel"],
    )

    formats = tuple(args.formats)
    save_figure(
        position_figure, output_directory, "terrain_ablation_position", formats, args.dpi
    )
    save_figure(
        orientation_figure,
        output_directory,
        "terrain_ablation_orientation",
        formats,
        args.dpi,
    )
    save_figure(
        velocity_figure,
        output_directory,
        "terrain_ablation_linear_velocity",
        formats,
        args.dpi,
    )

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
