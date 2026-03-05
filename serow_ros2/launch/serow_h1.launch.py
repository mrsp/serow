import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_node_env_from_config(config_path):
    """Build node env (e.g. LD_LIBRARY_PATH) from config YAML."""
    node_env = {}
    if not config_path or not os.path.isfile(config_path):
        return node_env
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        extra = data.get("path_to_serow_library")
        if extra:
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if extra not in ld_path.split(":"):
                ld_path = f"{ld_path}:{extra}" if ld_path else extra
            node_env["LD_LIBRARY_PATH"] = ld_path
    except Exception:
        pass
    return node_env


def _launch_setup(context):
    config_file = LaunchConfiguration("config_file").perform(context)
    node_env = _load_node_env_from_config(config_file)
    return [
        Node(
            package="serow_ros2",
            executable="serow_ros2",
            name="serow_ros2",
            output="screen",
            parameters=[{"config_file": LaunchConfiguration("config_file")}],
            additional_env=node_env if node_env else None,
        )
    ]


def generate_launch_description():
    default_config_file = os.path.join(
        get_package_share_directory("serow_ros2"), "config", "h1.yaml"
    )
    config_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config_file,
        description="Path to the YAML config file",
    )
    return LaunchDescription(
        [
            config_arg,
            OpaqueFunction(function=_launch_setup),
        ]
    )
