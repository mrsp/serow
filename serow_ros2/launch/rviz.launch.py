"""
Launch RViz2 with the serow config. The path topic can be overridden at launch time.

Examples:
  # Default path topic (/serow/odom/path)
  ros2 launch serow_ros2 rviz.launch.py

  # Use a different path topic
  ros2 launch serow_ros2 rviz.launch.py path_topic:=/my_robot/odom/path

  # Or with a custom config
  ros2 launch serow_ros2 rviz.launch.py config:=/path/to/other.rviz path_topic:=/other/path
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("serow_ros2")
    default_rviz = os.path.join(pkg_share, "serow_ros2.rviz")

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=default_rviz,
        description="Path to the RViz config file",
    )
    path_topic_arg = DeclareLaunchArgument(
        "path_topic",
        default_value="/serow/odom/path",
        description="Topic name for the Path display (remapped from /serow/odom/path in the config)",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", LaunchConfiguration("config")],
        remappings=[
            # Config uses /serow/odom/path; remap to the launch argument so it's variable
            ("/serow/odom/path", LaunchConfiguration("path_topic")),
        ],
    )

    return LaunchDescription([config_arg, path_topic_arg, rviz_node])
