from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to the config file
    config_file = os.path.join(get_package_share_directory('serow_ros2'), 'config', 'h1_config.yaml')
    print("path is ",config_file)
    # Declare the config file argument
    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to the YAML config file'
    )

    return LaunchDescription([
        config_arg,
        Node(
            package='serow_ros2',
            executable='serow_ros2',
            name='serow_ros2',
            output="screen",
            parameters=[{
                'config_file': LaunchConfiguration('config_file')
            }]
        ),
    ])
