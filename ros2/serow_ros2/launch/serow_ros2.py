from launch import LaunchDescription
from launch_ros.actions import Node
 
def generate_launch_description():
    return LaunchDescription([
        Node(
            package='serow_ros2',
            executable='serow_ros2',
            name='serow_ros2',
            output="screen"
        ),
    ])
