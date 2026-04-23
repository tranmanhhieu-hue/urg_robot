import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_dir = get_package_share_directory('robot_camera')
    params_file = os.path.join(package_dir, 'config', 'robot_camera_params.yaml')

    camera_node = Node(
        package='robot_camera',
        executable='robot_camera',
        name='robot_camera',
        output='screen',
        parameters=[params_file]
    )

    haptic_node = Node(
        package='robot_camera',
        executable='haptic_controller',
        name='haptic_controller',
        output='screen',
        parameters=[params_file]
    )

    esp32_bridge_node = Node(
        package='robot_camera',
        executable='esp32_bridge',
        name='esp32_bridge',
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        camera_node,
        haptic_node,
        esp32_bridge_node
    ])