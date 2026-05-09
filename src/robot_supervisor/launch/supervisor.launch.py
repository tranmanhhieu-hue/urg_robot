from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="robot_supervisor",
            executable="commander",
            output="screen"
        ),
        Node(
            package="robot_supervisor",
            executable="voice_chat",
            output="screen",
            emulate_tty=True,
        ),
        # Node(
        #     package='robot_supervisor',
        #     executable='navigation_alert_node',
        #     name='navigation_alert_node',
        #     output='screen',
        #     parameters=[{
        #         'path_topic': '/plan',
        #         'scan_topic': '/scan',
        #         'alert_topic': '/nav_alert_text',
        #         'robot_frame': 'base_link',
        #         'global_frame': 'map',
        #         'turn_lookahead_m': 1.0,
        #         'turn_after_m': 0.7,
        #         'turn_angle_threshold_deg': 35.0,
        #         'obstacle_distance_m': 1.0,
        #         'front_angle_deg': 25.0,
        #         'cooldown_sec': 5.0,
        #     }]
        # ),
    ])