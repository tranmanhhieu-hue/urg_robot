import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_dir = get_package_share_directory('robot_navigation')
    map_file = os.path.join(package_dir, 'maps', 'my_map.yaml')
    params_file = os.path.join(package_dir, 'config', 'nav2_params_v3.yaml')
    rviz_config = os.path.join(package_dir, 'rviz', 'nav2_default_view.rviz')
    ekf_config_file = os.path.join(package_dir, 'config', 'ekf_localization.yaml')

    # MAP SERVER
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            {'yaml_filename': map_file},
            {'use_sim_time': False}
        ]
    )

    # AMCL
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[params_file]
    )

    # PLANNER
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file]
    )

    # CONTROLLER
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[params_file]
    )

    # SMOOTHER SERVER
    smoother_server = Node(
        package='nav2_smoother',
        executable='smoother_server',
        name='smoother_server',
        output='screen',
        parameters=[params_file]
    )

    # BT NAVIGATOR
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file]
    )

    # BEHAVIOR SERVER
    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[params_file]
    )

    # EKF (nếu cần dùng thì bỏ comment)
    # ekf_node = Node(
    #     package='robot_localization',
    #     executable='ekf_node',
    #     name='ekf_filter_node',
    #     output='screen',
    #     parameters=[ekf_config_file, {'use_sim_time': False}]
    # )

    # LIFECYCLE MANAGER: LOCALIZATION
    lifecycle_manager_loc = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': [
                'map_server',
                'amcl'
            ]
        }]
    )

    # LIFECYCLE MANAGER: NAVIGATION
    lifecycle_manager_nav = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': [
                'planner_server',
                'controller_server',
                'smoother_server',
                'bt_navigator',
                'behavior_server'
            ]
        }]
    )

    # RVIZ
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        map_server,
        amcl,
        planner_server,
        controller_server,
        smoother_server,
        bt_navigator,
        behavior_server,
        lifecycle_manager_loc,
        lifecycle_manager_nav,
        rviz_node,
        # ekf_node,
    ])