# Copyright (c) 2021 Juan Miguel Jimeno
#
# Licensed under the Apache License, Version 2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
    TextSubstitution,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


# 1:1 to match every sandbox result and the Gazebo world launch
# (susag_new_model/launch/gazebo_barn.launch.py); see the comment there.
BARN_SCALE = 1.0
BARN_MAPS_DIR = (
    f'/home/saran/robohouse_ws/src/BARN_dataset/scaled_{BARN_SCALE:g}/map_files_fine'
)
# map_files_fine (2026-09-15): every 0.15 m BARN map resampled exactly to 0.05 m
# (obstacles are 0.15 m grid-aligned, so no geometry changes). The global costmap
# takes the map's resolution; at 0.15 m a single lidar mark on an obstacle face
# widened it by 15 cm and closed world 48's 7 cm-margin bottleneck (planner
# failures -> spin recovery -> collision). map_files/ is left untouched.

# Vicon-like localization for simulation (2026-09-15): the real trials use
# Vicon markers, not AMCL. With sim:=true and localization:=ground_truth (the
# default), this node publishes map->odom from Gazebo's /ground_truth/odom and
# AMCL's tf_broadcast is rewritten to false so only one node owns map->odom.
# localization:=amcl restores the old behaviour. Without sim:=true nothing
# changes (real-robot launches keep AMCL untouched).
GT_LOCALIZER = '/home/saran/robohouse_ws/src/susag_nav2/scripts/ground_truth_localizer.py'


def generate_launch_description():

    nav2_launch_path = PathJoinSubstitution(
        [FindPackageShare('nav2_bringup'), 'launch', 'bringup_launch.py']
    )

    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare('susag_nav2'), 'rviz', 'susag_nav.rviz']
    )

    world_idx = LaunchConfiguration('world_idx')
    default_map_path = [
        TextSubstitution(text=f'{BARN_MAPS_DIR}/yaml_'),
        world_idx,
        TextSubstitution(text='.yaml'),
    ]

    nav2_config_path = PathJoinSubstitution(
        [FindPackageShare('susag_nav2'), 'param', 'navigation_sim_tight.yaml']
    )

    use_gt_localization = PythonExpression([
        "'", LaunchConfiguration('sim'), "'.lower() == 'true' and '",
        LaunchConfiguration('localization'), "' == 'ground_truth'"])
    amcl_tf_broadcast = PythonExpression([
        "'false' if ('", LaunchConfiguration('sim'), "'.lower() == 'true' and '",
        LaunchConfiguration('localization'), "' == 'ground_truth') else 'true'"])
    configured_params = RewrittenYaml(
        source_file=LaunchConfiguration('nav2_params'),
        param_rewrites={'tf_broadcast': amcl_tf_broadcast},
        convert_types=True)

    return LaunchDescription([

        DeclareLaunchArgument(
            name='sim',
            default_value='false',
            description='Use simulation time if true'
        ),

        DeclareLaunchArgument(
            name='rviz',
            default_value='true',
            description='Run RViz'
        ),

        DeclareLaunchArgument(
            name='world_idx',
            default_value='0',
            description='BARN world/map index; uses the 1:1 (scaled_1) map'
        ),

        DeclareLaunchArgument(
            name='map',
            default_value=default_map_path,
            description='Optional full map YAML override'
        ),

        DeclareLaunchArgument(
            name='nav2_params',
            default_value=nav2_config_path,
            description='Full path to the nav2 params YAML file'
        ),

        DeclareLaunchArgument(
            name='localization',
            default_value='ground_truth',
            description="'ground_truth' (Vicon-like, sim:=true only) or 'amcl'"
        ),

        ExecuteProcess(
            cmd=['python3', GT_LOCALIZER, '--ros-args', '-p',
                 ['use_sim_time:=', LaunchConfiguration('sim')]],
            name='ground_truth_localizer',
            output='screen',
            condition=IfCondition(use_gt_localization)
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch_path),
            launch_arguments={
                'map': LaunchConfiguration('map'),
                'use_sim_time': LaunchConfiguration('sim'),
                'params_file': configured_params
            }.items()
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path],
            condition=IfCondition(LaunchConfiguration('rviz')),
            parameters=[
                {'use_sim_time': LaunchConfiguration('sim')}
            ]
        )
    ])
