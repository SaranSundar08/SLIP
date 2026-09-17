# Copyright (c) 2021 Juan Miguel Jimeno
#
# Licensed under the Apache License, Version 2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, OpaqueFunction, LogInfo
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
import math
import tempfile
import yaml


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

# dynamic_obstacles:=true (2026-09-15, sim only): moving obstacles are handled by
# prediction (DynamicObstacleCritic) instead of the frozen costmap, so the
# costmaps must not mark them at their current positions. This mode (1) starts
# tracked_obstacle_scan_filter.py and points both obstacle layers at
# /front_scan_filtered, (2) removes the depth-camera STVL layers (they would mark
# the obstacles too), and (3) sets flow_assist_only_when_path_blocked: false so
# TG-MPPI's branch / space-time modes exist every cycle (T-MPC runs guidance every
# cycle). Default false: every other launch is unchanged. TG-MPPI params only --
# ignored for the stock Nav2 MPPI baseline, which always runs unmodified.
SCAN_FILTER = '/home/saran/robohouse_ws/src/susag_nav2/scripts/tracked_obstacle_scan_filter.py'
TRACKING_NOISE = '/home/saran/robohouse_ws/src/susag_nav2/scripts/obstacle_tracking_noise.py'

# tracking_quality:=<preset> (2026-09-16). The obstacles are currently PERFECTLY
# observed -- Gazebo p3d ground truth, no noise, no latency, no misses -- which is
# what T-MPC had (motion capture + Kalman filter) but not what a lidar tracker on
# the real robot would produce. These presets degrade the states the controller
# plans on, so "how good must tracking be?" can be answered with a measurement
# instead of a guess. 'perfect' changes nothing and stays the reference condition.
TRACKING_PRESETS = {
    'perfect': None,
    'good':    {'pos_sigma': 0.03, 'vel_sigma': 0.05, 'latency': 0.05, 'dropout': 0.01,
                'rate': 20.0},
    'poor':    {'pos_sigma': 0.10, 'vel_sigma': 0.20, 'latency': 0.15, 'dropout': 0.10,
                'rate': 10.0},
}


# ablation:=<condition> (2026-09-17). Each condition switches off TG-MPPI parts so
# their contribution can be measured separately. 'no_topology' is plain MPPI plus the
# predicted-obstacle critic: with tgmppi_bias_enabled false the optimizer skips the
# pseudopod bias (and with it space-time, which is only reachable from inside the
# bias step), and falls back from the grouped winner-takes-all update to the standard
# single-softmax update, while DynamicObstacleCritic keeps running. That answers
# "is the prediction critic alone enough?" without touching the stock baseline.
ABLATIONS = {
    'full': {},
    'no_spacetime': {'tgmppi_spacetime_enabled': False},
    'no_topology': {'tgmppi_bias_enabled': False, 'tgmppi_spacetime_enabled': False},
}


def ablation_params(src_path, condition):
    with open(src_path) as f:
        data = yaml.safe_load(f)
    fp = data.get('controller_server', {}).get('ros__parameters', {}).get('FollowPath', {})
    changes = []
    for key, value in ABLATIONS[condition].items():
        if key in fp and fp[key] != value:
            fp[key] = value
            changes.append('%s -> %s' % (key, value))
    if not changes:
        return src_path, []
    out = tempfile.NamedTemporaryFile(mode='w', suffix='_ablation.yaml', delete=False)
    yaml.safe_dump(data, out)
    out.close()
    return out.name, changes


def backend_params(src_path, backend):
    """Force TG-MPPI's compute_backend (cpu|cuda).

    The shared critics are byte-identical CPU code in both forks; TG-MPPI only
    adds an #ifdef TGMPPI_WITH_CUDA branch to each. So with compute_backend:cuda
    the two controllers run DIFFERENT implementations of the same critic maths,
    and TG-MPPI additionally gets much shorter cycles -- an engineering
    advantage, not a method one. backend:=cpu puts both on the identical code
    path for a clean method comparison. No effect on the stock baseline, which
    has no such parameter.
    """
    with open(src_path) as f:
        data = yaml.safe_load(f)
    fp = data.get('controller_server', {}).get('ros__parameters', {}).get('FollowPath', {})
    if 'compute_backend' not in fp or fp['compute_backend'] == backend:
        return src_path, []
    previous = fp['compute_backend']
    fp['compute_backend'] = backend
    out = tempfile.NamedTemporaryFile(mode='w', suffix='_backend.yaml', delete=False)
    yaml.safe_dump(data, out)
    out.close()
    return out.name, ['compute_backend %s -> %s' % (previous, backend)]


def tracking_params(src_path):
    """Point the controller's obstacle subscriptions at the degraded /tracked/odom
    topics instead of ground truth."""
    with open(src_path) as f:
        data = yaml.safe_load(f)
    fp = data.get('controller_server', {}).get('ros__parameters', {}).get('FollowPath', {})
    topics = fp.get('tgmppi_spacetime_obstacle_topics')
    if not topics:
        return src_path, []
    fp['tgmppi_spacetime_obstacle_topics'] = [
        t.replace('/ground_truth/odom', '/tracked/odom') for t in topics]
    out = tempfile.NamedTemporaryFile(mode='w', suffix='_tracking.yaml', delete=False)
    yaml.safe_dump(data, out)
    out.close()
    return out.name, ['%d obstacle topic(s) -> /tracked/odom' % len(topics)]


def dynamic_obstacle_params(src_path):
    """Return (new_params_path, list_of_changes) with the dynamic-obstacle edits applied."""
    with open(src_path) as f:
        data = yaml.safe_load(f)
    changes = []
    for cm, stvl in (('local_costmap', 'stvl_local'), ('global_costmap', 'stvl_global')):
        params = data.get(cm, {}).get(cm, {}).get('ros__parameters', {})
        plugins = params.get('plugins', [])
        if stvl in plugins:
            params['plugins'] = [x for x in plugins if x != stvl]
            changes.append(f'{cm}: removed {stvl}')
        front = params.get('obstacle_layer', {}).get('front_scan')
        if isinstance(front, dict) and front.get('topic') != '/front_scan_filtered':
            front['topic'] = '/front_scan_filtered'
            changes.append(f'{cm}: obstacle_layer front_scan -> /front_scan_filtered')
    follow = data.get('controller_server', {}).get('ros__parameters', {}).get('FollowPath', {})
    if follow.get('flow_assist_only_when_path_blocked') is True:
        follow['flow_assist_only_when_path_blocked'] = False
        changes.append('FollowPath: flow_assist_only_when_path_blocked -> false')
    out = tempfile.NamedTemporaryFile(mode='w', suffix='_dynamic_obstacles.yaml', delete=False)
    yaml.safe_dump(data, out)
    out.close()
    return out.name, changes


# max_speed:=<m/s> (2026-09-16): raising vx_max alone is not enough. The MPPI
# rollout horizon (time_steps * model_dt) is a TIME, so it becomes a DISTANCE of
# vx_max*horizon, and anything shorter than that distance silently truncates the
# lookahead: the local costmap window, the flow field's flood radius, the
# dynamic-obstacle cull distance, and the space-time search's own grid speed.
# (vx_max was bumped 0.35 -> 0.5 on 2026-09-14 without these and had to be
# reverted.) Applies to BOTH controllers -- a vehicle speed limit is shared
# environment, not a method change -- unlike dynamic_obstacles below.
def speed_params(src_path, vmax):
    """Return (new_params_path, changes) with vx_max and everything that scales with it."""
    with open(src_path) as f:
        data = yaml.safe_load(f)
    fp = data.get('controller_server', {}).get('ros__parameters', {}).get('FollowPath', {})
    if not fp:
        return src_path, []
    changes = []
    horizon = float(fp.get('time_steps', 56)) * float(fp.get('model_dt', 0.05))
    reach = vmax * horizon
    fp['vx_max'] = vmax
    fp['vx_min'] = -min(0.5, vmax)         # reverse stays modest at speed
    fp['vx_std'] = round(0.33 * vmax, 3)   # explore ~1/3 of the range
    changes.append('vx_max %.2f, vx_min %.2f, vx_std %.2f (horizon %.1f s = %.1f m)'
                   % (vmax, fp['vx_min'], fp['vx_std'], horizon, reach))
    size = 2 * math.ceil(reach + 1.5)
    lc = data.get('local_costmap', {}).get('local_costmap', {}).get('ros__parameters', {})
    if lc and size > float(lc.get('width', 0)):
        lc['width'] = size
        lc['height'] = size
        if size > 8:
            lc['resolution'] = 0.05        # keep the cell count sane at a bigger window
        changes.append('local_costmap %dx%d m @ %.3f m' % (size, size, lc['resolution']))
    # A rolling local costmap with no static layer knows only what the FRONT
    # lidar saw this cycle. At 0.35 m/s the 2.5 m obstacle range covered 7 s of
    # travel; at 1.5 m/s it covers 1.7 s while the rollout reaches 4.2 m, and the
    # room walls were never in the local costmap at all -- so the robot drove
    # into walls it had not sensed (2026-09-16). Both fixes are shared
    # environment, so they apply to the baseline too.
    if lc:
        plugins = list(lc.get('plugins', []))
        if 'static_layer' not in plugins:
            gc = data.get('global_costmap', {}).get('global_costmap', {}).get('ros__parameters', {})
            lc['static_layer'] = dict(gc.get('static_layer') or
                                      {'plugin': 'nav2_costmap_2d::StaticLayer',
                                       'map_subscribe_transient_local': True})
            lc['plugins'] = ['static_layer'] + plugins
            changes.append('local_costmap += static_layer (walls no longer depend on the '
                           'front lidar having just seen them)')
        obstacle_range = round(min(10.0, reach + 1.5), 1)
        ol = lc.get('obstacle_layer', {})
        touched = []
        for src in ('front_scan', 'rear_scan'):
            blk = ol.get(src)
            if isinstance(blk, dict) and blk.get('obstacle_max_range', 0) < obstacle_range:
                blk['obstacle_max_range'] = obstacle_range
                blk['raytrace_max_range'] = round(obstacle_range + 1.0, 1)
                touched.append(src)
        if touched:
            changes.append('obstacle_max_range %.1f m / raytrace %.1f m on %s (lidar reaches '
                           '12 m; the rollout reaches %.1f m)'
                           % (obstacle_range, obstacle_range + 1.0, '+'.join(touched), reach))

    if 'tgmppi_body_radius' in fp:         # TG-MPPI only; absent in the stock baseline
        fp['tgmppi_body_radius'] = round(min(reach, size / 2.0 - 0.5), 2)
        changes.append('tgmppi_body_radius %.2f m' % fp['tgmppi_body_radius'])
    doc = fp.get('DynamicObstacleCritic')
    if isinstance(doc, dict):
        doc['cull_distance'] = round((vmax + 1.0) * horizon + 0.9, 1)
        # Check spacing must stay well inside an obstacle radius, or a crossing
        # obstacle passes BETWEEN two checked rollout points: at 1.5 m/s robot +
        # 1.0 m/s obstacle, point_step 3 spaced the checks 0.375 m apart against
        # a 0.25 m obstacle (collisions observed 2026-09-16).
        v_rel = vmax + 1.0
        doc['trajectory_point_step'] = max(1, int(0.15 / (float(fp.get('model_dt', 0.05)) * v_rel)))
        # Soft band must at least cover the braking distance v^2/2a; below that the
        # cost only starts once stopping is already impossible.
        decel = abs(float(fp.get('ax_min', -3.0))) or 3.0
        doc['soft_distance'] = max(0.15, round(vmax * vmax / (2.0 * decel), 2))
        changes.append('cull_distance %.1f m, trajectory_point_step %d (%.2f m spacing), '
                       'soft_distance %.2f m (braking %.2f m)'
                       % (doc['cull_distance'], doc['trajectory_point_step'],
                          doc['trajectory_point_step'] * float(fp.get('model_dt', 0.05)) * v_rel,
                          doc['soft_distance'], vmax * vmax / (2.0 * decel)))
    if 'tgmppi_spacetime_dt_layer' in fp:
        fp['tgmppi_spacetime_dt_layer'] = 0.20
        fp['tgmppi_spacetime_res'] = round(vmax * 0.20, 2)   # search speed == vx_max
        fp['tgmppi_spacetime_horizon'] = 3.0                 # CV error grows as t^2
        fp['tgmppi_spacetime_window'] = 3.0
        changes.append('space-time res %.2f m / dt 0.20 s (search speed %.2f m/s), '
                       'horizon 3.0 s, detection reach %.1f m'
                       % (fp['tgmppi_spacetime_res'], vmax, vmax * 3.0))
    out = tempfile.NamedTemporaryFile(mode='w', suffix='_max_speed.yaml', delete=False)
    yaml.safe_dump(data, out)
    out.close()
    return out.name, changes


def launch_nav2(context, nav2_launch_path, configured_params):
    params_path = configured_params.perform(context)
    actions = []
    ablation = LaunchConfiguration('ablation').perform(context).strip().lower()
    if ablation in ABLATIONS:
        params_path, achanges = ablation_params(params_path, ablation)
        if achanges:
            actions.append(LogInfo(msg='[ablation] %s: %s' % (ablation, '; '.join(achanges))))
    else:
        actions.append(LogInfo(msg="[ablation] unknown '%s' (choices: %s) -- running full"
                                   % (ablation, ', '.join(ABLATIONS))))
    backend = LaunchConfiguration('backend').perform(context).strip().lower()
    if backend in ('cpu', 'cuda'):
        params_path, bchanges = backend_params(params_path, backend)
        if bchanges:
            actions.append(LogInfo(msg='[backend] ' + '; '.join(bchanges)))
    elif backend:
        actions.append(LogInfo(msg="[backend] ignored: '%s' is not cpu or cuda" % backend))
    max_speed = LaunchConfiguration('max_speed').perform(context).strip()
    if max_speed:
        params_path, speed_changes = speed_params(params_path, float(max_speed))
        actions.append(LogInfo(msg='[max_speed] ' + '; '.join(speed_changes)))
    dynamic = (LaunchConfiguration('dynamic_obstacles').perform(context).lower() == 'true' and
               LaunchConfiguration('sim').perform(context).lower() == 'true')
    if dynamic:
        with open(params_path) as f:
            follow = (yaml.safe_load(f).get('controller_server', {})
                      .get('ros__parameters', {}).get('FollowPath', {}))
        if 'TgMppiController' not in str(follow.get('plugin', '')):
            # The baseline is stock Nav2 MPPI and must run unmodified: removing moving
            # obstacles from its costmap without prediction would blind it.
            actions.append(LogInfo(msg='[dynamic_obstacles] ignored: params are not TG-MPPI '
                                       '(the stock Nav2 baseline always runs unmodified)'))
            dynamic = False
    if dynamic:
        params_path, changes = dynamic_obstacle_params(params_path)
        actions.append(LogInfo(msg='[dynamic_obstacles] ' + '; '.join(changes)))
        quality = LaunchConfiguration('tracking_quality').perform(context).strip().lower()
        preset = TRACKING_PRESETS.get(quality, None)
        if quality not in TRACKING_PRESETS:
            actions.append(LogInfo(msg="[tracking_quality] unknown preset '%s', using "
                                       "'perfect' (choices: %s)"
                                       % (quality, ', '.join(TRACKING_PRESETS))))
        scan_filter_cmd = ['python3', SCAN_FILTER, '--ros-args', '-p', 'use_sim_time:=true']
        if preset:
            params_path, tchanges = tracking_params(params_path)
            noise_args = ['python3', TRACKING_NOISE, '--ros-args', '-p', 'use_sim_time:=true']
            for k, v in preset.items():
                noise_args += ['-p', '%s:=%s' % (k, v)]
            actions.append(ExecuteProcess(
                cmd=noise_args, name='obstacle_tracking_noise', output='screen'))
            # the costmap must be cleared with the same degraded states, not ground truth
            scan_filter_cmd += ['-p',
                                'odom_pattern:=^/moving_obstacle_[^/]+/tracked/odom$']
            actions.append(LogInfo(msg='[tracking_quality] %s: %s; %s'
                                       % (quality, preset, '; '.join(tchanges))))
        actions.append(ExecuteProcess(
            cmd=scan_filter_cmd, name='tracked_obstacle_scan_filter', output='screen'))
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_path),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'use_sim_time': LaunchConfiguration('sim'),
            'params_file': params_path,
        }.items()))
    return actions


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
        [FindPackageShare('susag_nav2'), 'param', 'navigation_tgmppi_tight.yaml']
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

        DeclareLaunchArgument(
            name='ablation',
            default_value='full',
            description="TG-MPPI ablation: 'full', 'no_spacetime', or 'no_topology' "
                        "(plain MPPI + predicted-obstacle critic). No effect on the stock baseline."
        ),

        DeclareLaunchArgument(
            name='backend',
            default_value='',
            description="Force TG-MPPI compute_backend: 'cpu' makes both controllers run "
                        "the identical shared-critic CPU code (clean method comparison), "
                        "'cuda' forces the GPU path. Empty = use the YAML value."
        ),

        DeclareLaunchArgument(
            name='tracking_quality',
            default_value='perfect',
            description="Obstacle-state quality with dynamic_obstacles:=true: 'perfect' "
                        "(ground truth, the reference), 'good' or 'poor' (noise, latency, "
                        "dropout, lower rate) -- measures how good a tracker must be."
        ),

        DeclareLaunchArgument(
            name='max_speed',
            default_value='',
            description='Override vx_max (m/s) and rescale every speed-dependent param '
                        '(costmap window, flood radius, cull distance, space-time grid). '
                        'Empty = use the YAML as-is, e.g. for narrow BARN worlds.'
        ),

        DeclareLaunchArgument(
            name='dynamic_obstacles',
            default_value='false',
            description='Sim only: predicted-obstacle mode (scan filter, no STVL, always-on TG-MPPI branches)'
        ),

        OpaqueFunction(function=lambda context: launch_nav2(
            context, nav2_launch_path, configured_params)),

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
