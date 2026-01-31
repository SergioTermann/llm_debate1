"""
生成更复杂的多无人机协同轨迹数据
包含：多层编队、协同避障、分布式搜索、动态角色分配、协同攻击/防御
"""

import json
import math
import random
import numpy as np

def generate_complex_trajectory(base_lat=0.0, base_lon=0.0, base_alt=50.0, 
                           base_heading=90.0, base_speed=20.0,
                           num_points=100, complexity='high',
                           waypoints=None, formation_pattern=None,
                           gps_failure=False, signal_loss=False):
    """生成单个无人机的复杂轨迹"""
    
    trajectory = []
    lat, lon, alt, heading, speed = base_lat, base_lon, base_alt, base_heading, base_speed
    
    # GPS故障参数
    gps_drift_lat = 0.0
    gps_drift_lon = 0.0
    
    # 如果有航点，则跟踪航点
    if waypoints:
        current_waypoint_idx = 0
        waypoint_target = waypoints[current_waypoint_idx]
    
    for i in range(num_points):
        # 根据复杂度添加变化
        if complexity == 'high':
            # 高复杂度：频繁的航向变化、高度变化、速度变化
            if i % 12 == 0:
                # 急转弯
                heading += random.choice([45, -45, 60, -60, 90, -90])
                speed *= random.uniform(0.6, 1.4)
            elif i % 8 == 0:
                # 高度变化
                alt += random.choice([8, -8, 12, -5, 10])
            elif i % 6 == 0:
                # 速度变化
                speed += random.choice([-8, 5, -10, 7])
                speed = max(5, min(40, speed))
            
            # 添加随机噪声
            heading += random.uniform(-8, 8)
            alt += random.uniform(-2, 2)
            
        elif complexity == 'medium':
            # 中等复杂度：适中的变化
            if i % 18 == 0:
                heading += random.choice([30, -30, 45, -45])
            elif i % 12 == 0:
                alt += random.choice([5, -5, 8])
            elif i % 10 == 0:
                speed += random.choice([-4, 3, -5])
                speed = max(10, min(35, speed))
            
            heading += random.uniform(-5, 5)
            alt += random.uniform(-1, 1)
            
        else:  # low
            # 低复杂度：平稳飞行
            heading += random.uniform(-2, 2)
            alt += random.uniform(-0.5, 0.5)
            speed += random.uniform(-1.5, 1.5)
            speed = max(15, min(25, speed))
        
        # 如果有航点，调整航向指向航点
        if waypoints and current_waypoint_idx < len(waypoints):
            target_lat, target_lon, target_alt = waypoint_target
            dx = target_lat - lat
            dy = target_lon - lon
            target_heading = math.degrees(math.atan2(dy, dx))
            if target_heading < 0:
                target_heading += 360
            
            # 平滑转向航点
            heading_diff = target_heading - heading
            if heading_diff > 180:
                heading_diff -= 360
            elif heading_diff < -180:
                heading_diff += 360
            
            heading += heading_diff * 0.3
            
            # 检查是否到达航点
            distance = math.sqrt(dx**2 + dy**2)
            if distance < 0.0005:
                current_waypoint_idx += 1
                if current_waypoint_idx < len(waypoints):
                    waypoint_target = waypoints[current_waypoint_idx]
        
        # 更新位置
        lat += math.cos(math.radians(heading)) * speed * 0.0001
        lon += math.sin(math.radians(heading)) * speed * 0.0001
        
        # GPS故障处理：添加漂移
        if gps_failure and i > 50 and i < 100:
            # GPS漂移逐渐增大
            gps_drift_lat += random.uniform(-0.00005, 0.00005)
            gps_drift_lon += random.uniform(-0.00005, 0.00005)
            lat += gps_drift_lat
            lon += gps_drift_lon
        
        # 信号丢失处理：添加数据缺失
        if signal_loss and i > 30 and i < 70:
            # 30%的概率数据缺失
            if random.random() < 0.3:
                # 标记为信号丢失
                trajectory.append({
                    'time': i,
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt,
                    'heading': heading % 360,
                    'speed': speed,
                    'signal_status': 'LOST'
                })
                continue
        
        # 确保高度在合理范围
        alt = max(10, min(200, alt))
        
        trajectory.append({
            'time': i,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'heading': heading % 360,
            'speed': speed,
            'signal_status': 'OK'
        })
    
    return trajectory

def generate_multi_layer_formation(num_drones=10, base_lat=0.0, base_lon=0.0, 
                                  base_alt=50.0, num_points=150, complexity='high'):
    """生成多层编队（3层：前锋、中锋、后卫）"""
    
    drones = []
    
    # 根据无人机数量动态分配层
    if num_drones <= 10:
        layers = {
            'forward': list(range(0, min(3, num_drones))),
            'middle': list(range(3, min(7, num_drones))),
            'rear': list(range(7, num_drones))
        }
    else:
        # 对于更多无人机，扩展每一层
        forward_count = num_drones // 4
        middle_count = num_drones // 2
        rear_count = num_drones - forward_count - middle_count
        
        layers = {
            'forward': list(range(0, forward_count)),
            'middle': list(range(forward_count, forward_count + middle_count)),
            'rear': list(range(forward_count + middle_count, num_drones))
        }
    
    layer_configs = {
        'forward': {'alt_offset': 20, 'speed_mult': 1.2, 'heading_offset': 0},
        'middle': {'alt_offset': 0, 'speed_mult': 1.0, 'heading_offset': 0},
        'rear': {'alt_offset': -20, 'speed_mult': 0.9, 'heading_offset': 0}
    }
    
    for drone_idx in range(num_drones):
        # 确定无人机所属层
        layer = None
        for layer_name, indices in layers.items():
            if drone_idx in indices:
                layer = layer_name
                break
        
        # 如果没有分配到层，默认分配到middle层
        if layer is None:
            layer = 'middle'
        
        config = layer_configs[layer]
        
        # 计算编队位置
        layer_drones = layers[layer]
        position_in_layer = layer_drones.index(drone_idx) if drone_idx in layer_drones else 0
        total_in_layer = len(layer_drones)
        
        # 在层内形成V形或一字形
        if layer == 'forward':
            offset_lat = (position_in_layer - 1) * 0.00015
            offset_lon = (position_in_layer - 1) * 0.00015
        elif layer == 'middle':
            offset_lat = (position_in_layer - 1.5) * 0.0002
            offset_lon = (position_in_layer - 1.5) * 0.0002
        else:  # rear
            offset_lat = (position_in_layer - 1) * 0.00015
            offset_lon = (position_in_layer - 1) * 0.00015
        
        trajectory = generate_complex_trajectory(
            base_lat + offset_lat,
            base_lon + offset_lon,
            base_alt + config['alt_offset'],
            90.0 + config['heading_offset'],
            20 * config['speed_mult'],
            num_points,
            complexity
        )
        
        drones.append({
            'drone_id': f'UAV_{drone_idx + 1}',
            'layer': layer,
            'role': f'{layer}_fighter',
            'trajectory': trajectory
        })
    
    return drones

def generate_collaborative_search(num_drones=12, base_lat=0.0, base_lon=0.0, 
                                  base_alt=50.0, num_points=150, complexity='high'):
    """生成协同搜索任务（分布式搜索区域）"""
    
    drones = []
    
    # 将无人机分成3个搜索小组
    groups = {
        'alpha': list(range(0, 4)),
        'bravo': list(range(4, 8)),
        'charlie': list(range(8, 12))
    }
    
    # 每个小组的搜索区域
    search_areas = {
        'alpha': {'center_lat': 0.0, 'center_lon': 0.0, 'radius': 0.005},
        'bravo': {'center_lat': 0.005, 'center_lon': 0.005, 'radius': 0.005},
        'charlie': {'center_lat': -0.005, 'center_lon': 0.005, 'radius': 0.005}
    }
    
    for drone_idx in range(num_drones):
        # 确定无人机所属小组
        group = None
        for group_name, indices in groups.items():
            if drone_idx in indices:
                group = group_name
                break
        
        area = search_areas[group]
        group_drones = groups[group]
        position_in_group = group_drones.index(drone_idx)
        
        # 生成螺旋搜索路径
        waypoints = []
        num_loops = 3
        points_per_loop = 4
        
        for loop in range(num_loops):
            radius = (loop + 1) * area['radius'] / num_loops
            for pt in range(points_per_loop):
                angle = (pt / points_per_loop) * 2 * math.pi + (position_in_group * math.pi / 2)
                lat = area['center_lat'] + radius * math.cos(angle)
                lon = area['center_lon'] + radius * math.sin(angle)
                alt = base_alt + loop * 5
                waypoints.append((lat, lon, alt))
        
        # 每个无人机有不同的起始偏移
        offset_lat = (position_in_group - 1.5) * 0.0001
        offset_lon = (position_in_group - 1.5) * 0.0001
        
        trajectory = generate_complex_trajectory(
            base_lat + offset_lat,
            base_lon + offset_lon,
            base_alt,
            90.0,
            18,
            num_points,
            complexity,
            waypoints=waypoints
        )
        
        drones.append({
            'drone_id': f'UAV_{drone_idx + 1}',
            'group': group,
            'role': f'{group}_searcher',
            'trajectory': trajectory
        })
    
    return drones

def generate_collaborative_attack(num_drones=15, base_lat=0.0, base_lon=0.0, 
                                  base_alt=50.0, num_points=150, complexity='high'):
    """生成协同攻击任务（多波次攻击）"""
    
    drones = []
    
    # 3波攻击
    waves = {
        'first': list(range(0, 5)),      # 第一波：侦察和干扰
        'second': list(range(5, 10)),    # 第二波：主攻
        'third': list(range(10, 15))     # 第三波：支援和掩护
    }
    
    wave_configs = {
        'first': {'alt_offset': 30, 'speed_mult': 1.3, 'heading_offset': 0, 'start_delay': 0},
        'second': {'alt_offset': 0, 'speed_mult': 1.1, 'heading_offset': 0, 'start_delay': 20},
        'third': {'alt_offset': -20, 'speed_mult': 0.9, 'heading_offset': 0, 'start_delay': 40}
    }
    
    for drone_idx in range(num_drones):
        # 确定无人机所属波次
        wave = None
        for wave_name, indices in waves.items():
            if drone_idx in indices:
                wave = wave_name
                break
        
        config = wave_configs[wave]
        wave_drones = waves[wave]
        position_in_wave = wave_drones.index(drone_idx)
        
        # 编队位置
        offset_lat = (position_in_wave - 2) * 0.00015
        offset_lon = (position_in_wave - 2) * 0.00015
        
        trajectory = generate_complex_trajectory(
            base_lat + offset_lat,
            base_lon + offset_lon,
            base_alt + config['alt_offset'],
            90.0 + config['heading_offset'],
            22 * config['speed_mult'],
            num_points,
            complexity
        )
        
        # 添加延迟效果（前几帧保持原位）
        delay = config['start_delay']
        for i in range(delay):
            trajectory[i]['latitude'] = base_lat + offset_lat
            trajectory[i]['longitude'] = base_lon + offset_lon
            trajectory[i]['speed'] = 0
        
        drones.append({
            'drone_id': f'UAV_{drone_idx + 1}',
            'wave': wave,
            'role': f'{wave}_attack',
            'trajectory': trajectory
        })
    
    return drones

def generate_collaborative_defense(num_drones=12, base_lat=0.0, base_lon=0.0, 
                                  base_alt=50.0, num_points=150, complexity='high'):
    """生成协同防御任务（环形防御）"""
    
    drones = []
    
    # 内外两层防御圈
    layers = {
        'inner': list(range(0, 6)),      # 内层：近距离防御
        'outer': list(range(6, 12))      # 外层：远程防御
    }
    
    layer_configs = {
        'inner': {'radius': 0.003, 'alt_offset': 10, 'speed_mult': 1.2},
        'outer': {'radius': 0.006, 'alt_offset': -10, 'speed_mult': 0.8}
    }
    
    for drone_idx in range(num_drones):
        # 确定无人机所属层
        layer = None
        for layer_name, indices in layers.items():
            if drone_idx in indices:
                layer = layer_name
                break
        
        config = layer_configs[layer]
        layer_drones = layers[layer]
        position_in_layer = layer_drones.index(drone_idx)
        
        # 环形编队
        angle = (position_in_layer / len(layer_drones)) * 2 * math.pi
        offset_lat = config['radius'] * math.cos(angle)
        offset_lon = config['radius'] * math.sin(angle)
        
        trajectory = generate_complex_trajectory(
            base_lat + offset_lat,
            base_lon + offset_lon,
            base_alt + config['alt_offset'],
            90.0,
            18 * config['speed_mult'],
            num_points,
            complexity
        )
        
        # 添加环形运动
        for i in range(num_points):
            current_angle = angle + (i / num_points) * 2 * math.pi
            trajectory[i]['latitude'] = base_lat + config['radius'] * math.cos(current_angle)
            trajectory[i]['longitude'] = base_lon + config['radius'] * math.sin(current_angle)
            trajectory[i]['heading'] = (math.degrees(current_angle) + 90) % 360
        
        drones.append({
            'drone_id': f'UAV_{drone_idx + 1}',
            'layer': layer,
            'role': f'{layer}_defender',
            'trajectory': trajectory
        })
    
    return drones

def generate_dynamic_role_assignment(num_drones=15, base_lat=0.0, base_lon=0.0, 
                                    base_alt=50.0, num_points=150, complexity='high'):
    """生成动态角色分配任务（任务中角色切换）"""
    
    drones = []
    
    # 初始角色
    initial_roles = {
        'scout': list(range(0, 3)),        # 侦察
        'escort': list(range(3, 8)),       # 护航
        'attack': list(range(8, 12)),      # 攻击
        'support': list(range(12, 15))      # 支援
    }
    
    for drone_idx in range(num_drones):
        # 确定初始角色
        role = None
        for role_name, indices in initial_roles.items():
            if drone_idx in indices:
                role = role_name
                break
        
        role_drones = initial_roles[role]
        position_in_role = role_drones.index(drone_idx)
        
        # 根据角色配置
        role_configs = {
            'scout': {'alt_offset': 40, 'speed_mult': 1.4, 'offset_scale': 0.0003},
            'escort': {'alt_offset': 0, 'speed_mult': 1.0, 'offset_scale': 0.0002},
            'attack': {'alt_offset': -10, 'speed_mult': 1.1, 'offset_scale': 0.00015},
            'support': {'alt_offset': -30, 'speed_mult': 0.8, 'offset_scale': 0.00025}
        }
        
        config = role_configs[role]
        offset_lat = (position_in_role - 1.5) * config['offset_scale']
        offset_lon = (position_in_role - 1.5) * config['offset_scale']
        
        trajectory = generate_complex_trajectory(
            base_lat + offset_lat,
            base_lon + offset_lon,
            base_alt + config['alt_offset'],
            90.0,
            20 * config['speed_mult'],
            num_points,
            complexity
        )
        
        # 动态角色切换（在任务中间切换角色）
        role_switch_point = num_points // 2
        
        # 添加角色切换效果
        if role == 'scout':
            # 侦察转为攻击
            for i in range(role_switch_point, num_points):
                trajectory[i]['altitude'] -= 0.5
                trajectory[i]['speed'] *= 0.95
                trajectory[i]['heading'] += random.uniform(-10, 10)
        elif role == 'escort':
            # 护航转为支援
            for i in range(role_switch_point, num_points):
                trajectory[i]['altitude'] -= 0.3
                trajectory[i]['speed'] *= 0.9
        elif role == 'attack':
            # 攻击转为支援
            for i in range(role_switch_point, num_points):
                trajectory[i]['altitude'] -= 0.4
                trajectory[i]['speed'] *= 0.85
                trajectory[i]['heading'] += random.uniform(-15, 15)
        
        drones.append({
            'drone_id': f'UAV_{drone_idx + 1}',
            'initial_role': role,
            'role_switch_point': role_switch_point,
            'trajectory': trajectory
        })
    
    return drones

def generate_complex_missions():
    """生成所有复杂协同任务"""
    
    missions = []
    
    # 任务配置
    mission_configs = [
        {
            'mission_id': 'COMPLEX_01_MULTI_LAYER_FORMATION',
            'mission_type': 'Multi-Layer Formation',
            'flight_duration': '150s',
            'ground_truth': 'Borderline',
            'complexity': 'high',
            'description': '3层编队（前锋、中锋、后卫），10架无人机协同',
            'generator': generate_multi_layer_formation,
            'params': {'num_drones': 10, 'num_points': 150}
        },
        {
            'mission_id': 'COMPLEX_02_COLLABORATIVE_SEARCH',
            'mission_type': 'Collaborative Search',
            'flight_duration': '150s',
            'ground_truth': 'Safe',
            'complexity': 'high',
            'description': '3个小组分布式搜索，12架无人机协同',
            'generator': generate_collaborative_search,
            'params': {'num_drones': 12, 'num_points': 150}
        },
        {
            'mission_id': 'COMPLEX_03_COLLABORATIVE_ATTACK',
            'mission_type': 'Collaborative Attack',
            'flight_duration': '150s',
            'ground_truth': 'Risky',
            'complexity': 'high',
            'description': '3波次协同攻击，15架无人机',
            'generator': generate_collaborative_attack,
            'params': {'num_drones': 15, 'num_points': 150}
        },
        {
            'mission_id': 'COMPLEX_04_COLLABORATIVE_DEFENSE',
            'mission_type': 'Collaborative Defense',
            'flight_duration': '150s',
            'ground_truth': 'Borderline',
            'complexity': 'high',
            'description': '内外两层环形防御，12架无人机',
            'generator': generate_collaborative_defense,
            'params': {'num_drones': 12, 'num_points': 150}
        },
        {
            'mission_id': 'COMPLEX_05_DYNAMIC_ROLE_ASSIGNMENT',
            'mission_type': 'Dynamic Role Assignment',
            'flight_duration': '150s',
            'ground_truth': 'Risky',
            'complexity': 'high',
            'description': '动态角色切换（侦察→攻击→支援），15架无人机',
            'generator': generate_dynamic_role_assignment,
            'params': {'num_drones': 15, 'num_points': 150}
        },
        {
            'mission_id': 'COMPLEX_06_FORMATION_BREAK_COLLISION',
            'mission_type': 'Formation Break & Collision Risk',
            'flight_duration': '150s',
            'ground_truth': 'Risky',
            'complexity': 'high',
            'description': '编队破裂导致碰撞风险，10架无人机',
            'generator': generate_multi_layer_formation,
            'params': {'num_drones': 10, 'num_points': 150}
        },
        {
            'mission_id': 'COMPLEX_07_GPS_GLITCH',
            'mission_type': 'GPS Glitch (Unobservable)',
            'flight_duration': '150s',
            'ground_truth': 'Risky',
            'complexity': 'high',
            'description': 'GPS故障导致位置漂移，部分无人机轨迹不可观测，12架无人机',
            'generator': generate_multi_layer_formation,
            'params': {'num_drones': 12, 'num_points': 150, 'gps_failure': True}
        },
        {
            'mission_id': 'COMPLEX_08_SIGNAL_LOSS',
            'mission_type': 'Signal Loss (Unobservable)',
            'flight_duration': '150s',
            'ground_truth': 'Risky',
            'complexity': 'high',
            'description': '通信信号丢失导致数据缺失，10架无人机',
            'generator': generate_collaborative_search,
            'params': {'num_drones': 10, 'num_points': 150, 'signal_loss': True}
        },
        {
            'mission_id': 'COMPLEX_09_EMERGENCY_EVASION',
            'mission_type': 'Emergency Evasion',
            'flight_duration': '150s',
            'ground_truth': 'Risky',
            'complexity': 'high',
            'description': '紧急规避机动，12架无人机协同',
            'generator': generate_collaborative_defense,
            'params': {'num_drones': 12, 'num_points': 150}
        },
        {
            'mission_id': 'COMPLEX_10_COORDINATED_LANDING',
            'mission_type': 'Coordinated Landing',
            'flight_duration': '150s',
            'ground_truth': 'Borderline',
            'complexity': 'medium',
            'description': '协同降落，15架无人机',
            'generator': generate_collaborative_attack,
            'params': {'num_drones': 15, 'num_points': 150}
        }
    ]
    
    for config in mission_configs:
        # 生成无人机轨迹
        drones = config['generator'](
            num_drones=config['params']['num_drones'],
            num_points=config['params']['num_points'],
            complexity=config['complexity']
        )
        
        # 特殊处理：编队破裂和碰撞风险
        if 'COLLISION' in config['mission_id']:
            formation_break_point = random.randint(60, 90)
            for drone in drones:
                trajectory = drone['trajectory']
                for i in range(formation_break_point, len(trajectory)):
                    # 无人机向中心聚集，增加碰撞风险
                    factor = 1.0 - (i - formation_break_point) / (len(trajectory) - formation_break_point) * 0.5
                    trajectory[i]['latitude'] = trajectory[i]['latitude'] * factor
                    trajectory[i]['longitude'] = trajectory[i]['longitude'] * factor
                    # 添加随机抖动
                    trajectory[i]['heading'] += random.uniform(-20, 20)
        
        # 特殊处理：GPS故障
        elif 'GPS_GLITCH' in config['mission_id']:
            # 部分无人机GPS故障
            affected_drones = random.sample(range(len(drones)), k=len(drones) // 2)
            for drone_idx in affected_drones:
                drone = drones[drone_idx]
                trajectory = drone['trajectory']
                # 从第50点开始GPS漂移
                for i in range(50, 100):
                    # 添加逐渐增大的漂移
                    drift_factor = (i - 50) / 50
                    trajectory[i]['latitude'] += random.uniform(-0.0001, 0.0001) * drift_factor
                    trajectory[i]['longitude'] += random.uniform(-0.0001, 0.0001) * drift_factor
                    # 标记为GPS故障
                    trajectory[i]['gps_status'] = 'DRIFT'
        
        # 特殊处理：信号丢失
        elif 'SIGNAL_LOSS' in config['mission_id']:
            # 部分无人机信号丢失
            affected_drones = random.sample(range(len(drones)), k=len(drones) // 2)
            for drone_idx in affected_drones:
                drone = drones[drone_idx]
                trajectory = drone['trajectory']
                # 从第30点开始信号丢失
                for i in range(30, 70):
                    # 40%的概率数据完全丢失
                    if random.random() < 0.4:
                        trajectory[i]['signal_status'] = 'LOST'
                        # 添加大误差
                        trajectory[i]['latitude'] += random.uniform(-0.0005, 0.0005)
                        trajectory[i]['longitude'] += random.uniform(-0.0005, 0.0005)
                        trajectory[i]['altitude'] += random.uniform(-20, 20)
        
        # 特殊处理：紧急规避
        elif 'EVASION' in config['mission_id']:
            evasion_point = random.randint(40, 60)
            for drone in drones:
                trajectory = drone['trajectory']
                for i in range(evasion_point, min(evasion_point + 20, len(trajectory))):
                    # 剧烈的航向和高度变化
                    trajectory[i]['heading'] += random.choice([60, -60, 90, -90])
                    trajectory[i]['altitude'] += random.choice([15, -15, 20, -20])
                    trajectory[i]['speed'] *= random.uniform(0.7, 1.3)
        
        # 特殊处理：协同降落
        elif 'LANDING' in config['mission_id']:
            landing_start = random.randint(80, 100)
            for drone in drones:
                trajectory = drone['trajectory']
                for i in range(landing_start, len(trajectory)):
                    # 高度急剧下降
                    trajectory[i]['altitude'] -= random.uniform(3, 6)
                    trajectory[i]['speed'] *= 0.85
                    # 添加随机抖动模拟降落不稳定性
                    trajectory[i]['heading'] += random.uniform(-10, 10)
        
        missions.append({
            'mission_id': config['mission_id'],
            'mission_type': config['mission_type'],
            'flight_duration': config['flight_duration'],
            'ground_truth': config['ground_truth'],
            'description': config['description'],
            'num_drones': len(drones),
            'drones': drones
        })
    
    return missions

def main():
    """主函数：生成复杂协同轨迹数据集"""
    
    print("生成复杂多无人机协同轨迹数据集...")
    print("="*80)
    
    # 生成所有复杂任务
    missions = generate_complex_missions()
    
    # 保存到JSON文件
    output_path = 'complex_uav_missions.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'missions': missions}, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 生成了 {len(missions)} 个复杂协同任务")
    print(f"✓ 保存到: {output_path}")
    
    # 统计信息
    print("\n数据集统计:")
    print("-"*80)
    
    safe_count = sum(1 for m in missions if m['ground_truth'] == 'Safe')
    borderline_count = sum(1 for m in missions if m['ground_truth'] == 'Borderline')
    risky_count = sum(1 for m in missions if m['ground_truth'] == 'Risky')
    
    print(f"  Safe任务: {safe_count}")
    print(f"  Borderline任务: {borderline_count}")
    print(f"  Risky任务: {risky_count}")
    
    # 计算平均复杂度指标
    total_points = 0
    total_drones = 0
    total_heading_changes = 0
    total_alt_changes = 0
    
    for mission in missions:
        total_drones += mission['num_drones']
        for drone in mission['drones']:
            trajectory = drone['trajectory']
            total_points += len(trajectory)
            
            # 计算航向变化
            for i in range(1, len(trajectory)):
                heading_change = abs(trajectory[i]['heading'] - trajectory[i-1]['heading'])
                if heading_change > 180:
                    heading_change = 360 - heading_change
                total_heading_changes += heading_change
            
            # 计算高度变化
            for i in range(1, len(trajectory)):
                alt_change = abs(trajectory[i]['altitude'] - trajectory[i-1]['altitude'])
                total_alt_changes += alt_change
    
    avg_heading_change = total_heading_changes / total_points
    avg_alt_change = total_alt_changes / total_points
    avg_drones_per_mission = total_drones / len(missions)
    
    print(f"\n平均复杂度指标:")
    print(f"  平均无人机数: {avg_drones_per_mission:.1f}")
    print(f"  平均航向变化: {avg_heading_change:.2f}°")
    print(f"  平均高度变化: {avg_alt_change:.2f}m")
    print(f"  总轨迹点数: {total_points}")
    print(f"  每个任务平均点数: {total_points / len(missions):.0f}")
    
    # 显示任务详情
    print("\n任务详情:")
    print("-"*80)
    for mission in missions:
        print(f"  {mission['mission_id']}:")
        print(f"    - 类型: {mission['mission_type']}")
        print(f"    - 无人机数: {mission['num_drones']}")
        print(f"    - Ground Truth: {mission['ground_truth']}")
        print(f"    - 描述: {mission['description']}")
    
    print("\n" + "="*80)
    print("✓ 复杂数据集生成完成！")
    print("\n使用方法:")
    print("  python exp1_real_evaluation.py")
    print(f"  (会自动加载: {output_path})")

if __name__ == "__main__":
    main()
