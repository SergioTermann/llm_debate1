"""
生成改进的UAV数据集，修改角度参数，使轨迹更加多样化和复杂化
修复heading连续性问题
"""

import json
import random
import math
from typing import List, Dict


def generate_trajectory_point(t: float, base_lat: float, base_lon: float, 
                            pattern: str, mission_type: str, 
                            drone_idx: int, total_drones: int,
                            prev_heading: float = None) -> Dict:
    """生成轨迹点，改进角度参数，保持heading连续性"""
    
    # 如果有前一个heading，基于它进行小幅度变化
    if prev_heading is not None:
        base_heading = prev_heading
        # 根据任务类型调整角度变化
        if mission_type == "Surveillance":
            heading_change = random.uniform(-10, 10)
        elif mission_type == "Search-Rescue":
            heading_change = random.uniform(-30, 30)
        elif mission_type == "Formation":
            heading_change = random.uniform(-3, 3)
        elif mission_type == "Adversarial":
            heading_change = random.uniform(-60, 60)
        else:
            heading_change = random.uniform(-20, 20)
        
        heading = (base_heading + heading_change) % 360
    else:
        # 第一个点，随机初始heading
        heading = random.uniform(0, 360)
    
    # 速度变化（基于前一个速度进行小幅度变化）
    base_speed = random.uniform(15, 35)
    speed = base_speed
    
    # 高度变化（基于前一个高度进行小幅度变化）
    base_altitude = random.uniform(80, 150)
    altitude = base_altitude
    
    # 位置变化（根据heading计算）
    heading_rad = math.radians(heading)
    lat_offset = (speed * math.cos(heading_rad) * t) * 0.00001
    lon_offset = (speed * math.sin(heading_rad) * t) * 0.00001
    
    # 添加编队偏移
    if pattern == "formation":
        formation_offset = (drone_idx - total_drones / 2) * 0.0001
        lat_offset += formation_offset * math.cos(heading_rad)
        lon_offset += formation_offset * math.sin(heading_rad)
    
    return {
        "timestamp": t,
        "gps": {
            "lat": base_lat + lat_offset,
            "lon": base_lon + lon_offset
        },
        "altitude": altitude,
        "speed": speed,
        "heading": heading
    }


def generate_drone_trajectory(drone_id: str, mission_type: str, 
                              pattern: str, drone_idx: int, 
                              total_drones: int, num_points: int = 20) -> List[Dict]:
    """生成单个无人机的轨迹"""
    
    base_lat = random.uniform(-0.001, 0.001)
    base_lon = random.uniform(0, 0.002)
    
    trajectory = []
    prev_heading = None
    
    for i in range(num_points):
        t = i * 0.1
        point = generate_trajectory_point(
            t, base_lat, base_lon, pattern, mission_type, 
            drone_idx, total_drones, prev_heading
        )
        trajectory.append(point)
        prev_heading = point['heading']
    
    return trajectory


def generate_mission(mission_id: str, safety_label: str) -> Dict:
    """生成单个任务"""
    
    mission_types = ["Surveillance", "Search-Rescue", "Formation", "Adversarial"]
    patterns = ["linear", "circular", "formation", "random"]
    
    mission_type = random.choice(mission_types)
    pattern = random.choice(patterns)
    drone_count = random.randint(3, 10)
    
    # 生成无人机轨迹
    drones = []
    for i in range(drone_count):
        drone_id = f"DRONE-{i+1:03d}"
        trajectory = generate_drone_trajectory(
            drone_id, mission_type, pattern, i, drone_count
        )
        drones.append({
            "id": drone_id,
            "trajectory": trajectory
        })
    
    # 根据安全标签调整评分
    if safety_label == "Safe":
        total_score = random.uniform(85, 100)
        flight_control_score = random.uniform(85, 100)
        swarm_coordination_score = random.uniform(85, 100)
        safety_score = random.uniform(85, 100)
        anomalies = []
    elif safety_label == "Borderline":
        total_score = random.uniform(60, 84)
        flight_control_score = random.uniform(60, 84)
        swarm_coordination_score = random.uniform(60, 84)
        safety_score = random.uniform(60, 84)
        anomalies = random.sample([
            "Minor altitude deviation",
            "Slight formation drift",
            "Brief communication delay",
            "Small speed fluctuation"
        ], k=random.randint(1, 2))
    else:
        total_score = random.uniform(30, 59)
        flight_control_score = random.uniform(30, 59)
        swarm_coordination_score = random.uniform(30, 59)
        safety_score = random.uniform(30, 59)
        anomalies = random.sample([
            "High collision risk",
            "Emergency response time > 3.0s",
            "Formation stability < 60%",
            "Communication loss",
            "Severe altitude deviation",
            "Unstable trajectory",
            "Excessive speed variation"
        ], k=random.randint(2, 4))
    
    mission = {
        "mission_id": mission_id,
        "mission_type": mission_type,
        "drone_count": drone_count,
        "flight_duration": f"{random.uniform(1.0, 5.0):.1f}h",
        "mission_info": {
            "duration_minutes": int(random.uniform(60, 300)),
            "complexity": random.choice(["low", "medium", "high"]),
            "terrain": random.choice(["Open", "Urban", "Mountainous"])
        },
        "drones": drones,
        "weighted_scores": {
            "total_score": total_score,
            "flight_control": {
                "score": flight_control_score,
                "details": {
                    "trajectory_stability": flight_control_score - random.uniform(0, 5),
                    "altitude_control": flight_control_score - random.uniform(0, 5),
                    "speed_consistency": flight_control_score - random.uniform(0, 5)
                }
            },
            "swarm_coordination": {
                "score": swarm_coordination_score,
                "details": {
                    "formation_maintenance": swarm_coordination_score - random.uniform(0, 5),
                    "communication_latency": swarm_coordination_score - random.uniform(0, 5),
                    "collision_avoidance": swarm_coordination_score - random.uniform(0, 5)
                }
            },
            "safety_assessment": {
                "score": safety_score,
                "details": {
                    "emergency_response": safety_score - random.uniform(0, 5),
                    "fail_safe_mechanisms": safety_score - random.uniform(0, 5),
                    "risk_mitigation": safety_score - random.uniform(0, 5)
                }
            }
        },
        "ground_truth": safety_label,
        "anomalies": anomalies
    }
    
    return mission


def generate_improved_dataset(num_missions: int = 30, output_file: str = "improved_uav_missions.json"):
    """生成改进的数据集"""
    
    print("="*80)
    print("生成改进的UAV数据集")
    print("="*80)
    
    missions_per_class = num_missions // 3
    safety_labels = ["Safe", "Borderline", "Risky"]
    
    missions = []
    
    for safety_label in safety_labels:
        print(f"\n生成 {safety_label} 任务...")
        for i in range(missions_per_class):
            mission_id = f"MISSION-{len(missions)+1:04d}"
            mission = generate_mission(mission_id, safety_label)
            missions.append(mission)
    
    random.shuffle(missions)
    
    output_data = {
        "metadata": {
            "version": "3.0",
            "created_at": "2026-01-24",
            "num_missions": len(missions),
            "description": "Improved UAV mission dataset with enhanced angle parameters and diverse trajectory patterns"
        },
        "missions": missions
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 改进的数据集已保存到: {output_file}")
    print(f"   总任务数: {len(missions)}")
    print(f"   Safe: {missions_per_class}")
    print(f"   Borderline: {missions_per_class}")
    print(f"   Risky: {missions_per_class}")
    print("="*80)
    
    return output_data


if __name__ == "__main__":
    random.seed(42)
    generate_improved_dataset(num_missions=30, output_file="improved_uav_missions.json")
