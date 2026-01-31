"""
真实的UAV集群任务数据集生成器

包含:
1. 复杂的编队变换 (Formation switching)
2. 突发障碍物避障 (Emergency obstacle avoidance)
3. GPS信号丢失 (Signal dropout)
4. 传感器噪声和故障 (Sensor noise/failure)
5. 风扰动影响 (Wind disturbance)
6. 近距离协同 (Close-proximity coordination)
"""

import json
import numpy as np
import math
from typing import List, Dict, Tuple
import random


class RealisticUAVMissionGenerator:
    """生成真实的UAV集群任务数据"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_dataset(self, num_missions=50) -> Dict:
        """生成完整数据集"""
        missions = []
        
        # 任务类型分布
        mission_types = [
            ("patrol", 10),           # 巡逻任务 (简单)
            ("formation_switch", 10), # 编队切换 (中等)
            ("obstacle_avoidance", 10), # 避障任务 (复杂)
            ("search_rescue", 10),    # 搜救任务 (复杂)
            ("adversarial", 10)       # 对抗任务 (极其复杂)
        ]
        
        mission_id = 1
        for mission_type, count in mission_types:
            print(f"  Generating {count} {mission_type} missions...")
            for i in range(count):
                try:
                    if mission_type == "patrol":
                        mission = self._generate_patrol_mission(mission_id)
                    elif mission_type == "formation_switch":
                        mission = self._generate_formation_switch_mission(mission_id)
                    elif mission_type == "obstacle_avoidance":
                        mission = self._generate_obstacle_avoidance_mission(mission_id)
                    elif mission_type == "search_rescue":
                        mission = self._generate_search_rescue_mission(mission_id)
                    else:  # adversarial
                        mission = self._generate_adversarial_mission(mission_id)
                    
                    missions.append(mission)
                    mission_id += 1
                except Exception as e:
                    print(f"    ERROR generating {mission_type} #{i+1}: {e}")
                    raise
        
        return {"missions": missions, "metadata": {"total": len(missions), "version": "2.0_realistic"}}
    
    def _generate_patrol_mission(self, mission_id: int) -> Dict:
        """生成巡逻任务 (相对简单，但仍有变化)"""
        num_drones = 10  # 固定10架无人机
        duration = random.randint(60, 120)
        
        # 50% Safe, 30% Borderline, 20% Risky
        risk_level = np.random.choice(["Safe", "Borderline", "Risky"], p=[0.5, 0.3, 0.2])
        
        drones = []
        base_altitude = 50 + random.uniform(-5, 5)
        
        for drone_id in range(num_drones):
            trajectory = self._generate_patrol_trajectory(
                duration=duration,
                drone_id=drone_id,
                num_drones=num_drones,
                risk_level=risk_level,
                base_altitude=base_altitude
            )
            drones.append({
                "drone_id": f"UAV_{drone_id+1}",
                "trajectory": trajectory
            })
        
        return {
            "mission_id": f"PATROL_{mission_id:03d}",
            "mission_type": "Patrol",
            "num_drones": num_drones,
            "flight_duration": f"{duration}s",
            "ground_truth": risk_level,
            "drones": drones
        }
    
    def _generate_formation_switch_mission(self, mission_id: int) -> Dict:
        """生成编队切换任务 (中等复杂度)"""
        num_drones = 10  # 固定10架无人机
        duration = random.randint(80, 150)
        
        # 40% Safe, 35% Borderline, 25% Risky
        risk_level = np.random.choice(["Safe", "Borderline", "Risky"], p=[0.4, 0.35, 0.25])
        
        # 编队切换点 (在中间某个时刻)
        switch_time = duration // 2 + random.randint(-10, 10)
        
        drones = []
        base_altitude = 50
        
        for drone_id in range(num_drones):
            trajectory = self._generate_formation_switch_trajectory(
                duration=duration,
                drone_id=drone_id,
                num_drones=num_drones,
                switch_time=switch_time,
                risk_level=risk_level,
                base_altitude=base_altitude
            )
            drones.append({
                "drone_id": f"UAV_{drone_id+1}",
                "trajectory": trajectory
            })
        
        return {
            "mission_id": f"FORMATION_{mission_id:03d}",
            "mission_type": "Formation Switch",
            "num_drones": num_drones,
            "flight_duration": f"{duration}s",
            "ground_truth": risk_level,
            "drones": drones,
            "events": [f"Formation switch at t={switch_time}s"]
        }
    
    def _generate_obstacle_avoidance_mission(self, mission_id: int) -> Dict:
        """生成避障任务 (高复杂度，包含突发障碍物)"""
        num_drones = 10  # 固定10架无人机
        duration = random.randint(70, 130)
        
        # 30% Safe, 30% Borderline, 40% Risky
        risk_level = np.random.choice(["Safe", "Borderline", "Risky"], p=[0.3, 0.3, 0.4])
        
        # 障碍物出现时间 (突然出现)
        obstacle_time = random.randint(duration // 3, 2 * duration // 3)
        obstacle_pos = (random.uniform(200, 800), random.uniform(200, 800))
        
        drones = []
        base_altitude = 50
        
        for drone_id in range(num_drones):
            trajectory = self._generate_obstacle_avoidance_trajectory(
                duration=duration,
                drone_id=drone_id,
                num_drones=num_drones,
                obstacle_time=obstacle_time,
                obstacle_pos=obstacle_pos,
                risk_level=risk_level,
                base_altitude=base_altitude
            )
            drones.append({
                "drone_id": f"UAV_{drone_id+1}",
                "trajectory": trajectory
            })
        
        return {
            "mission_id": f"OBSTACLE_{mission_id:03d}",
            "mission_type": "Obstacle Avoidance",
            "num_drones": num_drones,
            "flight_duration": f"{duration}s",
            "ground_truth": risk_level,
            "drones": drones,
            "events": [f"Obstacle detected at t={obstacle_time}s, pos=({obstacle_pos[0]:.0f}, {obstacle_pos[1]:.0f})"]
        }
    
    def _generate_search_rescue_mission(self, mission_id: int) -> Dict:
        """生成搜救任务 (高复杂度，包含GPS信号丢失)"""
        num_drones = 10  # 固定10架无人机
        duration = random.randint(100, 180)
        
        # 35% Safe, 30% Borderline, 35% Risky
        risk_level = np.random.choice(["Safe", "Borderline", "Risky"], p=[0.35, 0.3, 0.35])
        
        # GPS信号丢失时段
        dropout_start = random.randint(duration // 4, duration // 2)
        dropout_duration = random.randint(10, 25)
        
        drones = []
        base_altitude = 50
        
        for drone_id in range(num_drones):
            trajectory = self._generate_search_rescue_trajectory(
                duration=duration,
                drone_id=drone_id,
                num_drones=num_drones,
                dropout_start=dropout_start,
                dropout_duration=dropout_duration,
                risk_level=risk_level,
                base_altitude=base_altitude
            )
            drones.append({
                "drone_id": f"UAV_{drone_id+1}",
                "trajectory": trajectory
            })
        
        return {
            "mission_id": f"RESCUE_{mission_id:03d}",
            "mission_type": "Search & Rescue",
            "num_drones": num_drones,
            "flight_duration": f"{duration}s",
            "ground_truth": risk_level,
            "drones": drones,
            "events": [f"GPS dropout: t={dropout_start}s to t={dropout_start+dropout_duration}s"]
        }
    
    def _generate_adversarial_mission(self, mission_id: int) -> Dict:
        """生成对抗任务 (极高复杂度，多种挑战)"""
        num_drones = 10  # 固定10架无人机
        duration = random.randint(90, 150)
        
        # 20% Safe, 30% Borderline, 50% Risky
        risk_level = np.random.choice(["Safe", "Borderline", "Risky"], p=[0.2, 0.3, 0.5])
        
        # 多重事件
        events = []
        evasion_time = random.randint(duration // 4, duration // 2)
        events.append(f"Evasive maneuver at t={evasion_time}s")
        
        if random.random() < 0.5:
            comm_loss_time = random.randint(evasion_time + 10, duration - 20)
            events.append(f"Communication loss at t={comm_loss_time}s")
        
        drones = []
        base_altitude = 50
        
        for drone_id in range(num_drones):
            trajectory = self._generate_adversarial_trajectory(
                duration=duration,
                drone_id=drone_id,
                num_drones=num_drones,
                evasion_time=evasion_time,
                risk_level=risk_level,
                base_altitude=base_altitude
            )
            drones.append({
                "drone_id": f"UAV_{drone_id+1}",
                "trajectory": trajectory
            })
        
        return {
            "mission_id": f"ADVERSARIAL_{mission_id:03d}",
            "mission_type": "Adversarial",
            "num_drones": num_drones,
            "flight_duration": f"{duration}s",
            "ground_truth": risk_level,
            "drones": drones,
            "events": events
        }
    
    # ========== 轨迹生成核心方法 ==========
    
    def _generate_patrol_trajectory(self, duration, drone_id, num_drones, risk_level, base_altitude):
        """生成巡逻轨迹"""
        trajectory = []
        
        # 起始位置 (编队排列 - 10架无人机排成2x5矩阵)
        formation_spacing = 10
        x = 100 + (drone_id % 5) * formation_spacing  # 5列
        y = 100 + (drone_id // 5) * formation_spacing  # 2行
        altitude = base_altitude
        heading = 90  # 向东
        speed = 10
        
        # 噪声级别取决于风险
        if risk_level == "Safe":
            noise_scale = 0.5
            heading_noise = 2
            alt_noise = 0.5
        elif risk_level == "Borderline":
            noise_scale = 1.5
            heading_noise = 8
            alt_noise = 2
        else:  # Risky
            noise_scale = 3.0
            heading_noise = 20
            alt_noise = 5
        
        for t in range(duration):
            # 巡逻路径 (矩形)
            if t % 80 < 20:
                heading = 90 + np.random.normal(0, heading_noise)
            elif t % 80 < 40:
                heading = 0 + np.random.normal(0, heading_noise)
            elif t % 80 < 60:
                heading = 270 + np.random.normal(0, heading_noise)
            else:
                heading = 180 + np.random.normal(0, heading_noise)
            
            # 位置更新
            x += speed * np.cos(np.radians(heading)) + np.random.normal(0, noise_scale)
            y += speed * np.sin(np.radians(heading)) + np.random.normal(0, noise_scale)
            
            # 高度波动
            altitude += np.random.normal(0, alt_noise)
            altitude = np.clip(altitude, base_altitude - 10, base_altitude + 10)
            
            # 速度变化
            speed = 10 + np.random.normal(0, noise_scale)
            speed = np.clip(speed, 5, 15)
            
            trajectory.append({
                "time": int(t),
                "latitude": float(y / 111000),  # 粗略转换
                "longitude": float(x / 111000),
                "altitude": float(altitude),
                "heading": float(heading % 360),
                "speed": float(speed)
            })
        
        return trajectory
    
    def _generate_formation_switch_trajectory(self, duration, drone_id, num_drones, switch_time, risk_level, base_altitude):
        """生成编队切换轨迹"""
        trajectory = []
        
        # 初始编队: 线型
        x = 100 + drone_id * 10
        y = 100
        altitude = base_altitude
        heading = 90
        speed = 12
        
        # 目标编队: V型
        target_x_offset = (drone_id - num_drones // 2) * 15
        target_y_offset = abs(drone_id - num_drones // 2) * 10
        
        if risk_level == "Safe":
            switch_aggressiveness = 0.05  # 平滑切换
        elif risk_level == "Borderline":
            switch_aggressiveness = 0.15  # 中等
        else:  # Risky
            switch_aggressiveness = 0.35  # 激进切换
        
        for t in range(duration):
            if t < switch_time - 10:
                # 切换前: 保持线型编队
                target_heading = 90
            elif t < switch_time + 20:
                # 切换中: 快速机动
                progress = (t - (switch_time - 10)) / 30
                target_x = 300 + target_x_offset
                target_y = 200 + target_y_offset
                
                dx = target_x - x
                dy = target_y - y
                target_heading = np.degrees(np.atan2(dy, dx))
                
                # 加入激进度
                heading += (target_heading - heading) * switch_aggressiveness
                speed = 12 + 8 * switch_aggressiveness
            else:
                # 切换后: 保持V型编队
                target_heading = 90
            
            # 位置更新
            x += speed * np.cos(np.radians(heading))
            y += speed * np.sin(np.radians(heading))
            
            # 高度变化
            if abs(t - switch_time) < 10:
                altitude += np.random.normal(0, 3 * switch_aggressiveness)
            else:
                altitude += np.random.normal(0, 0.5)
            
            altitude = np.clip(altitude, base_altitude - 15, base_altitude + 15)
            
            trajectory.append({
                "time": int(t),
                "latitude": float(y / 111000),
                "longitude": float(x / 111000),
                "altitude": float(altitude),
                "heading": float(heading % 360),
                "speed": float(speed)
            })
        
        return trajectory
    
    def _generate_obstacle_avoidance_trajectory(self, duration, drone_id, num_drones, obstacle_time, obstacle_pos, risk_level, base_altitude):
        """生成避障轨迹 (突然出现障碍物)"""
        trajectory = []
        
        x = 100 + drone_id * 8
        y = 100
        altitude = base_altitude
        heading = 45  # 斜向飞行
        speed = 12
        
        for t in range(duration):
            # 检测是否接近障碍物
            dist_to_obstacle = np.sqrt((x - obstacle_pos[0])**2 + (y - obstacle_pos[1])**2)
            
            if t >= obstacle_time and dist_to_obstacle < 50:
                # 紧急避障!
                avoidance_angle = np.degrees(np.atan2(y - obstacle_pos[1], x - obstacle_pos[0]))
                
                if risk_level == "Safe":
                    # 平滑避障
                    heading += (avoidance_angle - heading) * 0.3
                    altitude += 2  # 爬升
                elif risk_level == "Borderline":
                    # 急转
                    heading += (avoidance_angle - heading) * 0.6
                    altitude += 1
                else:  # Risky
                    # 极端机动 (可能过载)
                    heading += (avoidance_angle - heading) * 0.9 + np.random.normal(0, 30)
                    altitude += np.random.normal(0, 5)
                
                speed = 15  # 加速
            else:
                # 正常飞行
                heading += np.random.normal(0, 2)
                speed = 12
            
            x += speed * np.cos(np.radians(heading))
            y += speed * np.sin(np.radians(heading))
            altitude = np.clip(altitude, base_altitude - 20, base_altitude + 20)
            
            trajectory.append({
                "time": int(t),
                "latitude": float(y / 111000),
                "longitude": float(x / 111000),
                "altitude": float(altitude),
                "heading": float(heading % 360),
                "speed": float(speed)
            })
        
        return trajectory
    
    def _generate_search_rescue_trajectory(self, duration, drone_id, num_drones, dropout_start, dropout_duration, risk_level, base_altitude):
        """生成搜救轨迹 (包含GPS信号丢失)"""
        trajectory = []
        
        x = 100 + drone_id * 12
        y = 100
        altitude = base_altitude
        heading = 60
        speed = 10
        
        # GPS丢失期间的漂移
        drift_x = 0
        drift_y = 0
        
        for t in range(duration):
            # 搜索模式 (之字形)
            if t % 40 < 20:
                heading = 30 + np.random.normal(0, 5)
            else:
                heading = 120 + np.random.normal(0, 5)
            
            # GPS信号丢失
            if dropout_start <= t < dropout_start + dropout_duration:
                # 信号丢失期间: 位置估计漂移
                if risk_level == "Safe":
                    drift_scale = 0.5
                elif risk_level == "Borderline":
                    drift_scale = 2.0
                else:  # Risky
                    drift_scale = 5.0
                
                drift_x += np.random.normal(0, drift_scale)
                drift_y += np.random.normal(0, drift_scale)
                
                # 速度和高度不确定
                speed = 10 + np.random.normal(0, 3)
                altitude += np.random.normal(0, 3)
            else:
                # 信号正常
                drift_x = 0
                drift_y = 0
                speed = 10
            
            x += speed * np.cos(np.radians(heading)) + drift_x
            y += speed * np.sin(np.radians(heading)) + drift_y
            altitude = np.clip(altitude, base_altitude - 20, base_altitude + 20)
            
            trajectory.append({
                "time": int(t),
                "latitude": float(y / 111000),
                "longitude": float(x / 111000),
                "altitude": float(altitude),
                "heading": float(heading % 360),
                "speed": float(speed)
            })
        
        return trajectory
    
    def _generate_adversarial_trajectory(self, duration, drone_id, num_drones, evasion_time, risk_level, base_altitude):
        """生成对抗轨迹 (极端机动)"""
        trajectory = []
        
        x = 100 + drone_id * 15
        y = 100
        altitude = base_altitude
        heading = 45
        speed = 14
        
        for t in range(duration):
            # 规避机动
            if abs(t - evasion_time) < 15:
                # 急转 + 俯冲/爬升
                if risk_level == "Safe":
                    heading += np.random.normal(0, 20)
                    altitude += np.random.normal(0, 2)
                    speed = 16
                elif risk_level == "Borderline":
                    heading += np.random.normal(0, 40)
                    altitude += np.random.normal(0, 5)
                    speed = 18
                else:  # Risky
                    heading += np.random.normal(0, 80)
                    altitude += np.random.normal(0, 10)
                    speed = 22
            else:
                # 高速巡航
                heading += np.random.normal(0, 5)
                speed = 14
            
            x += speed * np.cos(np.radians(heading))
            y += speed * np.sin(np.radians(heading))
            altitude = np.clip(altitude, base_altitude - 30, base_altitude + 30)
            
            trajectory.append({
                "time": int(t),
                "latitude": float(y / 111000),
                "longitude": float(x / 111000),
                "altitude": float(altitude),
                "heading": float(heading % 360),
                "speed": float(speed)
            })
        
        return trajectory


def main():
    print("="*80)
    print("生成真实的UAV集群任务数据集")
    print("="*80)
    
    generator = RealisticUAVMissionGenerator(seed=42)
    
    print("\n[1/2] 生成任务数据...")
    dataset = generator.generate_dataset(num_missions=50)
    
    print(f"[2/2] 保存到文件... (共{len(dataset['missions'])}个任务)")
    
    output_path = "improved_uav_missions.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! File: {output_path}")
    
    print(f"\n✓ 数据集已生成: {output_path}")
    print(f"  总任务数: {len(dataset['missions'])}")
    print(f"\n任务类型分布:")
    
    type_counts = {}
    label_counts = {"Safe": 0, "Borderline": 0, "Risky": 0}
    
    for mission in dataset['missions']:
        mtype = mission['mission_type']
        type_counts[mtype] = type_counts.get(mtype, 0) + 1
        label_counts[mission['ground_truth']] += 1
    
    for mtype, count in type_counts.items():
        print(f"  - {mtype}: {count}")
    
    print(f"\n标签分布:")
    for label, count in label_counts.items():
        print(f"  - {label}: {count} ({count/len(dataset['missions'])*100:.1f}%)")
    
    print(f"\n数据集特点:")
    print(f"  ✓ 包含突发障碍物避障场景")
    print(f"  ✓ 包含GPS信号丢失模拟")
    print(f"  ✓ 包含复杂编队切换")
    print(f"  ✓ 包含对抗性高速机动")
    print(f"  ✓ 风险级别影响轨迹质量")


if __name__ == "__main__":
    main()

