"""
场景化UAV数据集生成器 - 专为多智能体辩论设计
生成具有"语义歧义"和"规则冲突"的复杂场景
"""

import json
import numpy as np
import math
import random
from typing import List, Dict

class ScenarioGenerator:
    def __init__(self):
        self.base_alt = 50
        
    def generate_scenarios(self) -> Dict:
        scenarios = []
        
        # 1. 极速穿越 (Efficient Risk)
        scenarios.append(self._create_efficient_risk())
        
        # 2. 通信中断后的悬停 (Lost & Found)
        scenarios.append(self._create_lost_and_found())
        
        # 3. 编队压缩 (Formation Squeeze)
        scenarios.append(self._create_formation_squeeze())
        
        # 4. 鲁莽的王牌 (Reckless Ace)
        scenarios.append(self._create_reckless_ace())
        
        # 5. 保守的乌龟 (Lazy Pilot)
        scenarios.append(self._create_lazy_pilot())
        
        return {"missions": scenarios}

    def _create_efficient_risk(self):
        """场景1: 极速穿越 - 贴地飞行，贴近障碍，追求极致效率"""
        duration = 60
        trajectory = []
        x, y = 0, 0
        
        for t in range(duration):
            # 直线冲刺
            x += 25  # 极快速度 25m/s
            y += 0
            # 极低高度 (5m)
            alt = 5 + random.uniform(-0.5, 0.5)
            
            trajectory.append({
                "time": t, "latitude": y/111000, "longitude": x/111000,
                "altitude": alt, "heading": 90, "speed": 25
            })
            
        return {
            "mission_id": "SCENARIO_01_EFFICIENT_RISK",
            "mission_type": "High Speed Delivery",
            "flight_duration": "60s",
            "ground_truth": "Borderline", # 争议点：效率极高但太危险
            "drones": [{"drone_id": "UAV_1", "trajectory": trajectory}],
            "description": "Drone flies at 25m/s at only 5m altitude to save time."
        }

    def _create_lost_and_found(self):
        """场景2: 通信中断 - 中间有一段数据缺失，但前后行为是安全的悬停"""
        duration = 80
        trajectory = []
        x, y = 0, 0
        
        for t in range(duration):
            # 20-50s 期间数据丢失
            if 20 <= t < 50:
                continue
            
            if t < 20:
                x += 10
            elif t >= 50:
                x += 10 # 恢复飞行
                
            trajectory.append({
                "time": t, "latitude": y/111000, "longitude": x/111000,
                "altitude": 50, "heading": 90, "speed": 10 if t < 20 or t >= 50 else 0
            })
            
        return {
            "mission_id": "SCENARIO_02_LOST_SIGNAL",
            "mission_type": "Reconnaissance",
            "flight_duration": "80s",
            "ground_truth": "Safe", # 争议点：数据缺失通常是Risky，但这里它安全处理了
            "drones": [{"drone_id": "UAV_1", "trajectory": trajectory}],
            "description": "GPS signal lost for 30s. Drone hovers (implied) and resumes."
        }

    def _create_formation_squeeze(self):
        """场景3: 编队压缩 - 两架无人机贴得很近 (1m) 通过狭窄区域"""
        duration = 60
        drones = []
        
        # UAV 1
        traj1 = []
        # UAV 2
        traj2 = []
        
        for t in range(duration):
            x = t * 10
            
            # 20-40s 期间靠得很近
            if 20 <= t < 40:
                y_offset = 0.5 # 间距 1m
            else:
                y_offset = 5.0 # 间距 10m
                
            traj1.append({
                "time": t, "latitude": y_offset/111000, "longitude": x/111000,
                "altitude": 50, "heading": 90, "speed": 10
            })
            traj2.append({
                "time": t, "latitude": -y_offset/111000, "longitude": x/111000,
                "altitude": 50, "heading": 90, "speed": 10
            })
            
        return {
            "mission_id": "SCENARIO_03_FORMATION_SQUEEZE",
            "mission_type": "Formation Flight",
            "flight_duration": "60s",
            "ground_truth": "Safe", # 争议点：距离<2m通常是Risky，但这是战术动作
            "drones": [
                {"drone_id": "UAV_1", "trajectory": traj1},
                {"drone_id": "UAV_2", "trajectory": traj2}
            ],
            "description": "Formation compresses to 1m separation to pass obstacle."
        }

    def _create_reckless_ace(self):
        """场景4: 鲁莽的王牌 - 剧烈机动，数值很难看，但没有发生碰撞"""
        duration = 60
        trajectory = []
        x, y = 0, 0
        heading = 0
        
        for t in range(duration):
            # 疯狂改变航向
            heading += random.uniform(-45, 45)
            speed = 15 + random.uniform(-5, 5)
            alt = 50 + random.uniform(-10, 10)
            
            x += speed * math.cos(math.radians(heading))
            y += speed * math.sin(math.radians(heading))
            
            trajectory.append({
                "time": t, "latitude": y/111000, "longitude": x/111000,
                "altitude": alt, "heading": heading % 360, "speed": speed
            })
            
        return {
            "mission_id": "SCENARIO_04_RECKLESS_ACE",
            "mission_type": "Combat Maneuver",
            "flight_duration": "60s",
            "ground_truth": "Risky", # 争议点：虽然没撞，但由于不可预测性应判为Risky
            "drones": [{"drone_id": "UAV_1", "trajectory": trajectory}],
            "description": "Erratic flight path with high jerk and heading changes."
        }

    def _create_lazy_pilot(self):
        """场景5: 保守的乌龟 - 极其平稳，但效率极低（绕大圈）"""
        duration = 100
        trajectory = []
        
        # 绕一个巨大的半圆去终点
        center_x, center_y = 500, 0
        radius = 500
        
        for t in range(duration):
            angle = 180 + (t / duration) * 180 # 半圆
            x = center_x + radius * math.cos(math.radians(angle))
            y = center_y + radius * math.sin(math.radians(angle))
            
            trajectory.append({
                "time": t, "latitude": y/111000, "longitude": x/111000,
                "altitude": 50, "heading": (angle + 90) % 360, "speed": 5 # 很慢
            })
            
        return {
            "mission_id": "SCENARIO_05_LAZY_PILOT",
            "mission_type": "Transport",
            "flight_duration": "100s",
            "ground_truth": "Safe", # 争议点：非常安全，但效率极低 (Low Efficiency)
            "drones": [{"drone_id": "UAV_1", "trajectory": trajectory}],
            "description": "Perfectly smooth flight but takes a massive detour."
        }

def main():
    generator = ScenarioGenerator()
    data = generator.generate_scenarios()
    
    with open("scenario_uav_missions.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print("Generated 5 debate scenarios in 'scenario_uav_missions.json'")

if __name__ == "__main__":
    main()

