import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_real_evaluation import TrajectoryAnalyzer

print("="*80)
print("DEBUGGING MULTI-AGENT-DEBATE VETO MECHANISM")
print("="*80)

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

missions = data['missions']
print(f"Loaded {len(missions)} missions")

veto_count = 0
for idx, mission in enumerate(missions):
    print(f"\n{'='*70}")
    print(f"Mission {idx+1}/{len(missions)}: {mission['mission_id']}")
    print(f"{'='*70}")
    
    # 分析轨迹
    drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(drone['trajectory']) 
                     for drone in mission['drones']]
    formation_analysis = TrajectoryAnalyzer.analyze_formation(mission['drones'])
    
    # 检查硬约束
    max_heading = max([a.get('max_heading_change', 0) for a in drone_analyses])
    max_alt_std = max([a.get('altitude_std', 0) for a in drone_analyses])
    min_dist = formation_analysis.get('min_formation_dist', 999)
    
    print(f"  Max Heading Change: {max_heading:.1f}° (limit: 180°)")
    print(f"  Max Altitude Std: {max_alt_std:.1f}m (limit: 50m)")
    print(f"  Min Formation Dist: {min_dist:.1f}m (limit: 0.05m)")
    
    # 检查是否触发VETO
    violations = []
    if max_heading > 180:
        violations.append(f"Heading > 180°")
    if max_alt_std > 50:
        violations.append(f"Altitude std > 50m")
    if min_dist < 0.05:
        violations.append(f"Formation dist < 0.05m")
    
    if violations:
        print(f"  [VETO TRIGGERED] {', '.join(violations)}")
        veto_count += 1
    else:
        print(f"  [NO VETO] Proceeding to debate")

print("\n" + "="*80)
print(f"SUMMARY: {veto_count}/{len(missions)} missions triggered VETO ({veto_count/len(missions)*100:.1f}%)")
print("="*80)
