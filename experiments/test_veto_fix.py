import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_real_evaluation import TrajectoryAnalyzer

print("="*80)
print("TESTING VETO FIX - LANDING PHASE DETECTION")
print("="*80)

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

missions = data['missions']
print(f"Loaded {len(missions)} missions")

# 测试关键任务
test_cases = [
    missions[0],  # COMPLEX_01 - Risky (normal formation)
    missions[4],  # COMPLEX_05 - Risky (normal formation)
    missions[9],  # COMPLEX_10 - Safe (landing phase, formation distance 0.0m)
]

for mission in test_cases:
    print(f"\n{'='*70}")
    print(f"Mission: {mission['mission_id']}")
    print(f"Type: {mission.get('mission_type', 'N/A')}")
    print(f"Ground Truth: Safety={mission['ground_truth']['safety_label']}")
    print(f"{'='*70}")
    
    # 分析轨迹
    drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(drone['trajectory']) 
                     for drone in mission['drones']]
    formation_analysis = TrajectoryAnalyzer.analyze_formation(mission['drones'])
    
    # 检查关键指标
    max_heading = max([a.get('max_heading_change', 0) for a in drone_analyses])
    max_alt_std = max([a.get('altitude_std', 0) for a in drone_analyses])
    min_dist = formation_analysis.get('min_formation_dist', 999)
    
    print(f"  Max Heading Change: {max_heading:.1f}° (limit: 180°)")
    print(f"  Max Altitude Std: {max_alt_std:.1f}m (limit: 50m)")
    print(f"  Min Formation Dist: {min_dist:.1f}m (limit: 0.05m)")
    
    # 检查是否是着陆阶段
    mission_type = mission.get('mission_type', '')
    is_landing_takeoff = False
    if 'landing' in mission_type.lower() or 'takeoff' in mission_type.lower():
        is_landing_takeoff = True
        print(f"  [LANDING/TAKEOFF MISSION DETECTED from type]")
    
    if not is_landing_takeoff and len(drone_analyses) > 0 and len(drone_analyses[0].get('altitude_trend', [])) > 1:
        alt_trend = drone_analyses[0]['altitude_trend']
        alt_start = alt_trend[0]
        alt_end = alt_trend[-1]
        alt_change = alt_end - alt_start
        print(f"  Altitude Change: {alt_start:.1f}m -> {alt_end:.1f}m (Δ={alt_change:.1f}m)")
        
        if abs(alt_change) > 50:
            is_landing_takeoff = True
            print(f"  [LANDING/TAKEOFF PHASE DETECTED from altitude trend]")
    
    if is_landing_takeoff:
        print(f"  [NO VETO] Formation distance {min_dist:.1f}m is acceptable during landing/takeoff")
    elif min_dist < 0.05:
        print(f"  [VETO TRIGGERED] Formation distance {min_dist:.1f}m < 0.05m (Collision risk)")
    else:
        print(f"  [NO VETO] Formation distance {min_dist:.1f}m is safe")

print("\n" + "="*80)
print("EXPECTED RESULTS:")
print("="*80)
print("  COMPLEX_01: NO VETO (normal formation)")
print("  COMPLEX_05: NO VETO (normal formation)")
print("  COMPLEX_10: NO VETO (landing phase, formation distance 0.0m is OK)")
print("\n" + "="*80)
