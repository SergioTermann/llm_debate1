import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_real_evaluation import TrajectoryAnalyzer

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

missions = data['missions']

# 检查被错误分类的任务
error_cases = [
    missions[4],  # COMPLEX_05 - GT: Risky, Pred: Borderline
    missions[5],  # COMPLEX_06 - GT: Risky, Pred: Borderline
    missions[6],  # COMPLEX_07 - GT: Risky, Pred: Borderline
    missions[9],  # COMPLEX_10 - GT: Safe, Pred: Borderline
]

for mission in error_cases:
    print(f"\n{'='*70}")
    print(f"Mission: {mission['mission_id']}")
    print(f"Type: {mission.get('mission_type', 'N/A')}")
    print(f"Ground Truth: Safety={mission['ground_truth']['safety_label']}")
    print(f"{'='*70}")
    
    # 分析轨迹
    drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(d['trajectory']) for d in mission['drones']]
    formation_analysis = TrajectoryAnalyzer.analyze_formation(mission['drones'])
    
    # 计算指标
    smoothness = np.mean([a['trajectory_smoothness'] for a in drone_analyses])
    altitude = np.mean([a['altitude_stability'] for a in drone_analyses])
    speed = np.mean([a['speed_consistency'] for a in drone_analyses])
    formation = formation_analysis['formation_stability']
    coordination = formation_analysis['coordination_quality']
    overall_avg = (smoothness + altitude + speed + formation + coordination) / 5
    
    print(f"\nMetrics (0-100):")
    print(f"  Smoothness: {smoothness:.1f}")
    print(f"  Altitude Stability: {altitude:.1f}")
    print(f"  Speed Consistency: {speed:.1f}")
    print(f"  Formation Stability: {formation:.1f}")
    print(f"  Coordination: {coordination:.1f}")
    print(f"  OVERALL AVERAGE: {overall_avg:.1f}")
    
    # 检查致命缺陷
    max_heading = max([a.get('max_heading_change', 0) for a in drone_analyses])
    max_alt_std = max([a.get('altitude_std', 0) for a in drone_analyses])
    min_dist = formation_analysis.get('min_formation_dist', 999)
    
    print(f"\nFatal Flaws Check:")
    print(f"  Max Heading Change: {max_heading:.1f}° (limit: 180°) {'❌ FATAL' if max_heading > 180 else '✅ OK'}")
    print(f"  Max Altitude Std: {max_alt_std:.1f}m (limit: 50m) {'❌ FATAL' if max_alt_std > 50 else '✅ OK'}")
    print(f"  Min Formation Dist: {min_dist:.1f}m (limit: 0.05m) {'❌ FATAL' if min_dist < 0.05 else '✅ OK'}")
    
    # 分类建议
    print(f"\nClassification Analysis:")
    if overall_avg >= 70 and max_heading <= 180 and max_alt_std <= 50 and min_dist >= 0.05:
        print(f"  → Should be SAFE (Overall >= 70, no fatal flaws)")
    elif overall_avg < 55 or max_heading > 180 or max_alt_std > 50 or min_dist < 0.05:
        print(f"  → Should be RISKY (Overall < 55 OR fatal flaws)")
    else:
        print(f"  → Could be BORDERLINE (Overall 55-69, no fatal flaws)")

print("\n" + "="*80)
