"""诊断FW在各场景上的详细评分，找出为何没按预期失败"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from exp1_real_evaluation import TrajectoryAnalyzer, RealFixedWeightEvaluator
import numpy as np

with open(os.path.join(os.path.dirname(__file__), 'hard_uav_missions.json'), encoding='utf-8') as f:
    missions = json.load(f)['missions']

fw = RealFixedWeightEvaluator("test")

print("=== Detailed FW Score Breakdown ===\n")

# 只看 T7 和 T8
for m in missions:
    mid = m['mission_id']
    mtype = '_'.join(mid.split('_')[2:5])
    if mtype not in ('T5_SPLIT_FORMATION', 'T7_GHOST_SENSOR', 'T8_NEAR_MISS'):
        continue

    gt  = m['ground_truth']['safety_label']
    res = fw.evaluate(m)
    
    drones = m['drones']
    drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(d['trajectory']) for d in drones]
    formation = TrajectoryAnalyzer.analyze_formation(drones)
    unobs = fw.detect_unobservable_issues(drones)

    avg_sm = np.mean([a['trajectory_smoothness'] for a in drone_analyses])
    avg_al = np.mean([a['altitude_stability'] for a in drone_analyses])
    avg_sp = np.mean([a['speed_consistency'] for a in drone_analyses])
    avg_mh = np.mean([a['max_heading_change'] for a in drone_analyses])
    avg_ar = np.mean([a['altitude_range'] for a in drone_analyses])

    print(f"{mtype} | GT={gt:10s} | FW={res['safety_label']:10s} | score={res['score']:.1f}")
    print(f"  sm={avg_sm:.1f} al={avg_al:.1f} sp={avg_sp:.1f} "
          f"form={formation['formation_stability']:.1f} coord={formation['coordination_quality']:.1f}")
    print(f"  min_dist={formation['min_formation_dist']:.2f}m  "
          f"max_hdg_chg={avg_mh:.1f}deg  alt_range={avg_ar:.1f}m")
    print(f"  gps_drift={unobs['total_gps_drift']} pts  "
          f"severity={unobs['severity']}  affected={unobs['affected_drones']}")
    print()
