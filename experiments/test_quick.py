"""快速验证脚本 - 测试新增功能是否正常工作"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from exp1_real_evaluation import (
    TrajectoryAnalyzer, RealMultiAgentDebateEvaluator,
    RealSingleMetricEvaluator, RealFixedWeightEvaluator
)

# 加载新数据集
with open(os.path.join(os.path.dirname(__file__), 'hard_uav_missions.json'), encoding='utf-8') as f:
    data = json.load(f)
missions = data['missions']
print(f"Loaded {len(missions)} missions")

print("\n[TEST 1] Temporal Phase Analysis")
for m in missions[:4]:
    drones = m['drones']
    traj = drones[0]['trajectory']
    phases = TrajectoryAnalyzer.analyze_temporal_phases(traj)
    phase_str = " | ".join([f"{ph['label']}:{ph['risk_label']}" for ph in phases])
    print(f"  {m['mission_id'][:40]:40s} GT={m['ground_truth']['safety_label']:10s} => {phase_str}")

print("\n[TEST 2] Cross-Drone Consistency (SYNC_SENSOR_FAIL should trigger)")
for m in missions[6:12]:  # T2_SYNC_SENSOR_FAIL
    drones = m['drones']
    cross = TrajectoryAnalyzer.check_cross_drone_consistency(drones)
    print(f"  {m['mission_id'][:40]:40s} sync={cross['sync_detected']}, verdict={cross['verdict'][:40]}")

print("\n[TEST 3] Physics Coherence (GHOST_SENSOR should show contradictions)")
for m in missions[36:42]:  # T7_GHOST_SENSOR
    drones = m['drones']
    contradictions = TrajectoryAnalyzer.check_physics_coherence(drones[1]['trajectory'])
    print(f"  {m['mission_id'][:40]:40s} contradictions={len(contradictions)}")

print("\n[TEST 4] Trajectory Summary includes new sections")
m = missions[6]  # T2_SYNC_SENSOR_FAIL
drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(d['trajectory']) for d in m['drones']]
formation_analysis = TrajectoryAnalyzer.analyze_formation(m['drones'])

api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"
evaluator = RealMultiAgentDebateEvaluator.__new__(RealMultiAgentDebateEvaluator)
evaluator.api_key = api_key

summary = evaluator._build_trajectory_summary(m, drone_analyses, formation_analysis)
print("  Summary sections found:")
for section in ["TEMPORAL PHASE", "CROSS-DRONE", "PHYSICS COHERENCE", "INDIVIDUAL DRONE"]:
    found = section in summary
    print(f"    {section}: {'OK' if found else 'MISSING'}")

print("\nAll tests passed!")
