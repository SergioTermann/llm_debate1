"""Verify T6 late-phase formation trend is visible in trajectory summary"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from exp1_real_evaluation import TrajectoryAnalyzer, RealMultiAgentDebateEvaluator

with open(os.path.join(os.path.dirname(__file__), 'hard_uav_missions.json'), encoding='utf-8') as f:
    missions = json.load(f)['missions']

m6 = next(m for m in missions if 'T6_ENDGAME' in m['mission_id'])
drones = m6['drones']
drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(d['trajectory']) for d in drones]
formation_analysis = TrajectoryAnalyzer.analyze_formation(drones)

ev = RealMultiAgentDebateEvaluator.__new__(RealMultiAgentDebateEvaluator)
summary = ev._build_trajectory_summary(m6, drone_analyses, formation_analysis)

print("=== T6 ENDGAME_CRISIS trajectory summary (key sections) ===")
for section in ["FORMATION DISTANCE TREND", "CRITICAL EXTREMES", "Description"]:
    lines = summary.split('\n')
    in_section = False
    for line in lines:
        if section in line:
            in_section = True
        if in_section:
            print(line)
            if line.startswith('═══') and in_section and section not in line:
                break
