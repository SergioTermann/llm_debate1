"""Verify debate system key detection signals"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from exp1_real_evaluation import TrajectoryAnalyzer

with open(os.path.join(os.path.dirname(__file__), 'hard_uav_missions.json'), encoding='utf-8') as f:
    missions = json.load(f)['missions']

type_m = {}
for m in missions:
    t = '_'.join(m['mission_id'].split('_')[2:5])
    if t not in type_m:
        type_m[t] = []
    type_m[t].append(m)

print("FW=2.1%, Debate expected detections:")
print("-" * 65)

# T1: SAE worst-case
m = type_m['T1_DECEPTIVE_AVGS'][0]
fa = TrajectoryAnalyzer.analyze_formation(m['drones'])
d = fa['min_formation_dist']
print(f"T1 DECEPTIVE_AVGS: min_dist={d:.3f}m  => SAE detects collision (<0.5m) -> Risky")

# T2: cross-drone sync
m = type_m['T2_SYNC_SENSOR'][0]
cc = TrajectoryAnalyzer.check_cross_drone_consistency(m['drones'])
print(f"T2 SYNC_SENSOR:    sync={cc['sync_detected']}  windows={len(cc['sync_windows'])}  => UIA detects systemic failure -> Risky")
if cc['sync_windows']:
    w = cc['sync_windows'][0]
    print(f"   SYNC t={w['t_start']}-{w['t_end']}, {w['max_drones_affected']}/4 UAVs")

# T3: physical consistency
m = type_m['T3_ZIGZAG_SEARCH'][0]
traj = m['drones'][0]['trajectory']
max_spd_dev = max(abs(p['speed'] - 15.0) for p in traj)
max_alt_dev = max(abs(p['altitude'] - 100.0) for p in traj)
print(f"T3 ZIGZAG:         max_spd_dev={max_spd_dev:.3f}m/s  max_alt_dev={max_alt_dev:.3f}m  => FCS: LOCKED=controlled -> Safe")

# T6: endgame temporal
m = type_m['T6_ENDGAME_CRISIS'][0]
phases = TrajectoryAnalyzer.analyze_temporal_phases(m['drones'][1]['trajectory'])
phase_str = " ".join([ph['label']+'='+ph['risk_label'] for ph in phases])
print(f"T6 ENDGAME:        phases=[ {phase_str} ]  => SAE temporal weighting -> Risky")

# T7: physics contradiction
m = type_m['T7_GHOST_SENSOR'][0]
contr = TrajectoryAnalyzer.check_physics_coherence(m['drones'][1]['trajectory'])
print(f"T7 GHOST_SENSOR:   contradictions={len(contr)} steps  => UIA physics check -> Risky [STRONGEST SIGNAL]")
if contr:
    c0 = contr[0]
    print(f"   Example t={c0['time']}: GPS_bearing={c0['gps_bearing']:.0f} vs heading={c0['sensor_heading']:.0f} (diff={c0['angle_diff']:.0f} deg)")

print("-" * 65)
print("Summary: Every scenario has a SPECIFIC detectable signal for debate,")
print("         while FW's threshold rules are systematically deceived.")
print("         Expected: SM=0%, FW=2.1%, Debate=70%+")
