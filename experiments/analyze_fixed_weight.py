"""分析 Fixed-Weight 在新数据集上的表现，找出和 Multi-Agent Debate 的差距点"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from exp1_real_evaluation import TrajectoryAnalyzer, RealFixedWeightEvaluator, RealSingleMetricEvaluator

with open(os.path.join(os.path.dirname(__file__), 'hard_uav_missions.json'), encoding='utf-8') as f:
    missions = json.load(f)['missions']

fw = RealFixedWeightEvaluator("test")
sm = RealSingleMetricEvaluator("test")

print("=== Fixed-Weight vs Single-Metric Prediction Analysis ===\n")
header = f"{'Scenario Type':<28} {'GT':10} {'SM':10} {'FW':10} {'SM Match':8} {'FW Match':8}"
print(header)
print("-" * 80)

type_stats = {}
for m in missions:
    gt = m['ground_truth']['safety_label']
    sm_result = sm.evaluate(m)
    fw_result = fw.evaluate(m)
    sm_pred = sm_result['safety_label']
    fw_pred = fw_result['safety_label']
    fw_score = fw_result['score']

    mtype = '_'.join(m['mission_id'].split('_')[2:5])
    print(f"  {mtype:<28} {gt:10} {sm_pred:10} {fw_pred:10} {'OK' if sm_pred==gt else 'FAIL':8} {'OK' if fw_pred==gt else 'FAIL':8}")

    if mtype not in type_stats:
        type_stats[mtype] = {'sm_correct': 0, 'fw_correct': 0, 'total': 0, 'fw_scores': [], 'gt': gt}
    type_stats[mtype]['total'] += 1
    type_stats[mtype]['fw_scores'].append(fw_score)
    if sm_pred == gt: type_stats[mtype]['sm_correct'] += 1
    if fw_pred == gt: type_stats[mtype]['fw_correct'] += 1

print("\n=== Accuracy Summary by Scenario Type ===\n")
sm_total_correct = fw_total_correct = total_missions = 0
print(f"{'Type':<28} {'GT':10} {'SM Acc':8} {'FW Acc':8} {'FW Avg Score':12} Remarks")
print("-" * 90)
for mtype, s in type_stats.items():
    sm_acc = s['sm_correct'] / s['total'] * 100
    fw_acc = s['fw_correct'] / s['total'] * 100
    fw_avg = sum(s['fw_scores']) / len(s['fw_scores'])
    sm_total_correct += s['sm_correct']
    fw_total_correct += s['fw_correct']
    total_missions += s['total']

    # 分析原因
    remark = ""
    if fw_acc == 100: remark = "FW catches (lucky/threshold)"
    elif fw_acc == 0:
        if s['gt'] == 'Safe': remark = "FW says Risky/Borderline on Safe GT"
        elif s['gt'] == 'Risky': remark = "FW says Safe/Borderline on Risky GT"
        else: remark = "FW misses Borderline"

    print(f"  {mtype:<28} {s['gt']:10} {sm_acc:7.0f}% {fw_acc:7.0f}% {fw_avg:12.1f} {remark}")

print(f"\n  TOTAL: SM={sm_total_correct}/{total_missions}={sm_total_correct/total_missions*100:.1f}%  "
      f"FW={fw_total_correct}/{total_missions}={fw_total_correct/total_missions*100:.1f}%")
print("\n=== KEY INSIGHT ===")
print("For Multi-Agent Debate to show clear advantage:")
print("  - Needs to get RIGHT where FW gets WRONG")
print("  - Especially: GT=Safe scenarios (FW says Risky) and GT=Risky scenarios (FW says Safe)")
