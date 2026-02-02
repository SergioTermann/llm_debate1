import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_real_evaluation import TrajectoryAnalyzer, RealSingleMetricEvaluator, RealFixedWeightEvaluator

print("="*80)
print("COMPLETE EVALUATION (No LLM API)")
print("="*80)

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

missions = data['missions']
print(f"Loaded {len(missions)} missions")

api_key = "dummy"

evaluators = {
    "Single-Metric": RealSingleMetricEvaluator(api_key),
    "Fixed-Weight": RealFixedWeightEvaluator(api_key)
}

results = {name: [] for name in evaluators.keys()}

for idx, mission in enumerate(missions):
    print(f"\n{'='*70}")
    print(f"Mission {idx+1}/{len(missions)}: {mission['mission_id']}")
    print(f"{'='*70}")
    gt_safety = mission['ground_truth']['safety_label']
    gt_eff = mission['ground_truth']['efficiency_label']
    print(f"Ground Truth: Safety={gt_safety} | Efficiency={gt_eff}")
    
    for name, evaluator in evaluators.items():
        result = evaluator.evaluate(mission)
        results[name].append(result)
        safety_match = "[OK]" if result['safety_label'] == gt_safety else "[NO]"
        eff_match = "[OK]" if result['efficiency_label'] == gt_eff else "[NO]"
        print(f"  {name}: {result['safety_label']} {safety_match} | {result['efficiency_label']} {eff_match}")

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

for name in evaluators.keys():
    safety_correct = sum(1 for r, m in zip(results[name], missions) 
                        if r['safety_label'] == m['ground_truth']['safety_label'])
    eff_correct = sum(1 for r, m in zip(results[name], missions) 
                     if r['efficiency_label'] == m['ground_truth']['efficiency_label'])
    
    safety_acc = safety_correct / len(missions) * 100
    eff_acc = eff_correct / len(missions) * 100
    
    print(f"\n{name}:")
    print(f"  Safety Accuracy: {safety_correct}/{len(missions)} = {safety_acc:.1f}%")
    print(f"  Efficiency Accuracy: {eff_correct}/{len(missions)} = {eff_acc:.1f}%")
    
    # 计算precision, recall, F1 for safety
    tp = sum(1 for r, m in zip(results[name], missions) 
             if r['safety_label'] == "Risky" and m['ground_truth']['safety_label'] == "Risky")
    fp = sum(1 for r, m in zip(results[name], missions) 
             if r['safety_label'] == "Risky" and m['ground_truth']['safety_label'] != "Risky")
    fn = sum(1 for r, m in zip(results[name], missions) 
             if r['safety_label'] != "Risky" and m['ground_truth']['safety_label'] == "Risky")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1:.2f}")

print("\n" + "="*80)
