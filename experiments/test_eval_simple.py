import json
import os
from exp1_real_evaluation import TrajectoryAnalyzer, RealSingleMetricEvaluator, RealFixedWeightEvaluator

print("="*80)
print("SIMPLIFIED EVALUATION (No LLM API)")
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
    print(f"\nMission {idx+1}/{len(missions)}: {mission['mission_id']}")
    print(f"Ground Truth: Safety={mission['ground_truth']['safety_label']}")
    
    for name, evaluator in evaluators.items():
        result = evaluator.evaluate(mission)
        results[name].append(result)
        print(f"  {name}: {result['safety_label']} (score: {result['score']:.1f})")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for name in evaluators.keys():
    correct = sum(1 for r, m in zip(results[name], missions) 
                  if r['safety_label'] == m['ground_truth']['safety_label'])
    accuracy = correct / len(missions) * 100
    print(f"{name}: {correct}/{len(missions)} = {accuracy:.1f}%")
