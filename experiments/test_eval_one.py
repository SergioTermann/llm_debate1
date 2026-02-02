import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_real_evaluation import TrajectoryAnalyzer, RealSingleMetricEvaluator, RealFixedWeightEvaluator
import json

print("Testing evaluation without LLM...")

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

mission = data['missions'][0]
print(f"\nMission: {mission['mission_id']}")
print(f"Ground Truth Safety: {mission['ground_truth']['safety_label']}")
print(f"Ground Truth Efficiency: {mission['ground_truth']['efficiency_label']}")

# Test Single-Metric
evaluator = RealSingleMetricEvaluator("dummy")
result = evaluator.evaluate(mission)
print(f"\nSingle-Metric Result:")
print(f"  Safety: {result['safety_label']}")
print(f"  Efficiency: {result['efficiency_label']}")
print(f"  Score: {result['score']:.2f}")

# Test Fixed-Weight
evaluator2 = RealFixedWeightEvaluator("dummy")
result2 = evaluator2.evaluate(mission)
print(f"\nFixed-Weight Result:")
print(f"  Safety: {result2['safety_label']}")
print(f"  Efficiency: {result2['efficiency_label']}")
print(f"  Score: {result2['score']:.2f}")

print("\nTest completed successfully!")
