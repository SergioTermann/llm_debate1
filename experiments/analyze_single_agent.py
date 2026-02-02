import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, "exp1_real_results.json")

with open(results_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*80)
print("Single-Agent-LLM 详细预测分析")
print("="*80)

results = data['results']['Single-Agent-LLM']
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")
with open(dataset_path, 'r', encoding='utf-8') as f:
    missions_data = json.load(f)['missions']

correct = 0
total = len(results)

for i, result in enumerate(results):
    mission_id = result['mission_id']
    prediction = result['prediction']
    gt = missions_data[i]['ground_truth']['safety_label']
    
    match = prediction == gt
    if match:
        correct += 1
    
    status = "✓" if match else "✗"
    print(f"{status} {mission_id}: {prediction} (GT: {gt})")

print(f"\n准确率: {correct}/{total} = {correct/total*100:.1f}%")
print("="*80)
