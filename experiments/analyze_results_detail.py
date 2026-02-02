import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, "exp1_real_results.json")

with open(results_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*80)
print("详细预测分析")
print("="*80)

for method in ['Single-Agent-LLM', 'Multi-Agent-Debate']:
    print(f"\n{'='*40}")
    print(f"{method}")
    print(f"{'='*40}")
    
    results = data['results'][method]
    dataset_path = os.path.join(script_dir, "complex_uav_missions.json")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        missions_data = json.load(f)['missions']
    
    correct = 0
    for i, result in enumerate(results):
        mission_id = result['mission_id']
        prediction = result['prediction']
        gt = missions_data[i]['ground_truth']['safety_label']
        
        match = prediction == gt
        if match:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        issues = result.get('issues_identified', [])
        print(f"{status} {mission_id}: {prediction} (GT: {gt})")
        if issues:
            print(f"   Issues: {issues[:2]}")
    
    print(f"\n准确率: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")

print("\n" + "="*80)
