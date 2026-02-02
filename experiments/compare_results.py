import json

with open('complex_uav_missions.json', 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

with open('exp1_real_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

gt_dict = {m['mission_id']: m['ground_truth']['safety_label'] for m in gt_data['missions']}

print("=" * 70)
print("Ground Truth vs Multi-Agent-Debate Predictions")
print("=" * 70)

mad_results = results['results']['Multi-Agent-Debate']
for i, result in enumerate(mad_results):
    mission_id = result['mission_id']
    pred = result['prediction']
    gt = gt_dict[mission_id]
    status = "✓" if pred == gt else "✗"
    print(f"{status} {mission_id}")
    print(f"   Ground Truth: {gt}")
    print(f"   Prediction:   {pred}")
    print(f"   Issues: {result.get('issues', [])[:2]}")
    print()

correct = sum(1 for r in mad_results if r['prediction'] == gt_dict[r['mission_id']])
print(f"正确率: {correct}/{len(mad_results)} = {correct/len(mad_results)*100:.1f}%")
