import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

missions = data['missions']

# 统计ground truth
safety_counts = {"Safe": 0, "Borderline": 0, "Risky": 0}
efficiency_counts = {"High": 0, "Medium": 0, "Low": 0}

for mission in missions:
    gt = mission['ground_truth']
    safety_counts[gt['safety_label']] += 1
    efficiency_counts[gt['efficiency_label']] += 1

output = {
    "total_missions": len(missions),
    "safety_distribution": safety_counts,
    "efficiency_distribution": efficiency_counts,
    "missions": []
}

for mission in missions:
    gt = mission['ground_truth']
    output["missions"].append({
        "mission_id": mission['mission_id'],
        "safety_label": gt['safety_label'],
        "safety_score": gt['safety_score'],
        "efficiency_label": gt['efficiency_label'],
        "efficiency_score": gt['efficiency_score']
    })

with open(os.path.join(script_dir, "ground_truth_summary.json"), 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Ground truth summary saved to ground_truth_summary.json")
print(f"Total missions: {len(missions)}")
print(f"Safety distribution: {safety_counts}")
print(f"Efficiency distribution: {efficiency_counts}")
