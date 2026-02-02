import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

print(f"Dataset path: {dataset_path}")
print(f"File exists: {os.path.exists(dataset_path)}")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

missions = data['missions']
print(f"Loaded {len(missions)} missions")

# 显示ground truth
print("\nGround Truth Summary:")
for mission in missions:
    gt = mission['ground_truth']
    print(f"  {mission['mission_id']}: Safety={gt['safety_label']}, Efficiency={gt['efficiency_label']}")
