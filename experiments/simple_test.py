print("Testing evaluation...")

import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script dir: {script_dir}")

dataset_path = os.path.join(script_dir, "complex_uav_missions.json")
print(f"Dataset path: {dataset_path}")
print(f"Dataset exists: {os.path.exists(dataset_path)}")

try:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Dataset loaded successfully")
    print(f"Number of missions: {len(data['missions'])}")
    
    # 显示第一个任务的ground truth
    mission = data['missions'][0]
    gt = mission['ground_truth']
    print(f"\nFirst mission: {mission['mission_id']}")
    print(f"  Safety: {gt['safety_label']}")
    print(f"  Efficiency: {gt['efficiency_label']}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
