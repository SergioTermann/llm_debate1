import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("Loading dataset...")
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

missions = data['missions']
print(f"Loaded {len(missions)} missions")

print("\nMission 1 Ground Truth:")
gt1 = missions[0]['ground_truth']
print(f"  Safety Label: {gt1['safety_label']}")
print(f"  Safety Score: {gt1['safety_score']}")
print(f"  Efficiency Label: {gt1['efficiency_label']}")
print(f"  Efficiency Score: {gt1['efficiency_score']}")
print(f"  Formation Stability: {gt1['formation_stability']}")
print(f"  Coordination Quality: {gt1['coordination_quality']}")
print(f"  Trajectory Smoothness: {gt1['trajectory_smoothness']}")
print(f"  Altitude Stability: {gt1['altitude_stability']}")
print(f"  Speed Consistency: {gt1['speed_consistency']}")

print("\nMission 10 Ground Truth:")
gt10 = missions[9]['ground_truth']
print(f"  Safety Label: {gt10['safety_label']}")
print(f"  Safety Score: {gt10['safety_score']}")
print(f"  Efficiency Label: {gt10['efficiency_label']}")
print(f"  Efficiency Score: {gt10['efficiency_score']}")
print(f"  Formation Stability: {gt10['formation_stability']}")
print(f"  Coordination Quality: {gt10['coordination_quality']}")

print("\nAll Safety Labels:")
for i, mission in enumerate(missions):
    gt = mission['ground_truth']
    print(f"  {i+1}. {mission['mission_id']}: {gt['safety_label']}")

print("\nAll Efficiency Labels:")
for i, mission in enumerate(missions):
    gt = mission['ground_truth']
    print(f"  {i+1}. {mission['mission_id']}: {gt['efficiency_label']}")
