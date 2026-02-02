import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

missions = data['missions']

print("="*80)
print("GROUND TRUTH SUMMARY")
print("="*80)
print(f"{'Mission':<35} {'Safety':<10} {'Efficiency':<10} {'Formation':<10} {'Coordination':<12}")
print("-"*80)

for mission in missions:
    gt = mission['ground_truth']
    print(f"{mission['mission_id']:<35} {gt['safety_label']:<10} {gt['efficiency_label']:<10} {gt['formation_stability']:<10.1f} {gt['coordination_quality']:<12.1f}")

print("="*80)

# 统计
safety_counts = {"Safe": 0, "Borderline": 0, "Risky": 0}
efficiency_counts = {"High": 0, "Medium": 0, "Low": 0}

for mission in missions:
    gt = mission['ground_truth']
    safety_counts[gt['safety_label']] += 1
    efficiency_counts[gt['efficiency_label']] += 1

print("\nSAFETY DISTRIBUTION:")
for label, count in safety_counts.items():
    print(f"  {label}: {count} ({count/len(missions)*100:.1f}%)")

print("\nEFFICIENCY DISTRIBUTION:")
for label, count in efficiency_counts.items():
    print(f"  {label}: {count} ({count/len(missions)*100:.1f}%)")

print("="*80)
