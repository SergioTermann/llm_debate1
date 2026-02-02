import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("GROUND TRUTH VISUALIZATION")
print("=" * 60)

print("\n[1] Loading data...")
file_path = r'c:\Users\kevin\Desktop\llm_debate\experiments\complex_uav_missions.json'
print(f"File path: {file_path}")
print(f"File exists: {os.path.exists(file_path)}")

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data['missions'])} missions")

print("\n[2] Extracting data...")
mission_names = [m['mission_id'].replace('COMPLEX_', '') for m in data['missions']]
safety_scores = [m['ground_truth']['safety_score'] for m in data['missions']]
efficiency_scores = [m['ground_truth']['efficiency_score'] for m in data['missions']]
risk_counts = [len(m['ground_truth']['risk_factors']) for m in data['missions']]
unobservable_counts = [len(m['ground_truth']['unobservable_issues']) for m in data['missions']]

print(f"Safety scores: {safety_scores}")
print(f"Efficiency scores: {efficiency_scores}")
print(f"Risk counts: {risk_counts}")
print(f"Unobservable counts: {unobservable_counts}")

print("\n[3] Creating figure...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Ground Truth Analysis for Complex UAV Missions', fontsize=16, fontweight='bold')

print("[4] Plotting safety scores...")
ax = axes[0, 0]
colors = ['red' if s == 0 else ('orange' if s < 75 else 'green') for s in safety_scores]
bars = ax.bar(range(len(mission_names)), safety_scores, color=colors, alpha=0.7)
ax.set_xlabel('Mission')
ax.set_ylabel('Safety Score')
ax.set_title('Safety Scores by Mission')
ax.set_xticks(range(len(mission_names)))
ax.set_xticklabels(mission_names, rotation=45, ha='right')
ax.set_ylim(-5, 105)
ax.grid(axis='y', alpha=0.3)
for i, (bar, score) in enumerate(zip(bars, safety_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{score:.1f}', ha='center', va='bottom')

print("[5] Plotting efficiency scores...")
ax = axes[0, 1]
colors2 = ['green' if e >= 70 else ('orange' if e >= 50 else 'red') for e in efficiency_scores]
bars = ax.bar(range(len(mission_names)), efficiency_scores, color=colors2, alpha=0.7)
ax.set_xlabel('Mission')
ax.set_ylabel('Efficiency Score')
ax.set_title('Efficiency Scores by Mission')
ax.set_xticks(range(len(mission_names)))
ax.set_xticklabels(mission_names, rotation=45, ha='right')
ax.set_ylim(-5, 105)
ax.grid(axis='y', alpha=0.3)
for i, (bar, score) in enumerate(zip(bars, efficiency_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{score:.1f}', ha='center', va='bottom')

print("[6] Plotting risk factors...")
ax = axes[1, 0]
bars = ax.bar(range(len(mission_names)), risk_counts, color='red', alpha=0.7)
ax.set_xlabel('Mission')
ax.set_ylabel('Risk Factors Count')
ax.set_title('Risk Factors by Mission')
ax.set_xticks(range(len(mission_names)))
ax.set_xticklabels(mission_names, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for i, (bar, count) in enumerate(zip(bars, risk_counts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{count}', ha='center', va='bottom')

print("[7] Plotting unobservable issues...")
ax = axes[1, 1]
bars = ax.bar(range(len(mission_names)), unobservable_counts, color='orange', alpha=0.7)
ax.set_xlabel('Mission')
ax.set_ylabel('Unobservable Issues Count')
ax.set_title('Unobservable Issues by Mission')
ax.set_xticks(range(len(mission_names)))
ax.set_xticklabels(mission_names, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for i, (bar, count) in enumerate(zip(bars, unobservable_counts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{count}', ha='center', va='bottom')

print("[8] Adjusting layout...")
plt.tight_layout()

print("[9] Saving figure...")
output_path = os.path.abspath(r'c:\Users\kevin\Desktop\llm_debate\experiments\ground_truth_visualization.png')
print(f"Output path: {output_path}")
print(f"Output directory exists: {os.path.exists(os.path.dirname(output_path))}")

try:
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"File saved successfully")
except Exception as e:
    print(f"Error saving file: {e}")
    import traceback
    traceback.print_exc()

plt.close()

print(f"\n[10] Verifying file...")
print(f"File exists: {os.path.exists(output_path)}")
if os.path.exists(output_path):
    print(f"File size: {os.path.getsize(output_path)} bytes")
    print(f"Full path: {os.path.abspath(output_path)}")
    print("\nSUCCESS! Visualization created.")
else:
    print("\nFAILED! File was not created.")

print("=" * 60)
