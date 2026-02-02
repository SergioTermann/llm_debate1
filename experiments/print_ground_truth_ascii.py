import json
import numpy as np

file_path = r'c:\Users\kevin\Desktop\llm_debate\experiments\complex_uav_missions.json'
data = json.load(open(file_path, encoding='utf-8'))

mission_names = [m['mission_id'].replace('COMPLEX_', '') for m in data['missions']]

safety_scores = [m['ground_truth']['safety_score'] for m in data['missions']]
efficiency_scores = [m['ground_truth']['efficiency_score'] for m in data['missions']]
formation_stability = [m['ground_truth']['formation_stability'] for m in data['missions']]
coordination_quality = [m['ground_truth']['coordination_quality'] for m in data['missions']]
trajectory_smoothness = [m['ground_truth']['trajectory_smoothness'] for m in data['missions']]
altitude_stability = [m['ground_truth']['altitude_stability'] for m in data['missions']]
speed_consistency = [m['ground_truth']['speed_consistency'] for m in data['missions']]
mission_complexity = [m['ground_truth']['mission_complexity'] for m in data['missions']]

risk_counts = [len(m['ground_truth']['risk_factors']) for m in data['missions']]
unobservable_counts = [len(m['ground_truth']['unobservable_issues']) for m in data['missions']]

print("=" * 120)
print("GROUND TRUTH ANALYSIS FOR COMPLEX UAV MISSIONS")
print("=" * 120)
print()

print("1. SAFETY SCORES")
print("-" * 120)
print(f"{'Mission':<15} | {'Score':<10} | {'Label':<15} | {'Bar':<60}")
print("-" * 120)
for i, (name, score) in enumerate(zip(mission_names, safety_scores)):
    label = 'Risky' if score == 0 else ('Borderline' if score < 75 else 'Safe')
    bar_length = int(score / 2)
    bar = '█' * bar_length + '░' * (50 - bar_length)
    color = '🔴' if score == 0 else ('🟠' if score < 75 else '🟢')
    print(f"{name:<15} | {score:>9.1f} | {label:<15} | {color} {bar}")
print()

print("2. EFFICIENCY SCORES")
print("-" * 120)
print(f"{'Mission':<15} | {'Score':<10} | {'Label':<15} | {'Bar':<60}")
print("-" * 120)
for i, (name, score) in enumerate(zip(mission_names, efficiency_scores)):
    label = 'Low' if score < 50 else ('Medium' if score < 70 else 'High')
    bar_length = int(score / 2)
    bar = '█' * bar_length + '░' * (50 - bar_length)
    color = '🔴' if score < 50 else ('🟠' if score < 70 else '🟢')
    print(f"{name:<15} | {score:>9.1f} | {label:<15} | {color} {bar}")
print()

print("3. RISK FACTORS COUNT")
print("-" * 120)
print(f"{'Mission':<15} | {'Count':<10} | {'Bar':<60}")
print("-" * 120)
for i, (name, count) in enumerate(zip(mission_names, risk_counts)):
    bar_length = int(count * 2)
    bar = '█' * bar_length + '░' * (50 - bar_length)
    print(f"{name:<15} | {count:>9} | {'🔴' if count > 0 else '🟢'} {bar}")
print()

print("4. UNOBSERVABLE ISSUES COUNT")
print("-" * 120)
print(f"{'Mission':<15} | {'Count':<10} | {'Bar':<60}")
print("-" * 120)
for i, (name, count) in enumerate(zip(mission_names, unobservable_counts)):
    bar_length = int(count * 8)
    bar = '█' * bar_length + '░' * (50 - bar_length)
    print(f"{name:<15} | {count:>9} | {'🟠' if count > 0 else '🟢'} {bar}")
print()

print("5. MULTI-DIMENSIONAL METRICS")
print("-" * 120)
print(f"{'Mission':<15} | {'Formation':<12} | {'Coord':<12} | {'Trajectory':<12} | {'Altitude':<12} | {'Speed':<12}")
print("-" * 120)
for i, name in enumerate(mission_names):
    print(f"{name:<15} | {formation_stability[i]:>11.1f} | {coordination_quality[i]:>11.1f} | "
          f"{trajectory_smoothness[i]:>11.1f} | {altitude_stability[i]:>11.1f} | {speed_consistency[i]:>11.1f}")
print()

print("6. MISSION COMPLEXITY")
print("-" * 120)
print(f"{'Mission':<15} | {'Complexity':<15} | {'Bar':<60}")
print("-" * 120)
for i, (name, score) in enumerate(zip(mission_names, mission_complexity)):
    bar_length = int(score / 1.5)
    bar = '█' * bar_length + '░' * (50 - bar_length)
    print(f"{name:<15} | {score:>14.1f} | {'🟣' if score > 60 else ('🟠' if score > 40 else '🟢')} {bar}")
print()

print("7. SAFETY vs EFFICIENCY COMPARISON")
print("-" * 120)
print(f"{'Mission':<15} | {'Safety':<15} | {'Efficiency':<15} | {'Gap':<15}")
print("-" * 120)
for i, name in enumerate(mission_names):
    gap = efficiency_scores[i] - safety_scores[i]
    print(f"{name:<15} | {safety_scores[i]:>14.1f} | {efficiency_scores[i]:>14.1f} | {gap:>+14.1f}")
print()

print("=" * 120)
print("SUMMARY STATISTICS")
print("=" * 120)
print(f"Total Missions: {len(mission_names)}")
print(f"Average Safety Score: {np.mean(safety_scores):.2f}")
print(f"Average Efficiency Score: {np.mean(efficiency_scores):.2f}")
print(f"Average Risk Factors per Mission: {np.mean(risk_counts):.2f}")
print(f"Average Unobservable Issues per Mission: {np.mean(unobservable_counts):.2f}")
print(f"Average Mission Complexity: {np.mean(mission_complexity):.2f}")
print(f"Safest Mission: {mission_names[np.argmax(safety_scores)]} ({max(safety_scores):.1f})")
print(f"Most Efficient Mission: {mission_names[np.argmax(efficiency_scores)]} ({max(efficiency_scores):.1f})")
print(f"Most Complex Mission: {mission_names[np.argmax(mission_complexity)]} ({max(mission_complexity):.1f})")
print("=" * 120)
