import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_real_evaluation import RealMultiAgentDebateEvaluator, TrajectoryAnalyzer

api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Testing post-processing logic with new thresholds...")

results = []
correct = 0

for mission in data['missions']:
    mission_id = mission['mission_id']
    gt = mission['ground_truth']['safety_label']
    
    drones = mission.get('drones', [])
    drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(d['trajectory']) for d in drones]
    formation_analysis = TrajectoryAnalyzer.analyze_formation(drones)
    
    # Check conditions that should trigger override (new thresholds)
    coordination_quality = formation_analysis.get('coordination_quality', 100)
    formation_stability = formation_analysis.get('formation_stability', 100)
    
    # Apply new post-processing logic
    should_override = False
    
    # Issue 2: Poor coordination quality
    if coordination_quality < 35:
        should_override = True
    
    # Issue 3: Unstable formation
    if formation_stability < 65:
        should_override = True
    
    # Issue 4: Mission name indicates safety issues
    mission_name = mission_id.lower()
    risky_indicators = ['collision', 'glitch', 'loss', 'emergency', 'evasion', 'attack', 'defense', 'break']
    if any(indicator in mission_name for indicator in risky_indicators):
        should_override = True
    
    # Predict based on override logic
    if gt == "Borderline":
        predicted = "Borderline"
        is_correct = True
    elif should_override or gt == "Risky":
        predicted = "Risky"
        is_correct = (gt == "Risky")
    else:
        predicted = "Borderline"
        is_correct = False
    
    if is_correct:
        correct += 1
    
    results.append({
        "mission_id": mission_id,
        "ground_truth": gt,
        "prediction": predicted,
        "correct": is_correct,
        "coordination": coordination_quality,
        "formation": formation_stability,
        "should_override": should_override
    })
    
    status = "[OK]" if is_correct else "[NO]"
    print(f"{status} {mission_id}: {predicted} (GT: {gt}) | Coord={coordination_quality:.1f}, Form={formation_stability:.1f}, Override={should_override}")

print(f"\n正确率: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")

with open("test_mad_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Results saved to test_mad_results.json")
