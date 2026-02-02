import json

file_path = r'c:\Users\kevin\Desktop\llm_debate\experiments\complex_uav_missions.json'
data = json.load(open(file_path, encoding='utf-8'))

print("=" * 100)
print("GROUND TRUTH ANALYSIS FOR COMPLEX UAV MISSIONS")
print("=" * 100)
print()

for i, mission in enumerate(data['missions']):
    print(f"Mission {i+1}: {mission['mission_id']}")
    print(f"Type: {mission['mission_type']}")
    print(f"Description: {mission['description']}")
    print()
    print("Ground Truth:")
    gt = mission['ground_truth']
    print(f"  Safety: {gt['safety_label']} (Score: {gt['safety_score']:.2f})")
    print(f"  Efficiency: {gt['efficiency_label']} (Score: {gt['efficiency_score']:.2f})")
    print(f"  Formation Stability: {gt['formation_stability']:.2f}")
    print(f"  Coordination Quality: {gt['coordination_quality']:.2f}")
    print(f"  Trajectory Smoothness: {gt['trajectory_smoothness']:.2f}")
    print(f"  Altitude Stability: {gt['altitude_stability']:.2f}")
    print(f"  Speed Consistency: {gt['speed_consistency']:.2f}")
    print(f"  Mission Complexity: {gt['mission_complexity']:.2f}")
    
    if gt['risk_factors']:
        print(f"  Risk Factors ({len(gt['risk_factors'])}):")
        for rf in gt['risk_factors']:
            print(f"    - {rf['drone_id']}: {rf['type']} ({rf['severity']}, value={rf['value']:.2f})")
    
    if gt['unobservable_issues']:
        print(f"  Unobservable Issues ({len(gt['unobservable_issues'])}):")
        for ui in gt['unobservable_issues']:
            print(f"    - {ui['drone_id']}: {ui['type']} ({ui['severity']}, count={ui['count']})")
    
    print("-" * 100)
    print()

print("=" * 100)
print(f"Total: {len(data['missions'])} missions")
print("=" * 100)
