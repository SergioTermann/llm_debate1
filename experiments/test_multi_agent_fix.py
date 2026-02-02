import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_real_evaluation import TrajectoryAnalyzer, RealMultiAgentDebateEvaluator

print("="*80)
print("TESTING MULTI-AGENT-DEBATE FIXES")
print("="*80)

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

missions = data['missions']
print(f"Loaded {len(missions)} missions")

api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"

evaluator = RealMultiAgentDebateEvaluator(api_key, max_rounds=1, verbose=True)

# 测试关键任务
test_missions = [
    missions[0],  # COMPLEX_01 - Risky
    missions[4],  # COMPLEX_05 - Risky
    missions[9],  # COMPLEX_10 - Safe (VETO issue)
]

results = []
for mission in test_missions:
    print(f"\n{'='*70}")
    print(f"Testing: {mission['mission_id']}")
    print(f"Ground Truth: Safety={mission['ground_truth']['safety_label']}, Efficiency={mission['ground_truth']['efficiency_label']}")
    print(f"{'='*70}")
    
    result = evaluator.evaluate(mission)
    results.append(result)
    
    safety_match = "[OK]" if result['safety_label'] == mission['ground_truth']['safety_label'] else "[NO]"
    eff_match = "[OK]" if result['efficiency_label'] == mission['ground_truth']['efficiency_label'] else "[NO]"
    
    print(f"\nResult:")
    print(f"  Safety: {result['safety_label']} {safety_match}")
    print(f"  Efficiency: {result['efficiency_label']} {eff_match}")
    print(f"  Score: {result['score']}")
    if 'issues_identified' in result and result['issues_identified']:
        print(f"  Issues: {result['issues_identified']}")
    if 'debate_transcript' in result:
        print(f"  Route Layer: {result.get('route_layer', 'N/A')}")
        print(f"  Complexity: {result.get('complexity', 'N/A')}")

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

correct = sum(1 for r, m in zip(results, test_missions) 
             if r['safety_label'] == m['ground_truth']['safety_label'])
print(f"Accuracy: {correct}/{len(test_missions)} = {correct/len(test_missions)*100:.1f}%")

for i, (result, mission) in enumerate(zip(results, test_missions)):
    match = "✓" if result['safety_label'] == mission['ground_truth']['safety_label'] else "✗"
    print(f"  {match} {mission['mission_id']}: {result['safety_label']} (GT: {mission['ground_truth']['safety_label']})")

print("\n" + "="*80)
