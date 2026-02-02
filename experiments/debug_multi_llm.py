import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_real_evaluation import RealMultiAgentDebateEvaluator

print("="*80)
print("DEBUGGING MULTI-AGENT-DEBATE OUTPUT")
print("="*80)

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

missions = data['missions']
print(f"Loaded {len(missions)} missions")

api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"

evaluator = RealMultiAgentDebateEvaluator(api_key, max_rounds=1, verbose=True)

# 测试第一个任务
mission = missions[0]
print(f"\nTesting mission: {mission['mission_id']}")
print(f"Ground Truth: Safety={mission['ground_truth']['safety_label']}, Efficiency={mission['ground_truth']['efficiency_label']}")

result = evaluator.evaluate(mission)

print(f"\nResult:")
print(f"  Safety Label: {result['safety_label']}")
print(f"  Efficiency Label: {result['efficiency_label']}")
print(f"  Score: {result['score']}")
print(f"  Issues: {result.get('issues_identified', [])}")

if 'debate_transcript' in result:
    print(f"\nDebate Transcript:")
    for i, transcript in enumerate(result['debate_transcript'][:5]):
        print(f"  {i+1}. {transcript}")

print("\n" + "="*80)
