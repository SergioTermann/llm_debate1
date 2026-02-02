import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_real_evaluation import RealMultiAgentDebateEvaluator

api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

evaluator = RealMultiAgentDebateEvaluator(api_key, max_rounds=2, verbose=False)

print("Testing Multi-Agent-Debate on all missions...")

results = []
correct = 0

for mission in data['missions']:
    mission_id = mission['mission_id']
    gt = mission['ground_truth']['safety_label']
    
    try:
        result = evaluator.evaluate(mission)
        pred = result['safety_label']
        is_correct = pred == gt
        if is_correct:
            correct += 1
        
        results.append({
            "mission_id": mission_id,
            "ground_truth": gt,
            "prediction": pred,
            "correct": is_correct
        })
        
        status = "[OK]" if is_correct else "[NO]"
        print(f"{status} {mission_id}: {pred} (GT: {gt})")
    except Exception as e:
        print(f"ERROR {mission_id}: {e}")
        results.append({
            "mission_id": mission_id,
            "ground_truth": gt,
            "prediction": "ERROR",
            "correct": False
        })

print(f"\n正确率: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")

with open("test_mad_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Results saved to test_mad_results.json")
