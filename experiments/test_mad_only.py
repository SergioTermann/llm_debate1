import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_real_evaluation import RealMultiAgentDebateEvaluator, TrajectoryAnalyzer

api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Testing Multi-Agent-Debate with post-processing fix...")
print("="*70)

evaluator = RealMultiAgentDebateEvaluator(api_key, max_rounds=2, verbose=False)

results = []
correct = 0

for idx, mission in enumerate(data['missions']):
    mission_id = mission['mission_id']
    gt = mission['ground_truth']['safety_label']
    
    print(f"\n[{idx+1}/10] {mission_id}")
    print(f"    Ground Truth: {gt}")
    
    try:
        start_time = time.time()
        result = evaluator.evaluate(mission)
        elapsed = time.time() - start_time
        
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
        print(f"    {status} Prediction: {pred} (elapsed: {elapsed:.1f}s)")
        
    except Exception as e:
        print(f"    ERROR: {e}")
        results.append({
            "mission_id": mission_id,
            "ground_truth": gt,
            "prediction": "ERROR",
            "correct": False
        })

print("\n" + "="*70)
print(f"正确率: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")

output_file = "exp1_real_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "dataset": "complex_uav_missions.json",
        "num_missions": len(results),
        "results": {
            "Multi-Agent-Debate": results,
            "Single-Metric": [],
            "Fixed-Weight": [],
            "Single-Agent-LLM": []
        },
        "metrics": {}
    }, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {output_file}")
