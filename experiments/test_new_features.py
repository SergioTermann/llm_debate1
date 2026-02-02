"""
Quick test on 5 missions to verify all features work
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_real_evaluation import RealMultiAgentDebateEvaluator

api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"

with open('complex_uav_missions_50.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Testing all new features on 5 missions...")
print("="*70)

evaluator = RealMultiAgentDebateEvaluator(
    api_key, 
    max_rounds=2, 
    verbose=True,
    ablation_settings={
        'role_rotation': True,
        'evidence_verification': True,
        'adversarial': True,
        'hierarchical': True
    }
)

for idx in range(5):
    mission = data['missions'][idx]
    print(f"\n{'='*70}")
    print(f"[{idx+1}/5] {mission['mission_id']}")
    print(f"Ground Truth: {mission['ground_truth']['safety_label']}")
    print(f"{'='*70}")
    
    try:
        result = evaluator.evaluate(mission)
        pred = result['safety_label']
        status = "OK" if pred == mission['ground_truth']['safety_label'] else "NO"
        print(f"\nPrediction: {pred} | Status: {status}")
        print(f"Route Layer: {result.get('route_layer', 'N/A')}")
        print(f"Complexity: {result.get('complexity', 'N/A'):.3f}")
    except Exception as e:
        print(f"Error: {e}")

print("\n\nTest completed!")
