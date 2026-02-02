"""
Complete Evaluation on Paper Scale Dataset (50 Missions)
Full comparison with paper baseline methods
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_real_evaluation import RealSingleMetricEvaluator, RealFixedWeightEvaluator
from exp1_real_evaluation import RealSingleAgentLLMEvaluator, RealMultiAgentDebateEvaluator

api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevroc"

DATASET_PATH = "complex_uav_missions_50.json"

def compute_metrics(predictions, ground_truths):
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / len(predictions) if predictions else 0
    
    binary_preds = [1 if p == "Risky" else 0 for p in predictions]
    binary_gt = [1 if g == "Risky" else 0 for g in ground_truths]
    
    tp = sum(1 for p, g in zip(binary_preds, binary_gt) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(binary_preds, binary_gt) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(binary_preds, binary_gt) if p == 0 and g == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def main():
    print("="*80)
    print("COMPLETE EVALUATION - Paper Scale (50 Missions)")
    print("="*80)
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    missions = data['missions']
    print(f"Total missions: {len(missions)}")
    
    ground_truths = [m['ground_truth']['safety_label'] for m in missions]
    
    print("\nGround Truth Distribution:")
    safe_count = sum(1 for g in ground_truths if g == "Safe")
    borderline_count = sum(1 for g in ground_truths if g == "Borderline")
    risky_count = sum(1 for g in ground_truths if g == "Risky")
    print(f"  Safe: {safe_count}, Borderline: {borderline_count}, Risky: {risky_count}")
    
    evaluators = {
        "Single-Metric": RealSingleMetricEvaluator(api_key),
        "Fixed-Weight": RealFixedWeightEvaluator(api_key),
        "Single-Agent-LLM": RealSingleAgentLLMEvaluator(api_key),
        "Multi-Agent-Debate": RealMultiAgentDebateEvaluator(api_key, max_rounds=2, verbose=False)
    }
    
    results = {name: [] for name in evaluators.keys()}
    
    print("\nRunning evaluations...")
    for idx, mission in enumerate(missions):
        mission_id = mission['mission_id']
        gt = mission['ground_truth']['safety_label']
        
        print(f"\n[{idx+1}/50] {mission_id} (GT: {gt})")
        
        for name, evaluator in evaluators.items():
            try:
                result = evaluator.evaluate(mission)
                pred = result['safety_label']
                results[name].append(pred)
                status = "OK" if pred == gt else "NO"
                print(f"  [{name}: {pred[:8]}] {status}")
            except Exception as e:
                print(f"  [{name}: ERROR] {e}")
                results[name].append("Borderline")
    
    print("\n" + "="*80)
    print("FINAL RESULTS - Compared to Paper Table III")
    print("="*80)
    
    print(f"\n{'Method':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*70)
    
    metrics_table = {}
    for name in evaluators.keys():
        metrics = compute_metrics(results[name], ground_truths)
        metrics_table[name] = metrics
        print(f"{name:<20} {metrics['accuracy']:<12.2%} {metrics['precision']:<12.2%} "
              f"{metrics['recall']:<12.2%} {metrics['f1_score']:<12.2%}")
    
    print("\n" + "="*80)
    print("Detailed Comparison with Paper")
    print("="*80)
    print(f"\nPaper Results (50 missions):")
    print(f"  Single-Metric: 62% Acc, 58% Precision, 71% Recall, 0.64 F1")
    print(f"  Fixed-Weights: 71% Acc, 69% Precision, 78% Recall, 0.73 F1")
    print(f"  Single-Agent:  74% Acc, 72% Precision, 81% Recall, 0.76 F1")
    print(f"  Ours (Debate): 91% Acc, 89% Precision, 96% Recall, 0.92 F1")
    
    output_file = "paper_scale_results.json"
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": DATASET_PATH,
        "num_missions": len(missions),
        "ground_truth_distribution": {
            "safe": safe_count,
            "borderline": borderline_count,
            "risky": risky_count
        },
        "results": results,
        "metrics": metrics_table
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
