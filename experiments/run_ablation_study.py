"""
Ablation Study Script (Paper Scale - 50 Missions)
Tests the contribution of each core mechanism

Paper: Multi-Agent Debate Framework for Comprehensive UAV Swarm Performance Evaluation
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_real_evaluation import RealSingleMetricEvaluator, RealFixedWeightEvaluator
from exp1_real_evaluation import RealSingleAgentLLMEvaluator, RealMultiAgentDebateEvaluator

api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"

DATASET_PATH = "complex_uav_missions_50.json"
if not os.path.exists(DATASET_PATH):
    print(f"Dataset not found: {DATASET_PATH}")
    print("Please run generate_paper_scale_dataset.py first")
    sys.exit(1)

ABLATION_VARIANTS = {
    "Full System": {
        "role_rotation": True,
        "evidence_verification": True,
        "adversarial": True,
        "hierarchical": True
    },
    "w/o Role Rotation": {
        "role_rotation": False,
        "evidence_verification": True,
        "adversarial": True,
        "hierarchical": True
    },
    "w/o Evidence Verification": {
        "role_rotation": True,
        "evidence_verification": False,
        "adversarial": True,
        "hierarchical": True
    },
    "w/o Adversarial": {
        "role_rotation": True,
        "evidence_verification": True,
        "adversarial": False,
        "hierarchical": True
    },
    "w/o Hierarchical (All L3)": {
        "role_rotation": True,
        "evidence_verification": True,
        "adversarial": True,
        "hierarchical": False
    },
    "Single Agent (All Roles)": {
        "role_rotation": False,
        "evidence_verification": False,
        "adversarial": False,
        "hierarchical": False
    }
}

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

def run_ablation_experiment(variant_name: str, ablation_settings: Dict, missions: List[Dict], max_missions: int = None):
    """Run evaluation with specific ablation settings"""
    print(f"\n{'='*70}")
    print(f"ABLATION VARIANT: {variant_name}")
    print(f"Settings: {ablation_settings}")
    print(f"{'='*70}")
    
    if max_missions:
        missions_subset = missions[:max_missions]
        print(f"Running on {len(missions_subset)} missions (limited for testing)")
    else:
        missions_subset = missions
    
    ground_truths = [m['ground_truth']['safety_label'] for m in missions_subset]
    
    evaluators = {
        "Single-Metric": RealSingleMetricEvaluator(api_key),
        "Fixed-Weight": RealFixedWeightEvaluator(api_key),
        "Single-Agent-LLM": RealSingleAgentLLMEvaluator(api_key)
    }
    
    if variant_name == "Single Agent (All Roles)":
        evaluators["Multi-Agent-Debate"] = RealSingleAgentLLMEvaluator(api_key)
    else:
        evaluators["Multi-Agent-Debate"] = RealMultiAgentDebateEvaluator(
            api_key, 
            max_rounds=2, 
            verbose=False,
            ablation_settings=ablation_settings
        )
    
    results = {name: [] for name in evaluators.keys()}
    
    for idx, mission in enumerate(missions_subset):
        mission_id = mission['mission_id']
        print(f"\n[{idx+1}/{len(missions_subset)}] {mission_id}", end=" ", flush=True)
        
        for name, evaluator in evaluators.items():
            try:
                result = evaluator.evaluate(mission)
                pred = result['safety_label']
                results[name].append(pred)
                status = "OK" if pred == mission['ground_truth']['safety_label'] else "XX"
                print(f"[{name}:{pred[:3]}]", end=" ", flush=True)
            except Exception as e:
                print(f"[{name}:ERR]", end=" ", flush=True)
                results[name].append("Borderline")
    
    print("\n")
    
    metrics_table = {}
    for name in evaluators.keys():
        metrics = compute_metrics(results[name], ground_truths)
        metrics_table[name] = metrics
        print(f"{name}: Acc={metrics['accuracy']:.2%}, Recall={metrics['recall']:.2%}, F1={metrics['f1_score']:.2%}")
    
    return metrics_table

def main():
    print("="*80)
    print("ABLATION STUDY - Paper Scale (50 Missions)")
    print("="*80)
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    missions = data['missions']
    print(f"Total missions in dataset: {len(missions)}")
    
    all_results = {}
    
    for variant_name, ablation_settings in ABLATION_VARIANTS.items():
        try:
            metrics = run_ablation_experiment(variant_name, ablation_settings, missions)
            all_results[variant_name] = metrics
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nError in {variant_name}: {e}")
            all_results[variant_name] = {"error": str(e)}
    
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Variant':<30} {'Accuracy':<12} {'Recall':<12} {'F1':<12}")
    print("-"*70)
    
    for variant_name, metrics in all_results.items():
        if "error" not in metrics:
            print(f"{variant_name:<30} {metrics['accuracy']:<12.2%} {metrics['recall']:<12.2%} {metrics['f1_score']:<12.2%}")
        else:
            print(f"{variant_name:<30} ERROR")
    
    output_file = "ablation_study_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": DATASET_PATH,
            "total_missions": len(missions),
            "ablation_variants": ABLATION_VARIANTS,
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
