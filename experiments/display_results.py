import json

with open('exp1_real_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*80)
print("EXPERIMENT 1: REAL EVALUATION RESULTS")
print("="*80)
print(f"Dataset: {data['dataset']}")
print(f"Total Missions: {data['num_missions']}")
print()

print("="*80)
print("PERFORMANCE COMPARISON")
print("="*80)
print(f"{'Method':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-"*80)

for method_name, metrics in data['metrics'].items():
    print(f"{method_name:<25} {metrics['accuracy']:<12.2%} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} {metrics['f1_score']:<12.3f}")

print("="*80)
print("\nDETAILED RESULTS:")
print("="*80)

for method_name, results in data['results'].items():
    print(f"\n{method_name}:")
    print(f"  {'Mission':<35} {'Prediction':<12} {'Efficiency':<12} {'Score':<10}")
    print("  " + "-"*70)
    for result in results:
        print(f"  {result['mission_id']:<35} {result['prediction']:<12} {result['efficiency_prediction']:<12} {result['score']:<10.2f}")

print("\n" + "="*80)
