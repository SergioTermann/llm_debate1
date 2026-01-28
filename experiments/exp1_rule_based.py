"""
简化的实验脚本 - 只使用基于规则的评估器，不调用LLM
"""

import json
import os
import math
import numpy as np
from typing import List, Dict
from tqdm import tqdm


class TrajectoryAnalyzer:
    """轨迹分析器 - 从原始数据中提取特征"""
    
    @staticmethod
    def analyze_single_drone(trajectory: List[Dict]) -> Dict:
        """分析单个无人机的轨迹"""
        if len(trajectory) < 2:
            return {
                "trajectory_smoothness": 0,
                "altitude_stability": 0,
                "speed_consistency": 0
            }
        
        altitudes = [p['altitude'] for p in trajectory]
        speeds = [p['speed'] for p in trajectory]
        headings = [p['heading'] for p in trajectory]
        
        # 轨迹平滑度 - 基于heading变化
        heading_changes = []
        for i in range(1, len(headings)):
            diff = abs(headings[i] - headings[i-1])
            if diff > 180:
                diff = 360 - diff
            heading_changes.append(diff)
        
        avg_heading_change = np.mean(heading_changes) if heading_changes else 0
        trajectory_smoothness = max(0, 100 - avg_heading_change * 2)
        
        # 高度稳定性
        altitude_std = np.std(altitudes) if len(altitudes) > 1 else 0
        altitude_stability = max(0, 100 - altitude_std * 0.5)
        
        # 速度一致性
        speed_std = np.std(speeds) if len(speeds) > 1 else 0
        speed_consistency = max(0, 100 - speed_std * 2)
        
        return {
            "trajectory_smoothness": trajectory_smoothness,
            "altitude_stability": altitude_stability,
            "speed_consistency": speed_consistency
        }
    
    @staticmethod
    def analyze_formation(drones: List[Dict]) -> Dict:
        """分析编队保持情况"""
        if len(drones) < 2:
            return {"formation_stability": 100, "coordination_quality": 100}
        
        all_trajectories = [drone['trajectory'] for drone in drones]
        min_length = min(len(t) for t in all_trajectories)
        
        if min_length < 2:
            return {"formation_stability": 100, "coordination_quality": 100}
        
        formation_distances = []
        for i in range(min_length):
            positions = []
            for trajectory in all_trajectories:
                point = trajectory[i]
                positions.append((point['gps']['lat'], point['gps']['lon']))
            
            for j in range(len(positions)):
                for k in range(j+1, len(positions)):
                    lat1, lon1 = positions[j]
                    lat2, lon2 = positions[k]
                    dist = math.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)
                    formation_distances.append(dist)
        
        if not formation_distances:
            return {"formation_stability": 100, "coordination_quality": 100}
        
        formation_std = np.std(formation_distances)
        formation_stability = max(0, 100 - formation_std * 10000)
        
        speed_correlations = []
        heading_correlations = []
        
        for i in range(min_length):
            speeds_at_time = [t[i]['speed'] for t in all_trajectories]
            headings_at_time = [t[i]['heading'] for t in all_trajectories]
            
            speed_std = np.std(speeds_at_time)
            heading_std = np.std(headings_at_time)
            
            speed_correlations.append(speed_std)
            heading_correlations.append(heading_std)
        
        avg_speed_std = np.mean(speed_correlations) if speed_correlations else 0
        avg_heading_std = np.mean(heading_correlations) if heading_correlations else 0
        
        coordination_quality = max(0, 100 - (avg_speed_std * 3 + avg_heading_std * 0.5))
        
        return {
            "formation_stability": formation_stability,
            "coordination_quality": coordination_quality
        }


class RealSingleMetricEvaluator:
    """真实的单指标评估器"""
    
    def evaluate(self, mission_data: Dict) -> Dict:
        drones = mission_data.get('drones', [])
        if not drones:
            return {"method": "Single-Metric", "safety_label": "Borderline", "score": 50, "issues_identified": []}
        
        trajectory = drones[0]['trajectory']
        analysis = TrajectoryAnalyzer.analyze_single_drone(trajectory)
        
        overall_score = (
            analysis['trajectory_smoothness'] * 0.4 +
            analysis['altitude_stability'] * 0.3 +
            analysis['speed_consistency'] * 0.3
        )
        
        if overall_score >= 80:
            safety_label = "Safe"
        elif overall_score >= 60:
            safety_label = "Borderline"
        else:
            safety_label = "Risky"
        
        return {
            "method": "Single-Metric",
            "safety_label": safety_label,
            "score": overall_score,
            "issues_identified": [],
            "analysis": analysis
        }


class RealFixedWeightEvaluator:
    """真实的固定权重评估器"""
    
    def evaluate(self, mission_data: Dict) -> Dict:
        drones = mission_data.get('drones', [])
        if not drones:
            return {"method": "Fixed-Weight", "safety_label": "Borderline", "score": 50, "issues_identified": []}
        
        drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(d['trajectory']) for d in drones]
        formation_analysis = TrajectoryAnalyzer.analyze_formation(drones)
        
        avg_smoothness = np.mean([a['trajectory_smoothness'] for a in drone_analyses])
        avg_altitude_stability = np.mean([a['altitude_stability'] for a in drone_analyses])
        avg_speed_consistency = np.mean([a['speed_consistency'] for a in drone_analyses])
        
        overall_score = (
            avg_smoothness * 0.25 +
            avg_altitude_stability * 0.25 +
            avg_speed_consistency * 0.2 +
            formation_analysis['formation_stability'] * 0.15 +
            formation_analysis['coordination_quality'] * 0.15
        )
        
        if overall_score >= 80:
            safety_label = "Safe"
        elif overall_score >= 60:
            safety_label = "Borderline"
        else:
            safety_label = "Risky"
        
        return {
            "method": "Fixed-Weight",
            "safety_label": safety_label,
            "score": overall_score,
            "issues_identified": [],
            "analysis": {
                "avg_smoothness": avg_smoothness,
                "avg_altitude_stability": avg_altitude_stability,
                "avg_speed_consistency": avg_speed_consistency,
                "formation_stability": formation_analysis['formation_stability'],
                "coordination_quality": formation_analysis['coordination_quality']
            }
        }


def load_missions(dataset_path: str) -> List[Dict]:
    """加载任务数据"""
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'missions' in data:
        return data['missions']
    return [data]


def compute_metrics(predictions: List[str], ground_truth: List[str]) -> Dict:
    """计算评估指标"""
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions) if predictions else 0
    
    pred_binary = [1 if p == "Safe" else 0 for p in predictions]
    gt_binary = [1 if g == "Safe" else 0 for g in ground_truth]
    
    precision, recall, f1, _ = precision_recall_fscore_support(gt_binary, pred_binary, average='binary', zero_division=0)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}


def main():
    print("="*80)
    print("EXPERIMENT 1: Rule-Based Evaluation (No LLM, No Data Leakage)")
    print("="*80)
    
    missions = load_missions("improved_uav_missions.json")
    if not missions:
        print("ERROR: No missions loaded.")
        return
    
    print(f"\nLoaded {len(missions)} missions")
    
    evaluators = {
        "Single-Metric": RealSingleMetricEvaluator(),
        "Fixed-Weight": RealFixedWeightEvaluator()
    }
    
    missions_to_evaluate = missions[:15]
    print(f"\nEvaluating {len(missions_to_evaluate)} missions")
    
    results = {name: [] for name in evaluators.keys()}
    
    print("\nRunning evaluations...")
    for mission in tqdm(missions_to_evaluate, desc="Evaluating missions"):
        for name, evaluator in evaluators.items():
            try:
                result = evaluator.evaluate(mission)
                results[name].append({
                    "mission_id": mission['mission_id'],
                    "prediction": result['safety_label'],
                    "score": result.get('score', 0),
                    "issues": result.get('issues_identified', [])
                })
            except Exception as e:
                print(f"\nError in {name} for {mission['mission_id']}: {e}")
                results[name].append({
                    "mission_id": mission['mission_id'],
                    "prediction": "Borderline",
                    "score": 50,
                    "issues": []
                })
    
    ground_truth = [m['ground_truth'] for m in missions_to_evaluate]
    
    metrics_table = {}
    for name in evaluators.keys():
        predictions = [r['prediction'] for r in results[name]]
        metrics = compute_metrics(predictions, ground_truth)
        metrics_table[name] = metrics
    
    print("\nRESULTS")
    print("\n" + "="*70)
    print("Performance Comparison on {} Missions".format(len(missions_to_evaluate)))
    print("="*70)
    print(f"{'Method':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    
    for name, metrics in metrics_table.items():
        print(f"{name:<25} {metrics['accuracy']:<12.2%} {metrics['precision']:<12.2%} "
              f"{metrics['recall']:<12.2%} {metrics['f1_score']:<12.2%}")
    
    print("="*70)
    
    # 打印详细结果
    print("\nDetailed Results:")
    print("="*70)
    for i, mission in enumerate(missions_to_evaluate, 1):
        print(f"\n[{i}] {mission['mission_id']} (GT: {mission['ground_truth']})")
        for name in evaluators.keys():
            pred = results[name][i-1]['prediction']
            score = results[name][i-1]['score']
            correct = "✓" if pred == mission['ground_truth'] else "✗"
            print(f"  {name:<20} {pred:<12} (score: {score:.1f}) {correct}")
    
    print("="*70)
    
    output_file = "exp1_rule_based_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": "improved_uav_missions.json",
            "num_missions": len(missions_to_evaluate),
            "results": results,
            "metrics": metrics_table
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    from sklearn.metrics import precision_recall_fscore_support
    main()
