"""
测试复杂数据集的评估效果
"""

import json
import os
import sys
sys.path.append('..')

from exp1_real_evaluation import RealSingleMetricEvaluator, RealFixedWeightEvaluator, TrajectoryAnalyzer

def test_complex_dataset():
    """测试复杂数据集"""
    
    print("="*80)
    print("测试复杂数据集评估效果")
    print("="*80)
    
    # 加载复杂数据集
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "complex_uav_missions.json")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    missions = data['missions']
    print(f"\n加载了 {len(missions)} 个复杂任务")
    
    # 创建评估器（不使用LLM）
    evaluators = {
        "Single-Metric": RealSingleMetricEvaluator("test-key"),
        "Fixed-Weight": RealFixedWeightEvaluator("test-key")
    }
    
    # 评估每个任务
    results = {name: [] for name in evaluators.keys()}
    
    for mission in missions:
        print(f"\n{'='*80}")
        print(f"任务: {mission['mission_id']}")
        print(f"类型: {mission['mission_type']}")
        print(f"描述: {mission['description']}")
        print(f"Ground Truth: Safety={mission['ground_truth']}")
        print(f"{'='*80}")
        
        # 分析轨迹
        trajectory = mission['drones'][0]['trajectory']
        drone_analysis = TrajectoryAnalyzer.analyze_single_drone(trajectory)
        formation_analysis = TrajectoryAnalyzer.analyze_formation(mission['drones'])
        
        # 生成DSL
        dsl = TrajectoryAnalyzer.generate_trajectory_dsl(trajectory)
        print(f"\nDSL片段 (前200字符):")
        print(f"  {dsl[:200]}...")
        
        # 提取事件
        events = TrajectoryAnalyzer.extract_events(trajectory)
        print(f"\n检测到的事件:")
        for event in events[:5]:
            print(f"  - {event}")
        if len(events) > 5:
            print(f"  ... 还有 {len(events)-5} 个事件")
        
        # 计算复杂度
        H_comp = (len(events) / max(1, len(trajectory))) * 10
        print(f"\n复杂度指标:")
        print(f"  - 轨迹点数: {len(trajectory)}")
        print(f"  - 事件数量: {len(events)}")
        print(f"  - 复杂度 H_comp: {H_comp:.3f}")
        print(f"  - 路由层: {'FAST_CONSENSUS' if H_comp < 0.3 else 'DEEP_ANALYSIS' if H_comp < 0.7 else 'META_DEBATE'}")
        
        # 评估
        for name, evaluator in evaluators.items():
            result = evaluator.evaluate(mission)
            results[name].append(result)
            
            print(f"\n{name}评估结果:")
            print(f"  - Safety: {result['safety_label']}")
            print(f"  - Efficiency: {result.get('efficiency_label', 'N/A')}")
            print(f"  - Score: {result['score']}")
            print(f"  - Issues: {len(result.get('issues_identified', []))}")
            
            # 检查是否匹配Ground Truth
            if result['safety_label'] == mission['ground_truth']:
                print(f"  ✓ 匹配 Ground Truth")
            else:
                print(f"  ✗ 不匹配 (期望: {mission['ground_truth']})")
    
    # 统计结果
    print("\n" + "="*80)
    print("评估结果统计")
    print("="*80)
    
    for name in evaluators.keys():
        correct = sum(1 for i, r in enumerate(results[name]) 
                     if r['safety_label'] == missions[i]['ground_truth'])
        accuracy = correct / len(missions) * 100
        print(f"\n{name}:")
        print(f"  - 正确: {correct}/{len(missions)}")
        print(f"  - 准确率: {accuracy:.1f}%")
    
    # 按Ground Truth分类统计
    print("\n" + "="*80)
    print("按Ground Truth分类")
    print("="*80)
    
    for gt in ['Safe', 'Borderline', 'Risky']:
        gt_missions = [m for m in missions if m['ground_truth'] == gt]
        print(f"\n{gt} ({len(gt_missions)} 个任务):")
        for mission in gt_missions:
            print(f"  - {mission['mission_id']}: {mission['description']}")

if __name__ == "__main__":
    test_complex_dataset()
