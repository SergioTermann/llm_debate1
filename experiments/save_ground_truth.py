import json

data = json.load(open('complex_uav_missions.json', encoding='utf-8'))

with open('ground_truth_display.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("COMPLEX_UAV_MISSIONS.GROUND_TRUTH 详细信息\n")
    f.write("=" * 100 + "\n")
    
    for i, mission in enumerate(data['missions']):
        f.write(f"\n{'=' * 100}\n")
        f.write(f"任务 {i+1}: {mission['mission_id']}\n")
        f.write(f"类型: {mission['mission_type']}\n")
        f.write(f"描述: {mission['description']}\n")
        f.write(f"{'=' * 100}\n")
        f.write(f"\nGround Truth:\n")
        gt = mission['ground_truth']
        f.write(f"  安全性: {gt['safety_label']} (评分: {gt['safety_score']:.2f})\n")
        f.write(f"  效率: {gt['efficiency_label']} (评分: {gt['efficiency_score']:.2f})\n")
        f.write(f"  编队稳定性: {gt['formation_stability']:.2f}\n")
        f.write(f"  协调质量: {gt['coordination_quality']:.2f}\n")
        f.write(f"  轨迹平滑度: {gt['trajectory_smoothness']:.2f}\n")
        f.write(f"  高度稳定性: {gt['altitude_stability']:.2f}\n")
        f.write(f"  速度一致性: {gt['speed_consistency']:.2f}\n")
        f.write(f"  任务复杂度: {gt['mission_complexity']:.2f}\n")
        
        if gt['risk_factors']:
            f.write(f"\n  风险因素 ({len(gt['risk_factors'])} 个):\n")
            for rf in gt['risk_factors']:
                f.write(f"    - {rf['drone_id']}: {rf['type']} ({rf['severity']}, 值={rf['value']:.2f})\n")
        
        if gt['unobservable_issues']:
            f.write(f"\n  不可观测问题 ({len(gt['unobservable_issues'])} 个):\n")
            for ui in gt['unobservable_issues']:
                f.write(f"    - {ui['drone_id']}: {ui['type']} ({ui['severity']}, 计数={ui['count']})\n")
    
    f.write(f"\n{'=' * 100}\n")
    f.write(f"总计: {len(data['missions'])} 个任务\n")
    f.write(f"{'=' * 100}\n")

print("Ground Truth信息已保存到 ground_truth_display.txt")
