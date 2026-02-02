import json

data = json.load(open('complex_uav_missions.json', encoding='utf-8'))

output = []
output.append("=" * 80)
output.append("COMPLEX_UAV_MISSIONS.GROUND_TRUTH 示例")
output.append("=" * 80)

for i, mission in enumerate(data['missions']):
    output.append(f"\n任务 {i+1}: {mission['mission_id']}")
    output.append(f"类型: {mission['mission_type']}")
    output.append(f"描述: {mission['description']}")
    output.append(f"Ground Truth:")
    gt = mission['ground_truth']
    output.append(f"  安全性: {gt['safety_label']} (评分: {gt['safety_score']:.2f})")
    output.append(f"  效率: {gt['efficiency_label']} (评分: {gt['efficiency_score']:.2f})")
    output.append(f"  编队稳定性: {gt['formation_stability']:.2f}")
    output.append(f"  协调质量: {gt['coordination_quality']:.2f}")
    output.append(f"  轨迹平滑度: {gt['trajectory_smoothness']:.2f}")
    output.append(f"  高度稳定性: {gt['altitude_stability']:.2f}")
    output.append(f"  速度一致性: {gt['speed_consistency']:.2f}")
    output.append(f"  任务复杂度: {gt['mission_complexity']:.2f}")
    
    if gt['risk_factors']:
        output.append(f"  风险因素 ({len(gt['risk_factors'])} 个):")
        for rf in gt['risk_factors'][:3]:
            output.append(f"    - {rf['drone_id']}: {rf['type']} ({rf['severity']}, 值={rf['value']:.2f})")
        if len(gt['risk_factors']) > 3:
            output.append(f"    ... 还有 {len(gt['risk_factors']) - 3} 个")
    
    if gt['unobservable_issues']:
        output.append(f"  不可观测问题 ({len(gt['unobservable_issues'])} 个):")
        for ui in gt['unobservable_issues'][:3]:
            output.append(f"    - {ui['drone_id']}: {ui['type']} ({ui['severity']}, 计数={ui['count']})")
        if len(gt['unobservable_issues']) > 3:
            output.append(f"    ... 还有 {len(gt['unobservable_issues']) - 3} 个")

with open('ground_truth_output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print("输出已保存到 ground_truth_output.txt")
