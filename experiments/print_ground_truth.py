import json

data = json.load(open('complex_uav_missions.json', encoding='utf-8'))

print("=" * 100)
print("COMPLEX_UAV_MISSIONS.GROUND_TRUTH 详细信息")
print("=" * 100)

for i, mission in enumerate(data['missions']):
    print(f"\n{'=' * 100}")
    print(f"任务 {i+1}: {mission['mission_id']}")
    print(f"类型: {mission['mission_type']}")
    print(f"描述: {mission['description']}")
    print(f"{'=' * 100}")
    print(f"\n📊 Ground Truth:")
    gt = mission['ground_truth']
    print(f"  安全性: {gt['safety_label']} (评分: {gt['safety_score']:.2f})")
    print(f"  效率: {gt['efficiency_label']} (评分: {gt['efficiency_score']:.2f})")
    print(f"  编队稳定性: {gt['formation_stability']:.2f}")
    print(f"  协调质量: {gt['coordination_quality']:.2f}")
    print(f"  轨迹平滑度: {gt['trajectory_smoothness']:.2f}")
    print(f"  高度稳定性: {gt['altitude_stability']:.2f}")
    print(f"  速度一致性: {gt['speed_consistency']:.2f}")
    print(f"  任务复杂度: {gt['mission_complexity']:.2f}")
    
    if gt['risk_factors']:
        print(f"\n  ⚠️  风险因素 ({len(gt['risk_factors'])} 个):")
        for rf in gt['risk_factors']:
            print(f"    - {rf['drone_id']}: {rf['type']} ({rf['severity']}, 值={rf['value']:.2f})")
    
    if gt['unobservable_issues']:
        print(f"\n  📡 不可观测问题 ({len(gt['unobservable_issues'])} 个):")
        for ui in gt['unobservable_issues']:
            print(f"    - {ui['drone_id']}: {ui['type']} ({ui['severity']}, 计数={ui['count']})")

print(f"\n{'=' * 100}")
print(f"总计: {len(data['missions'])} 个任务")
print(f"{'=' * 100}")
