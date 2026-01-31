"""
测试DSL Tokenization功能
基于论文中的Trajectory DSL规范
"""

import json
from exp1_real_evaluation import TrajectoryAnalyzer

def test_dsl_tokenization():
    """测试DSL tokenization功能"""
    
    # 加载测试数据
    with open('..\\scenario_uav_missions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 测试第一个任务
    mission = data['missions'][0]
    trajectory = mission['drones'][0]['trajectory']
    
    print("="*80)
    print(f"测试任务: {mission['mission_id']}")
    print(f"轨迹点数: {len(trajectory)}")
    print("="*80)
    
    # 1. 测试轨迹分割
    print("\n【1】轨迹分割 (Segmentation)")
    print("-"*80)
    segments = TrajectoryAnalyzer.segment_trajectory(trajectory)
    print(f"分割成 {len(segments)} 个段:")
    for seg in segments:
        print(f"  SEG[{seg['id']}]: t={seg['start_idx']}-{seg['end_idx']}, duration={seg['duration']}, points={len(seg['points'])}")
    
    # 2. 测试事件提取
    print("\n【2】事件提取 (Event Extraction)")
    print("-"*80)
    events = TrajectoryAnalyzer.extract_events(trajectory)
    print(f"检测到 {len(events)} 个关键事件:")
    for event in events:
        print(f"  EVENT: t={event['time']}, type={event['type']}, value={event['value']:.1f}")
        print(f"    {event['description']}")
    
    # 3. 测试关注区域识别
    print("\n【3】关注区域识别 (Attention Regions)")
    print("-"*80)
    attn_regions = TrajectoryAnalyzer.identify_attention_regions(trajectory)
    print(f"识别到 {len(attn_regions)} 个关注区域:")
    for attn in attn_regions:
        print(f"  ATTN: t={attn['time_start']}-{attn['time_end']}")
        print(f"    reason: {attn['reason']}")
    
    # 4. 测试完整的DSL生成
    print("\n【4】完整DSL生成 (DSL Generation)")
    print("="*80)
    dsl_output = TrajectoryAnalyzer.generate_trajectory_dsl(trajectory)
    print(dsl_output)
    
    # 5. 统计token数量
    print("\n【5】Token统计")
    print("-"*80)
    dsl_tokens = dsl_output.split()
    print(f"DSL输出总token数: {len(dsl_tokens)}")
    print(f"原始轨迹点数: {len(trajectory)}")
    print(f"压缩比: {len(trajectory) / len(dsl_tokens):.2f}x")
    
    # 6. 测试多个任务
    print("\n【6】多个任务的DSL示例")
    print("="*80)
    for i in range(min(3, len(data['missions']))):
        mission = data['missions'][i]
        trajectory = mission['drones'][0]['trajectory']
        dsl = TrajectoryAnalyzer.generate_trajectory_dsl(trajectory)
        
        print(f"\n任务 {i+1}: {mission['mission_id']}")
        print(f"轨迹点数: {len(trajectory)}")
        print(f"DSL长度: {len(dsl.split())} tokens")
        print(f"压缩比: {len(trajectory) / len(dsl.split()):.2f}x")
        
        # 显示前3个段
        lines = dsl.split('\n')
        for line in lines[:5]:
            if line.strip():
                print(f"  {line}")

if __name__ == "__main__":
    test_dsl_tokenization()
