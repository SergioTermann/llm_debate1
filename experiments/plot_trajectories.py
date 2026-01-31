"""
绘制所有复杂协同任务的轨迹
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import os

def plot_all_trajectories():
    """绘制所有任务的轨迹"""
    
    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "complex_uav_missions.json")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    missions = data['missions']
    
    # 创建大图，每个任务一个子图
    num_missions = len(missions)
    cols = 3
    rows = (num_missions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))
    fig.suptitle('Complex Multi-UAV Collaborative Missions - All Trajectories', 
                 fontsize=16, fontweight='bold')
    
    # 颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for idx, mission in enumerate(missions):
        ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx]
        
        # 绘制每个无人机的轨迹
        drones = mission['drones']
        num_drones = len(drones)
        
        for drone_idx, drone in enumerate(drones):
            trajectory = drone['trajectory']
            
            # 提取坐标
            lats = [p['latitude'] for p in trajectory]
            lons = [p['longitude'] for p in trajectory]
            alts = [p['altitude'] for p in trajectory]
            
            # 绘制轨迹（俯视图）
            color = colors[drone_idx % 20]
            ax.plot(lons, lats, 'o-', color=color, 
                   linewidth=1.5, markersize=2, alpha=0.7,
                   label=f'UAV_{drone_idx+1}' if drone_idx < 3 else '')
            
            # 标记不可观测点（GPS故障或信号丢失）
            for i, point in enumerate(trajectory):
                if point.get('gps_status') == 'DRIFT':
                    ax.scatter(point['longitude'], point['latitude'], 
                              marker='x', color='purple', s=20, zorder=10)
                elif point.get('signal_status') == 'LOST':
                    ax.scatter(point['longitude'], point['latitude'], 
                              marker='s', color='gray', s=20, zorder=10, alpha=0.5)
            
            # 标记起点和终点
            ax.scatter(lons[0], lats[0], marker='s', color='green', 
                      s=50, zorder=5, edgecolors='black', linewidths=1)
            ax.scatter(lons[-1], lats[-1], marker='^', color='red', 
                      s=50, zorder=5, edgecolors='black', linewidths=1)
        
        # 设置标题和标签
        gt_color = {'Safe': 'green', 'Borderline': 'orange', 'Risky': 'red'}
        gt = mission['ground_truth']
        title = f"{mission['mission_id']}\n{mission['mission_type']}\n"
        title += f"({mission['num_drones']} UAVs, GT: {gt})"
        ax.set_title(title, fontsize=10, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor=gt_color[gt], alpha=0.3))
        
        ax.set_xlabel('Longitude', fontsize=8)
        ax.set_ylabel('Latitude', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        
        # 保持比例一致
        ax.set_aspect('equal')
    
    # 删除多余的子图
    for idx in range(num_missions, rows * cols):
        if rows > 1:
            fig.delaxes(axes[idx // cols, idx % cols])
        else:
            fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(script_dir, 'all_trajectories_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存到: {output_path}")
    
    plt.show()

def plot_mission_detail(mission_idx=0):
    """绘制单个任务的详细轨迹（3D视图）"""
    
    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "complex_uav_missions.json")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mission = data['missions'][mission_idx]
    drones = mission['drones']
    
    # 创建3D图
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # 绘制每个无人机的轨迹
    for drone_idx, drone in enumerate(drones):
        trajectory = drone['trajectory']
        
        # 提取坐标
        lats = [p['latitude'] for p in trajectory]
        lons = [p['longitude'] for p in trajectory]
        alts = [p['altitude'] for p in trajectory]
        
        # 绘制3D轨迹
        color = colors[drone_idx % 20]
        ax.plot(lons, lats, alts, 'o-', color=color, 
               linewidth=2, markersize=3, alpha=0.8,
               label=f'UAV_{drone_idx+1}')
        
        # 标记起点和终点
        ax.scatter(lons[0], lats[0], alts[0], marker='s', color='green', 
                  s=100, zorder=5, edgecolors='black', linewidths=2)
        ax.scatter(lons[-1], lats[-1], alts[-1], marker='^', color='red', 
                  s=100, zorder=5, edgecolors='black', linewidths=2)
    
    # 设置标题和标签
    gt_color = {'Safe': 'green', 'Borderline': 'orange', 'Risky': 'red'}
    gt = mission['ground_truth']
    title = f"{mission['mission_id']}: {mission['mission_type']}\n"
    title += f"{mission['description']}\n"
    title += f"({mission['num_drones']} UAVs, Ground Truth: {gt})"
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_zlabel('Altitude (m)', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    # 保存图片
    output_path = os.path.join(script_dir, f'mission_{mission_idx+1}_detail_3d.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存到: {output_path}")
    
    plt.show()

def plot_mission_2d(mission_idx=0):
    """绘制单个任务的2D轨迹（俯视图和侧视图）"""
    
    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "complex_uav_missions.json")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mission = data['missions'][mission_idx]
    drones = mission['drones']
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # 俯视图
    ax1 = axes[0]
    for drone_idx, drone in enumerate(drones):
        trajectory = drone['trajectory']
        lats = [p['latitude'] for p in trajectory]
        lons = [p['longitude'] for p in trajectory]
        
        color = colors[drone_idx % 20]
        ax1.plot(lons, lats, 'o-', color=color, 
                linewidth=2, markersize=3, alpha=0.8,
                label=f'UAV_{drone_idx+1}' if drone_idx < 3 else '')
        
        # 标记不可观测点（GPS故障或信号丢失）
        for i, point in enumerate(trajectory):
            if point.get('gps_status') == 'DRIFT':
                ax1.scatter(point['longitude'], point['latitude'], 
                          marker='x', color='purple', s=30, zorder=10)
            elif point.get('signal_status') == 'LOST':
                ax1.scatter(point['longitude'], point['latitude'], 
                          marker='s', color='gray', s=30, zorder=10, alpha=0.5)
        
        ax1.scatter(lons[0], lats[0], marker='s', color='green', 
                  s=80, zorder=5, edgecolors='black', linewidths=1.5)
        ax1.scatter(lons[-1], lats[-1], marker='^', color='red', 
                  s=80, zorder=5, edgecolors='black', linewidths=1.5)
    
    # 添加图例说明
    ax1.scatter([], [], marker='x', color='purple', s=30, label='GPS Drift')
    ax1.scatter([], [], marker='s', color='gray', s=30, alpha=0.5, label='Signal Lost')
    
    ax1.set_xlabel('Longitude', fontsize=10)
    ax1.set_ylabel('Latitude', fontsize=10)
    ax1.set_title('Top View (Latitude vs Longitude)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=7, ncol=3)
    ax1.set_aspect('equal')
    
    # 侧视图（高度 vs 经度）
    ax2 = axes[1]
    for drone_idx, drone in enumerate(drones):
        trajectory = drone['trajectory']
        lons = [p['longitude'] for p in trajectory]
        alts = [p['altitude'] for p in trajectory]
        
        color = colors[drone_idx % 20]
        ax2.plot(lons, alts, 'o-', color=color, 
                linewidth=2, markersize=3, alpha=0.8,
                label=f'UAV_{drone_idx+1}' if drone_idx < 3 else '')
        
        # 标记不可观测点
        for i, point in enumerate(trajectory):
            if point.get('gps_status') == 'DRIFT':
                ax2.scatter(point['longitude'], point['altitude'], 
                          marker='x', color='purple', s=30, zorder=10)
            elif point.get('signal_status') == 'LOST':
                ax2.scatter(point['longitude'], point['altitude'], 
                          marker='s', color='gray', s=30, zorder=10, alpha=0.5)
        
        ax2.scatter(lons[0], alts[0], marker='s', color='green', 
                  s=80, zorder=5, edgecolors='black', linewidths=1.5)
        ax2.scatter(lons[-1], alts[-1], marker='^', color='red', 
                  s=80, zorder=5, edgecolors='black', linewidths=1.5)
    
    # 添加图例说明
    ax2.scatter([], [], marker='x', color='purple', s=30, label='GPS Drift')
    ax2.scatter([], [], marker='s', color='gray', s=30, alpha=0.5, label='Signal Lost')
    
    ax2.set_xlabel('Longitude', fontsize=10)
    ax2.set_ylabel('Altitude (m)', fontsize=10)
    ax2.set_title('Side View (Altitude vs Longitude)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=7, ncol=3)
    
    # 设置总标题
    gt_color = {'Safe': 'green', 'Borderline': 'orange', 'Risky': 'red'}
    gt = mission['ground_truth']
    title = f"{mission['mission_id']}: {mission['mission_type']}\n"
    title += f"{mission['description']} (Ground Truth: {gt})"
    fig.suptitle(title, fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor=gt_color[gt], alpha=0.3))
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(script_dir, f'mission_{mission_idx+1}_detail_2d.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存到: {output_path}")
    
    plt.show()

def plot_all_missions_2d():
    """绘制所有任务的2D详细视图"""
    
    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "complex_uav_missions.json")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    missions = data['missions']
    
    # 为每个任务创建一个图
    for mission_idx, mission in enumerate(missions):
        plot_mission_2d(mission_idx)

if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("绘制复杂协同任务轨迹")
    print("="*80)
    
    # 从命令行参数获取选择
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\n选择绘制模式:")
        print("1. 绘制所有任务概览（俯视图）")
        print("2. 绘制单个任务详细3D视图")
        print("3. 绘制单个任务详细2D视图")
        print("4. 绘制所有任务详细2D视图")
        choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == '1':
        print("\n绘制所有任务概览...")
        plot_all_trajectories()
    elif choice == '2':
        if len(sys.argv) > 2:
            mission_idx = int(sys.argv[2]) - 1
        else:
            mission_idx = int(input("\n请输入任务编号 (1-10): ")) - 1
        print(f"\n绘制任务 {mission_idx+1} 的3D详细视图...")
        plot_mission_detail(mission_idx)
    elif choice == '3':
        if len(sys.argv) > 2:
            mission_idx = int(sys.argv[2]) - 1
        else:
            mission_idx = int(input("\n请输入任务编号 (1-10): ")) - 1
        print(f"\n绘制任务 {mission_idx+1} 的2D详细视图...")
        plot_mission_2d(mission_idx)
    elif choice == '4':
        print("\n绘制所有任务详细2D视图...")
        plot_all_missions_2d()
    else:
        print("无效选择！")
