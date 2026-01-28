"""
可视化UAV轨迹数据，生成多种图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os


def load_dataset(file_path: str) -> Dict:
    """加载数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_trajectory_2d(mission: Dict, save_path: str = None):
    """绘制2D轨迹图（俯视图）"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 为每个无人机绘制轨迹
    colors = plt.cm.tab10(np.linspace(0, 1, len(mission['drones'])))
    
    for i, drone in enumerate(mission['drones']):
        trajectory = drone['trajectory']
        lats = [p['gps']['lat'] for p in trajectory]
        lons = [p['gps']['lon'] for p in trajectory]
        headings = [p['heading'] for p in trajectory]
        
        # 绘制轨迹线
        ax.plot(lons, lats, 'o-', color=colors[i], 
                label=f"{drone['id']}", linewidth=2, markersize=4)
        
        # 绘制起点和终点
        ax.scatter(lons[0], lats[0], color='green', s=100, marker='s', 
                   edgecolors='black', linewidth=2, zorder=5)
        ax.scatter(lons[-1], lats[-1], color='red', s=100, marker='^', 
                   edgecolors='black', linewidth=2, zorder=5)
        
        # 绘制航向箭头（每隔几个点）
        for j in range(0, len(trajectory), 3):
            heading_rad = np.radians(headings[j])
            arrow_length = 0.00002
            dx = arrow_length * np.cos(heading_rad)
            dy = arrow_length * np.sin(heading_rad)
            ax.arrow(lons[j], lats[j], dx, dy, 
                    head_width=0.000005, head_length=0.000005,
                    fc=colors[i], ec=colors[i], alpha=0.5)
    
    # 设置标题和标签
    ax.set_title(f"2D Trajectory - {mission['mission_id']} ({mission['mission_type']})\n"
                 f"Ground Truth: {mission['ground_truth']} | "
                 f"Total Score: {mission['weighted_scores']['total_score']:.1f}", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # 添加图例说明
    ax.scatter([], [], color='green', s=100, marker='s', 
               edgecolors='black', linewidth=2, label='Start')
    ax.scatter([], [], color='red', s=100, marker='^', 
               edgecolors='black', linewidth=2, label='End')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 2D轨迹图已保存: {save_path}")
    
    plt.close()


def plot_trajectory_3d(mission: Dict, save_path: str = None):
    """绘制3D轨迹图"""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(mission['drones'])))
    
    for i, drone in enumerate(mission['drones']):
        trajectory = drone['trajectory']
        lats = [p['gps']['lat'] for p in trajectory]
        lons = [p['gps']['lon'] for p in trajectory]
        alts = [p['altitude'] for p in trajectory]
        
        # 绘制3D轨迹
        ax.plot(lons, lats, alts, 'o-', color=colors[i], 
                label=f"{drone['id']}", linewidth=2, markersize=4)
        
        # 标记起点和终点
        ax.scatter(lons[0], lats[0], alts[0], color='green', s=100, 
                   marker='s', edgecolors='black', linewidth=2, zorder=5)
        ax.scatter(lons[-1], lats[-1], alts[-1], color='red', s=100, 
                   marker='^', edgecolors='black', linewidth=2, zorder=5)
    
    ax.set_title(f"3D Trajectory - {mission['mission_id']} ({mission['mission_type']})\n"
                 f"Ground Truth: {mission['ground_truth']} | "
                 f"Total Score: {mission['weighted_scores']['total_score']:.1f}", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_zlabel('Altitude (m)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 3D轨迹图已保存: {save_path}")
    
    plt.close()


def plot_altitude_profile(mission: Dict, save_path: str = None):
    """绘制高度变化曲线"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(mission['drones'])))
    
    for i, drone in enumerate(mission['drones']):
        trajectory = drone['trajectory']
        timestamps = [p['timestamp'] for p in trajectory]
        altitudes = [p['altitude'] for p in trajectory]
        
        ax.plot(timestamps, altitudes, 'o-', color=colors[i], 
                label=f"{drone['id']}", linewidth=2, markersize=4)
    
    ax.set_title(f"Altitude Profile - {mission['mission_id']}", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Altitude (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 高度曲线图已保存: {save_path}")
    
    plt.close()


def plot_speed_profile(mission: Dict, save_path: str = None):
    """绘制速度变化曲线"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(mission['drones'])))
    
    for i, drone in enumerate(mission['drones']):
        trajectory = drone['trajectory']
        timestamps = [p['timestamp'] for p in trajectory]
        speeds = [p['speed'] for p in trajectory]
        
        ax.plot(timestamps, speeds, 'o-', color=colors[i], 
                label=f"{drone['id']}", linewidth=2, markersize=4)
    
    ax.set_title(f"Speed Profile - {mission['mission_id']}", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Speed (m/s)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 速度曲线图已保存: {save_path}")
    
    plt.close()


def plot_heading_profile(mission: Dict, save_path: str = None):
    """绘制航向变化曲线"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(mission['drones'])))
    
    for i, drone in enumerate(mission['drones']):
        trajectory = drone['trajectory']
        timestamps = [p['timestamp'] for p in trajectory]
        headings = [p['heading'] for p in trajectory]
        
        ax.plot(timestamps, headings, 'o-', color=colors[i], 
                label=f"{drone['id']}", linewidth=2, markersize=4)
    
    ax.set_title(f"Heading Profile - {mission['mission_id']}", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Heading (degrees)', fontsize=12)
    ax.set_ylim(0, 360)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 航向曲线图已保存: {save_path}")
    
    plt.close()


def plot_mission_summary(mission: Dict, save_path: str = None):
    """绘制任务摘要图（包含所有信息）"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 2D轨迹
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.tab10(np.linspace(0, 1, len(mission['drones'])))
    
    for i, drone in enumerate(mission['drones']):
        trajectory = drone['trajectory']
        lats = [p['gps']['lat'] for p in trajectory]
        lons = [p['gps']['lon'] for p in trajectory]
        ax1.plot(lons, lats, 'o-', color=colors[i], 
                label=f"{drone['id']}", linewidth=2, markersize=3)
    
    ax1.set_title(f"2D Trajectory - {mission['mission_id']}", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8)
    
    # 高度曲线
    ax2 = fig.add_subplot(gs[1, 0])
    for i, drone in enumerate(mission['drones']):
        trajectory = drone['trajectory']
        timestamps = [p['timestamp'] for p in trajectory]
        altitudes = [p['altitude'] for p in trajectory]
        ax2.plot(timestamps, altitudes, 'o-', color=colors[i], 
                label=f"{drone['id']}", linewidth=2, markersize=3)
    ax2.set_title('Altitude Profile', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=8)
    
    # 速度曲线
    ax3 = fig.add_subplot(gs[1, 1])
    for i, drone in enumerate(mission['drones']):
        trajectory = drone['trajectory']
        timestamps = [p['timestamp'] for p in trajectory]
        speeds = [p['speed'] for p in trajectory]
        ax3.plot(timestamps, speeds, 'o-', color=colors[i], 
                label=f"{drone['id']}", linewidth=2, markersize=3)
    ax3.set_title('Speed Profile', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Speed (m/s)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=8)
    
    # 航向曲线
    ax4 = fig.add_subplot(gs[2, 0])
    for i, drone in enumerate(mission['drones']):
        trajectory = drone['trajectory']
        timestamps = [p['timestamp'] for p in trajectory]
        headings = [p['heading'] for p in trajectory]
        ax4.plot(timestamps, headings, 'o-', color=colors[i], 
                label=f"{drone['id']}", linewidth=2, markersize=3)
    ax4.set_title('Heading Profile', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Heading (degrees)')
    ax4.set_ylim(0, 360)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=8)
    
    # 评分信息
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    scores = mission['weighted_scores']
    info_text = f"""
    Mission Information
    {'='*30}
    Mission ID: {mission['mission_id']}
    Type: {mission['mission_type']}
    Drones: {mission['drone_count']}
    Duration: {mission['flight_duration']}
    Ground Truth: {mission['ground_truth']}
    
    Scores
    {'='*30}
    Total Score: {scores['total_score']:.1f}
    Flight Control: {scores['flight_control']['score']:.1f}
    Swarm Coordination: {scores['swarm_coordination']['score']:.1f}
    Safety Assessment: {scores['safety_assessment']['score']:.1f}
    
    Anomalies: {len(mission['anomalies'])}
    """
    
    if mission['anomalies']:
        info_text += "\nAnomaly List:\n"
        for anomaly in mission['anomalies']:
            info_text += f"  - {anomaly}\n"
    
    ax5.text(0.1, 0.9, info_text, transform=ax5.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f"Mission Summary - {mission['mission_id']} ({mission['mission_type']})\n"
                f"Ground Truth: {mission['ground_truth']} | "
                f"Total Score: {scores['total_score']:.1f}", 
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 任务摘要图已保存: {save_path}")
    
    plt.close()


def visualize_all_missions(dataset: Dict, output_dir: str = "trajectory_visualizations"):
    """可视化所有任务"""
    
    print("="*80)
    print("开始可视化所有任务")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    missions = dataset['missions']
    
    for i, mission in enumerate(missions, 1):
        print(f"\n[{i}/{len(missions)}] 可视化任务: {mission['mission_id']}")
        
        mission_dir = os.path.join(output_dir, mission['mission_id'])
        os.makedirs(mission_dir, exist_ok=True)
        
        # 生成各种可视化
        plot_trajectory_2d(mission, os.path.join(mission_dir, "trajectory_2d.png"))
        plot_trajectory_3d(mission, os.path.join(mission_dir, "trajectory_3d.png"))
        plot_altitude_profile(mission, os.path.join(mission_dir, "altitude_profile.png"))
        plot_speed_profile(mission, os.path.join(mission_dir, "speed_profile.png"))
        plot_heading_profile(mission, os.path.join(mission_dir, "heading_profile.png"))
        plot_mission_summary(mission, os.path.join(mission_dir, "mission_summary.png"))
    
    print(f"\n✅ 所有任务可视化完成！")
    print(f"   输出目录: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    # 加载数据集
    dataset_file = "improved_uav_missions.json"
    
    if not os.path.exists(dataset_file):
        print(f"❌ 数据集文件不存在: {dataset_file}")
        print("   请先运行 generate_improved_dataset.py 生成数据集")
        exit(1)
    
    dataset = load_dataset(dataset_file)
    
    # 可视化所有任务
    visualize_all_missions(dataset, output_dir="trajectory_visualizations")
