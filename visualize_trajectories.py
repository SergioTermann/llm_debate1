import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

def load_mission_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_trajectory_2d(mission_data, mission_id=None, save_path=None):
    missions = mission_data['missions']
    
    if mission_id:
        if isinstance(mission_id, list):
            missions = [m for m in missions if m['mission_id'] in mission_id]
        else:
            missions = [m for m in missions if m['mission_id'] == mission_id]
    
    num_missions = len(missions)
    cols = 2
    rows = (num_missions + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8 * rows))
    if num_missions == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, mission in enumerate(missions):
        ax = axes[idx]
        
        for drone_idx, drone in enumerate(mission['drones']):
            trajectory = drone['trajectory']
            lats = [point['latitude'] for point in trajectory]
            lons = [point['longitude'] for point in trajectory]
            alts = [point['altitude'] for point in trajectory]
            
            color = colors[drone_idx % 10]
            
            ax.plot(lons, lats, color=color, linewidth=2, 
                    label=f"{drone['drone_id']}", alpha=0.8)
            
            ax.scatter(lons[0], lats[0], color=color, s=100, marker='s', 
                      edgecolors='black', linewidth=1.5, zorder=5)
            ax.scatter(lons[-1], lats[-1], color=color, s=100, marker='^', 
                      edgecolors='black', linewidth=1.5, zorder=5)
            
            for i in range(0, len(trajectory), len(trajectory)//5):
                heading = trajectory[i]['heading']
                dx = 0.00002 * np.cos(np.radians(heading))
                dy = 0.00002 * np.sin(np.radians(heading))
                ax.arrow(lons[i], lats[i], dx, dy, 
                        head_width=0.000005, head_length=0.000003,
                        fc=color, ec=color, alpha=0.6)
        
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(f"{mission['mission_id']}\n{mission['description']}", 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    for idx in range(num_missions, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()

def plot_trajectory_3d(mission_data, mission_id=None, save_path=None):
    missions = mission_data['missions']
    
    if mission_id:
        missions = [m for m in missions if m['mission_id'] == mission_id]
    
    for mission in missions:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for drone_idx, drone in enumerate(mission['drones']):
            trajectory = drone['trajectory']
            lats = np.array([point['latitude'] for point in trajectory])
            lons = np.array([point['longitude'] for point in trajectory])
            alts = np.array([point['altitude'] for point in trajectory])
            
            color = colors[drone_idx % 10]
            
            ax.plot(lons, lats, alts, color=color, linewidth=2, 
                   label=f"{drone['drone_id']}", alpha=0.8)
            
            ax.scatter(lons[0], lats[0], alts[0], color=color, s=100, 
                      marker='s', edgecolors='black', linewidth=1.5, zorder=5)
            ax.scatter(lons[-1], lats[-1], alts[-1], color=color, s=100, 
                      marker='^', edgecolors='black', linewidth=1.5, zorder=5)
        
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_zlabel('Altitude (m)', fontsize=11)
        ax.set_title(f"{mission['mission_id']} - 3D Trajectory\n{mission['description']}", 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_file = save_path.replace('.png', f'_{mission["mission_id"]}_3d.png')
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_file}")
        
        plt.show()

def plot_mission_comparison(mission_data, mission_ids, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, mission_id in enumerate(mission_ids[:4]):
        ax = axes[idx]
        mission = next((m for m in mission_data['missions'] if m['mission_id'] == mission_id), None)
        
        if not mission:
            continue
        
        for drone_idx, drone in enumerate(mission['drones']):
            trajectory = drone['trajectory']
            lats = [point['latitude'] for point in trajectory]
            lons = [point['longitude'] for point in trajectory]
            
            color = colors[drone_idx % 10]
            
            ax.plot(lons, lats, color=color, linewidth=2, 
                   label=f"{drone['drone_id']}", alpha=0.8)
            
            ax.scatter(lons[0], lats[0], color=color, s=80, marker='s', 
                      edgecolors='black', linewidth=1.5, zorder=5)
            ax.scatter(lons[-1], lats[-1], color=color, s=80, marker='^', 
                      edgecolors='black', linewidth=1.5, zorder=5)
        
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(f"{mission['mission_id']}\n{mission['description']}", 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()

def main():
    data_file = 'experiments/complex_uav_missions.json'
    mission_data = load_mission_data(data_file)
    
    print("Available missions:")
    for mission in mission_data['missions'][:10]:
        print(f"  - {mission['mission_id']}: {mission['description']}")
    
    print("\nPlotting typical missions (2D)...")
    typical_missions = ['MISSION_01_SURVEILLANCE', 'MISSION_02_FORMATION', 
                       'MISSION_03_SEARCH_RESCUE', 'MISSION_04_ADVERSARIAL']
    plot_trajectory_2d(mission_data, mission_id=typical_missions, 
                      save_path='trajectory_typical_2d.png')
    
    print("\nPlotting first mission (3D)...")
    plot_trajectory_3d(mission_data, mission_id='MISSION_01_SURVEILLANCE', 
                      save_path='trajectory_3d.png')
    
    print("\nPlotting mission comparison...")
    mission_ids = ['MISSION_01_SURVEILLANCE', 'MISSION_02_FORMATION', 
                  'MISSION_03_SEARCH_RESCUE', 'MISSION_04_ADVERSARIAL']
    plot_mission_comparison(mission_data, mission_ids, save_path='trajectory_comparison.png')

if __name__ == '__main__':
    main()
