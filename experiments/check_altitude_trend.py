import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

mission = data['missions'][9]  # COMPLEX_10
print(f"Mission: {mission['mission_id']}")
print(f"Ground Truth: Safety={mission['ground_truth']['safety_label']}")
print(f"\nDrone 0 Trajectory (first 5 points):")
for i, point in enumerate(mission['drones'][0]['trajectory'][:5]):
    print(f"  {i}: timestamp={point.get('timestamp', 'N/A')}, x={point['x']:.1f}, y={point['y']:.1f}, z={point['z']:.1f}")

print(f"\nDrone 0 Trajectory (last 5 points):")
for i, point in enumerate(mission['drones'][0]['trajectory'][-5:]):
    print(f"  {len(mission['drones'][0]['trajectory'])-5+i}: timestamp={point.get('timestamp', 'N/A')}, x={point['x']:.1f}, y={point['y']:.1f}, z={point['z']:.1f}")

print(f"\nTotal points: {len(mission['drones'][0]['trajectory'])}")
