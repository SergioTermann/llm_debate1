import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "complex_uav_missions.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

mission = data['missions'][9]  # COMPLEX_10
print(f"Mission: {mission['mission_id']}")
print(f"Description: {mission['description']}")
print(f"Ground Truth: Safety={mission['ground_truth']['safety_label']}")

trajectory = mission['drones'][0]['trajectory']
print(f"\nTotal points: {len(trajectory)}")

# 分析高度变化
altitudes = [p['altitude'] for p in trajectory]
min_alt = min(altitudes)
max_alt = max(altitudes)
alt_range = max_alt - min_alt

print(f"Altitude range: {min_alt:.1f}m -> {max_alt:.1f}m (Δ={alt_range:.1f}m)")

# 检查最后20%的点
last_20_percent = int(len(trajectory) * 0.8)
last_altitudes = altitudes[last_20_percent:]
print(f"\nLast 20% of trajectory:")
print(f"  Min: {min(last_altitudes):.1f}m")
print(f"  Max: {max(last_altitudes):.1f}m")
print(f"  Range: {max(last_altitudes) - min(last_altitudes):.1f}m")

# 检查最后10个点
print(f"\nLast 10 points:")
for i, point in enumerate(trajectory[-10:]):
    print(f"  {len(trajectory)-10+i}: altitude={point['altitude']:.1f}m, speed={point['speed']:.1f}m/s")
