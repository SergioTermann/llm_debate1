import json
with open('complex_uav_missions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Total missions: {len(data['missions'])}")
for mission in data['missions']:
    print(f"{mission['mission_id']}: {mission['ground_truth']['safety_label']}")
