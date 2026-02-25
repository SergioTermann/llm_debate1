import json
import random
import copy
import math
import os
from typing import List, Dict

def load_missions(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'missions' in data:
        return data['missions']
    return [data]

def save_missions(missions, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"missions": missions}, f, indent=2, ensure_ascii=False)

def add_gaussian_noise(val, sigma):
    return val + random.gauss(0, sigma)

def degrade_formation(mission):
    """Scenario 1: Borderline Formation Stability (Degrade by ~30%)"""
    new_mission = copy.deepcopy(mission)
    new_mission['mission_id'] += "_HARD_FORMATION"
    new_mission['ground_truth']['safety_label'] = "Borderline" # Or Risky depending on severity
    
    for drone in new_mission['drones']:
        for point in drone['trajectory']:
            # Add random position noise to simulate poor formation keeping
            point['latitude'] += random.gauss(0, 0.00005) # ~5m error
            point['longitude'] += random.gauss(0, 0.00005)
            
    return new_mission

def inject_gps_drift(mission):
    """Scenario 2: GPS Drift Bursts (Unobservable Issue)"""
    new_mission = copy.deepcopy(mission)
    new_mission['mission_id'] += "_HARD_GPS_DRIFT"
    # If drift is significant, it should be Risky, but if short, maybe Borderline/Safe
    # We'll make it significant enough to be Risky but "hard" because other metrics look fine
    new_mission['ground_truth']['safety_label'] = "Risky"
    
    duration = len(new_mission['drones'][0]['trajectory'])
    start_drift = random.randint(int(duration * 0.3), int(duration * 0.7))
    drift_len = random.randint(5, 15)
    
    target_drone = random.choice(new_mission['drones'])
    
    for i in range(start_drift, min(start_drift + drift_len, duration)):
        point = target_drone['trajectory'][i]
        point['gps_status'] = "DRIFT"
        # Drift position significantly
        point['latitude'] += 0.0002 * (i - start_drift + 1) # ~20m drift increasing
        point['longitude'] += 0.0002 * (i - start_drift + 1)
        
    return new_mission

def create_conflicting_signals(mission):
    """Scenario 3: Conflicting Signals (High Smoothness, Poor Formation)"""
    new_mission = copy.deepcopy(mission)
    new_mission['mission_id'] += "_HARD_CONFLICT"
    new_mission['ground_truth']['safety_label'] = "Risky" # Formation collision risk overrides smoothness
    
    # Make individual trajectories super smooth
    for drone in new_mission['drones']:
        traj = drone['trajectory']
        # Smooth out altitude and speed
        avg_alt = sum([p['altitude'] for p in traj]) / len(traj) if len(traj) > 0 else 100
        avg_speed = sum([p['speed'] for p in traj]) / len(traj) if len(traj) > 0 else 10
        for p in traj:
            p['altitude'] = avg_alt + random.gauss(0, 0.1) # Very stable
            p['speed'] = avg_speed + random.gauss(0, 0.1) # Very consistent
            
    # But collapse formation distance to dangerous levels at some point
    mid_point = len(new_mission['drones'][0]['trajectory']) // 2
    
    # Move drone 2 to drone 1's position
    if len(new_mission['drones']) >= 2:
        d1 = new_mission['drones'][0]['trajectory']
        d2 = new_mission['drones'][1]['trajectory']
        
        for i in range(mid_point - 5, mid_point + 5):
            if i < len(d1) and i < len(d2):
                # Drone 2 gets dangerously close to Drone 1
                d2[i]['latitude'] = d1[i]['latitude'] + 0.000001 # Almost collision
                d2[i]['longitude'] = d1[i]['longitude'] + 0.000001
                
    return new_mission

def create_ambiguous_maneuver(mission):
    """Scenario 4: Ambiguous Maneuver (Rapid heading changes but controlled)"""
    new_mission = copy.deepcopy(mission)
    new_mission['mission_id'] += "_HARD_AMBIGUOUS"
    new_mission['ground_truth']['safety_label'] = "Safe" # It's a controlled zigzag
    
    # Pick a drone to do a zigzag
    target_drone = new_mission['drones'][0]
    traj = target_drone['trajectory']
    
    for i in range(10, len(traj)-10):
        # Zigzag heading every few seconds
        if (i // 5) % 2 == 0:
            traj[i]['heading'] = (traj[i]['heading'] + 45) % 360
        else:
            traj[i]['heading'] = (traj[i]['heading'] - 45) % 360
            
        # But keep speed and altitude PERFECTLY stable to show control
        traj[i]['speed'] = 15.0
        traj[i]['altitude'] = 100.0
        
    return new_mission

def create_data_gaps(mission):
    """Scenario 5: Data Gaps (Uncertainty)"""
    new_mission = copy.deepcopy(mission)
    new_mission['mission_id'] += "_HARD_GAPS"
    new_mission['ground_truth']['safety_label'] = "Borderline" # Missing data makes it uncertain
    
    # Remove chunks of data
    for drone in new_mission['drones']:
        traj = drone['trajectory']
        # Remove 20% of points in the middle
        cut_start = int(len(traj) * 0.4)
        cut_end = int(len(traj) * 0.6)
        del traj[cut_start:cut_end]
        
    return new_mission

def main():
    print("Generating HARD dataset...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "complex_uav_missions.json")
    try:
        missions = load_missions(input_path)
    except FileNotFoundError:
        print("Source file not found. Creating dummy data for testing.")
        # Create a dummy mission structure if file not found (for demonstration)
        missions = [{
            "mission_id": "DUMMY_MISSION",
            "mission_type": "surveillance",
            "flight_duration": "100s",
            "drones": [
                {"drone_id": "d1", "trajectory": [{"time": i, "latitude": 0, "longitude": 0, "altitude": 100, "heading": 0, "speed": 10, "gps_status": "OK", "signal_status": "OK"} for i in range(50)]},
                {"drone_id": "d2", "trajectory": [{"time": i, "latitude": 0.0001, "longitude": 0.0001, "altitude": 100, "heading": 0, "speed": 10, "gps_status": "OK", "signal_status": "OK"} for i in range(50)]}
            ],
            "ground_truth": {"safety_label": "Safe", "efficiency_label": "High"}
        }]

    # Select a subset of base missions to transform
    base_missions = missions[:5] # Take first 5 missions as base
    
    hard_missions = []
    
    for mission in base_missions:
        # Generate variants
        hard_missions.append(degrade_formation(mission))
        hard_missions.append(inject_gps_drift(mission))
        hard_missions.append(create_conflicting_signals(mission))
        hard_missions.append(create_ambiguous_maneuver(mission))
        hard_missions.append(create_data_gaps(mission))
        
    # Add some original missions too for control
    hard_missions.extend(base_missions)
    
    output_path = os.path.join(script_dir, "hard_uav_missions.json")
    save_missions(hard_missions, output_path)
    print(f"Generated {len(hard_missions)} hard missions in {output_path}")

if __name__ == "__main__":
    main()

