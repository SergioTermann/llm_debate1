import json
import numpy as np
import random
from typing import List, Dict, Tuple

np.random.seed(42)
random.seed(42)

MISSION_TYPES = {
    "surveillance": {
        "count": 40,
        "safety_profile": {"safe": 24, "borderline": 10, "risky": 6},
        "complexity": "low"
    },
    "formation": {
        "count": 30,
        "safety_profile": {"safe": 16, "borderline": 8, "risky": 6},
        "complexity": "medium"
    },
    "search_rescue": {
        "count": 20,
        "safety_profile": {"safe": 8, "borderline": 6, "risky": 6},
        "complexity": "high"
    },
    "adversarial_intercept": {
        "count": 10,
        "safety_profile": {"safe": 2, "borderline": 2, "risky": 6},
        "complexity": "high"
    }
}

PROBLEMS = {
    "low": ["gradual_drift", "minor_delay"],
    "medium": ["formation_gap", "timing_offset", "altitude_variation"],
    "high": ["collision_course", "gps_glitch", "signal_loss", 
             "emergency_evasion", "formation_break", "coordination_loss"]
}

def generate_trajectory(num_points: int = 150, base_altitude: float = 100.0,
                       base_speed: float = 20.0, problems: List[str] = None) -> List[Dict]:
    """Generate UAV trajectory with optional problems"""
    trajectory = []
    lat, lon = 0.0, 0.0
    altitude = base_altitude
    speed = base_speed
    heading = 90.0
    
    problems = problems or []
    has_gps_glitch = "gps_glitch" in problems
    has_signal_loss = "signal_loss" in problems
    
    for t in range(num_points):
        if has_gps_glitch and 40 <= t <= 50:
            lat += np.random.uniform(-0.005, 0.005)
            lon += np.random.uniform(-0.005, 0.005)
        elif has_signal_loss and 30 <= t <= 45:
            signal = "LOST"
        else:
            signal = "OK"
        
        heading += np.random.uniform(-3, 3)
        heading = heading % 360
        
        if "altitude_variation" in problems and 20 <= t <= 60:
            altitude = base_altitude + np.random.uniform(-20, 20)
        
        speed = base_speed + np.random.uniform(-2, 2)
        
        lat += np.cos(np.radians(heading)) * 0.0001
        lon += np.sin(np.radians(heading)) * 0.0001
        
        trajectory.append({
            "time": t,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "altitude": round(altitude, 2),
            "heading": round(heading, 1),
            "speed": round(speed, 2),
            "signal_status": signal if signal == "OK" else signal,
            "gps_status": "DRIFT" if has_gps_glitch and 40 <= t <= 50 else "OK"
        })
        
        altitude = max(50, altitude + np.random.uniform(-0.5, 0.5))
    
    return trajectory

def generate_swarm_trajectories(num_drones: int, formation_type: str,
                               problems: List[str] = None) -> List[Dict]:
    """Generate coordinated swarm trajectories"""
    trajectories = []
    base_positions = []
    
    for i in range(num_drones):
        if formation_type == "line":
            base_lat = i * 0.0005
            base_lon = 0
        elif formation_type == "v_shape":
            offset = (i - num_drones/2) * 0.0003
            base_lat = offset * 0.5
            base_lon = abs(offset)
        elif formation_type == "circle":
            angle = 2 * np.pi * i / num_drones
            base_lat = 0.0003 * np.cos(angle)
            base_lon = 0.0003 * np.sin(angle)
        else:
            base_lat = np.random.uniform(-0.002, 0.002)
            base_lon = np.random.uniform(-0.002, 0.002)
        base_positions.append((base_lat, base_lon))
    
    for i, (base_lat, base_lon) in enumerate(base_positions):
        traj = []
        for t in range(150):
            lat = base_lat + (t % 50) * 0.00001 * np.cos(i)
            lon = base_lon + (t % 50) * 0.00001 * np.sin(i)
            alt = 100 + np.random.uniform(-5, 5) + i * 2
            heading = (90 + t) % 360
            speed = 20 + np.random.uniform(-1, 1)
            
            signal = "OK"
            if "signal_loss" in problems and 30 <= t <= 45 and i == 0:
                signal = "LOST"
            
            traj.append({
                "time": t,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "altitude": round(alt, 2),
                "heading": round(heading, 1),
                "speed": round(speed, 2),
                "signal_status": signal,
                "gps_status": "OK"
            })
        trajectories.append({"drone_id": f"UAV_{i+1}", "trajectory": traj})
    
    return trajectories

def assign_safety_label(problems: List[str], mission_type: str) -> Tuple[str, str]:
    """Determine ground truth safety and efficiency labels"""
    risk_count = sum(1 for p in problems if p in 
                    ["collision_course", "gps_glitch", "signal_loss", 
                     "emergency_evasion", "formation_break"])
    
    if risk_count >= 2:
        safety = "Risky"
    elif risk_count == 1:
        if "formation_break" in problems or "coordination_loss" in problems:
            safety = "Risky"
        else:
            safety = "Borderline"
    else:
        safety = "Safe"
    
    if "adversarial_intercept" in mission_type:
        efficiency = "Low"
    elif "search_rescue" in mission_type:
        efficiency = "Medium"
    else:
        efficiency = "High"
    
    return safety, efficiency

def generate_mission(mission_id: str, mission_type: str, idx: int) -> Dict:
    """Generate a single mission"""
    config = MISSION_TYPES[mission_type]
    
    safety_dist = config['safety_profile']
    total = sum(safety_dist.values())
    
    labels = ['Safe'] * safety_dist['safe'] + ['Borderline'] * safety_dist['borderline'] + ['Risky'] * safety_dist['risky']
    
    safety = labels[(idx - 1) % total]
    
    problems_count = random.choices(
        [0, 1, 2, 3],
        weights=[30, 35, 25, 10]
    )[0]
    
    problem_pool = PROBLEMS["low"] + PROBLEMS["medium"] + PROBLEMS["high"]
    if config["complexity"] == "low":
        problem_pool = PROBLEMS["low"]
    elif config["complexity"] == "medium":
        problem_pool = PROBLEMS["low"] + PROBLEMS["medium"]
    
    if safety == "Risky":
        problems_count = random.choices([2, 3], weights=[60, 40])[0]
    elif safety == "Borderline":
        problems_count = random.choices([1, 2], weights=[60, 40])[0]
    else:
        problems_count = random.choices([0, 1], weights=[70, 30])[0]
    
    problems = random.sample(problem_pool, min(problems_count, len(problem_pool)))
    
    num_drones = random.choice([4, 6, 8, 10])
    formation_type = random.choice(["line", "v_shape", "circle", "random"])
    
    drones = generate_swarm_trajectories(num_drones, formation_type, problems)
    
    if mission_type == "adversarial_intercept":
        efficiency = "Low"
    elif mission_type == "search_rescue":
        efficiency = "Medium"
    else:
        efficiency = "High"
    
    issues = []
    critical_issues = ["gps_glitch", "signal_loss", "collision_course"]
    high_issues = ["formation_break", "emergency_evasion"]
    
    if safety == "Risky":
        problems = random.sample(critical_issues + high_issues, random.randint(2, 3))
    elif safety == "Borderline":
        problems = random.sample(high_issues, random.randint(1, 2))
    else:
        problems = random.sample(PROBLEMS["low"], random.randint(0, 1))
    
    if "gps_glitch" in problems:
        issues.append({"type": "GPS_DRIFT", "time": "40-50s", "severity": "critical"})
    if "signal_loss" in problems:
        issues.append({"type": "SIGNAL_LOST", "time": "30-45s", "severity": "critical"})
    if "formation_break" in problems:
        issues.append({"type": "FORMATION_BREAK", "time": "60-80s", "severity": "high"})
    if "collision_course" in problems:
        issues.append({"type": "COLLISION_RISK", "time": "90-100s", "severity": "critical"})
    if "emergency_evasion" in problems:
        issues.append({"type": "EMERGENCY_EVASION", "time": "70-85s", "severity": "high"})
    
    return {
        "mission_id": mission_id,
        "mission_type": mission_type,
        "flight_duration": "150s",
        "description": f"{mission_type.replace('_', ' ').title()} mission with {num_drones} UAVs",
        "num_drones": num_drones,
        "drones": drones,
        "ground_truth": {
            "safety_label": safety,
            "efficiency_label": efficiency,
            "critical_issues": issues,
            "problem_pattern": problems
        }
    }

print("Generating 100 UAV missions (Paper Scale)...")
print("="*60)

missions = []
mission_counter = 1

for mtype, config in MISSION_TYPES.items():
    print(f"\nGenerating {config['count']} {mtype.replace('_', ' ').title()} missions...")
    for i in range(config['count']):
        mission_id = f"MISSION_{mission_counter:02d}_{mtype.upper()}"
        mission = generate_mission(mission_id, mtype, mission_counter)
        missions.append(mission)
        mission_counter += 1
        print(f"  [{mission_counter}/100] {mission_id}: {mission['ground_truth']['safety_label']}")

dataset = {
    "dataset_info": {
        "name": "UAV Swarm Evaluation Dataset (Paper Scale)",
        "total_missions": len(missions),
        "mission_types": list(MISSION_TYPES.keys()),
        "ground_truth_source": "Simulated with expert annotations",
        "reference": "Multi-Agent Debate Framework IEEE Paper"
    },
    "missions": missions
}

output_file = "complex_uav_missions.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\n" + "="*60)
print(f"Dataset saved to: {output_file}")
print(f"Total missions: {len(missions)}")

safety_counts = {"Safe": 0, "Borderline": 0, "Risky": 0}
efficiency_counts = {"High": 0, "Medium": 0, "Low": 0}
for m in missions:
    safety_counts[m['ground_truth']['safety_label']] += 1
    efficiency_counts[m['ground_truth']['efficiency_label']] += 1

print(f"\nGround Truth Distribution:")
print(f"  Safety: Safe={safety_counts['Safe']}, Borderline={safety_counts['Borderline']}, Risky={safety_counts['Risky']}")
print(f"  Efficiency: High={efficiency_counts['High']}, Medium={efficiency_counts['Medium']}, Low={efficiency_counts['Low']}")
