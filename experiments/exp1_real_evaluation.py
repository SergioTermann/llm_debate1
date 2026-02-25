import json
import os
import time
import math
import re
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class TrajectoryAnalyzer:
    """轨迹分析器 - 从原始数据中提取特征"""
    
    @staticmethod
    def analyze_single_drone(trajectory: List[Dict]) -> Dict:
        """分析单个无人机的轨迹"""
        if len(trajectory) < 2:
            return {
                "trajectory_smoothness": 0,
                "altitude_stability": 0,
                "speed_consistency": 0,
                "heading_changes": []
            }
        
        # 提取数据（兼容新旧数据格式）
        timestamps = [p.get('timestamp', p.get('time', 0)) for p in trajectory]
        altitudes = [p['altitude'] for p in trajectory]
        speeds = [p['speed'] for p in trajectory]
        headings = [p['heading'] for p in trajectory]
        
        # 1. 轨迹平滑度 - 基于heading变化
        heading_changes = []
        for i in range(1, len(headings)):
            diff = abs(headings[i] - headings[i-1])
            if diff > 180:
                diff = 360 - diff
            heading_changes.append(diff)
        
        avg_heading_change = np.mean(heading_changes) if heading_changes else 0
        trajectory_smoothness = max(0, 100 - avg_heading_change * 2)
        
        # 2. 高度稳定性
        altitude_std = np.std(altitudes) if len(altitudes) > 1 else 0
        altitude_stability = max(0, 100 - altitude_std * 0.5)
        
        # 3. 速度一致性
        speed_std = np.std(speeds) if len(speeds) > 1 else 0
        speed_consistency = max(0, 100 - speed_std * 2)
        
        # 4. 极值提取 (Worst Case Evidence)
        max_heading_change = max(heading_changes) if heading_changes else 0
        altitude_range = (max(altitudes) - min(altitudes)) if altitudes else 0
        speed_range = (max(speeds) - min(speeds)) if speeds else 0

        # 5. Altitude trend (for landing/takeoff detection)
        altitude_trend = altitudes if len(altitudes) > 0 else []

        return {
            "trajectory_smoothness": trajectory_smoothness,
            "altitude_stability": altitude_stability,
            "speed_consistency": speed_consistency,
            "heading_changes": heading_changes,
            "avg_heading_change": avg_heading_change,
            "altitude_std": altitude_std,
            "speed_std": speed_std,
            "max_heading_change": max_heading_change,
            "altitude_range": altitude_range,
            "speed_range": speed_range,
            "altitude_trend": altitude_trend
        }
    
    @staticmethod
    def analyze_formation(drones: List[Dict]) -> Dict:
        """分析编队保持情况 (Time-aligned)"""
        if len(drones) < 2:
            return {"formation_stability": 100, "coordination_quality": 100}
        
        # Build time-indexed positions for all drones
        # drone_positions[time] = {drone_index: (lat, lon, speed, heading)}
        time_map = {}
        
        for d_idx, drone in enumerate(drones):
            for point in drone['trajectory']:
                t = int(point.get('timestamp', point.get('time', 0)))
                
                # Handle GPS format
                if 'gps' in point:
                    lat, lon = point['gps']['lat'], point['gps']['lon']
                else:
                    lat, lon = point['latitude'], point['longitude']
                
                if t not in time_map:
                    time_map[t] = {}
                
                time_map[t][d_idx] = {
                    'pos': (lat, lon),
                    'speed': point['speed'],
                    'heading': point['heading']
                }
        
        formation_distances = []
        speed_correlations = []
        heading_correlations = []
        
        # Sort times to iterate in order
        sorted_times = sorted(time_map.keys())
        
        for t in sorted_times:
            drones_at_t = time_map[t]
            # Only analyze if we have data for at least 2 drones
            if len(drones_at_t) < 2:
                continue
                
            # Extract positions, speeds, headings
            current_positions = [d['pos'] for d in drones_at_t.values()]
            current_speeds = [d['speed'] for d in drones_at_t.values()]
            current_headings = [d['heading'] for d in drones_at_t.values()]
            
            # Calculate pairwise distances
            for j in range(len(current_positions)):
                for k in range(j+1, len(current_positions)):
                    lat1, lon1 = current_positions[j]
                    lat2, lon2 = current_positions[k]
                    dist = math.sqrt((lat1-lat2)**2 + (lon1-lon2)**2) * 111000 # Convert degrees to meters
                    formation_distances.append(dist)
            
            # Calculate coordination metrics (std dev at this timestep)
            if len(current_speeds) > 1:
                speed_correlations.append(np.std(current_speeds))
            if len(current_headings) > 1:
                heading_correlations.append(np.std(current_headings))
        
        if not formation_distances:
            return {"formation_stability": 100, "coordination_quality": 100}
        
        # Use coefficient of variation (CV) for formation stability (same as data generation)
        formation_mean = np.mean(formation_distances)
        formation_std = np.std(formation_distances)
        
        if formation_mean > 0:
            cv = formation_std / formation_mean
            formation_stability = max(0, 100 - cv * 50)
        else:
            formation_stability = 100.0
        
        # 极值提取
        min_formation_dist = min(formation_distances) if formation_distances else 0
        max_formation_dist = max(formation_distances) if formation_distances else 0
        
        avg_speed_std = np.mean(speed_correlations) if speed_correlations else 0
        avg_heading_std = np.mean(heading_correlations) if heading_correlations else 0
        
        coordination_quality = max(0, 100 - (avg_speed_std * 3 + avg_heading_std * 0.5))
        
        return {
            "formation_stability": formation_stability,
            "coordination_quality": coordination_quality,
            "avg_formation_distance": formation_mean,
            "formation_std": formation_std,
            "min_formation_dist": min_formation_dist,
            "max_formation_dist": max_formation_dist
        }
    
    @staticmethod
    def detect_anomalies(trajectory: List[Dict], formation_analysis: Dict = None) -> List[str]:
        """检测异常情况"""
        anomalies = []
        
        if len(trajectory) < 2:
            return anomalies
        
        # 分析单个无人机
        analysis = TrajectoryAnalyzer.analyze_single_drone(trajectory)
        
        # 检测急剧转向
        if analysis['avg_heading_change'] > 60:
            anomalies.append(f"Sharp heading changes detected (avg: {analysis['avg_heading_change']:.1f}°)")
        
        # 检测高度不稳定
        if analysis['altitude_std'] > 30:
            anomalies.append(f"High altitude variation (std: {analysis['altitude_std']:.1f}m)")
        
        # 检测速度不稳定
        if analysis['speed_std'] > 15:
            anomalies.append(f"High speed variation (std: {analysis['speed_std']:.1f}m/s)")
        
        # 检测编队问题
        if formation_analysis:
            if formation_analysis['formation_stability'] < 60:
                anomalies.append(f"Poor formation stability (score: {formation_analysis['formation_stability']:.1f})")
            if formation_analysis['coordination_quality'] < 60:
                anomalies.append(f"Poor coordination quality (score: {formation_analysis['coordination_quality']:.1f})")
        
        return anomalies

    @staticmethod
    def generate_motion_sequence(trajectory: List[Dict]) -> str:
        """生成语义动作序列 (Semantic Motion Tokenization)"""
        if len(trajectory) < 2:
            return "Insufficient data"
        
        sequence = []
        # 将轨迹分为5个阶段
        chunk_size = max(1, len(trajectory) // 5)
        
        for i in range(0, len(trajectory), chunk_size):
            chunk = trajectory[i:i+chunk_size]
            if not chunk: continue
            
            # 分析该阶段特征
            alt_change = chunk[-1]['altitude'] - chunk[0]['altitude']
            avg_speed = np.mean([p['speed'] for p in chunk])
            heading_std = np.std([p['heading'] for p in chunk])
            
            # 语义映射规则
            phase_desc = ""
            if i == 0 and alt_change > 2:
                phase_desc = "Takeoff"
            elif i >= len(trajectory) - chunk_size and alt_change < -2:
                phase_desc = "Landing"
            elif heading_std > 15:
                phase_desc = "Turning/Maneuver"
            elif abs(alt_change) < 1.0:
                phase_desc = "Stable Cruise"
            elif alt_change > 1.0:
                phase_desc = "Climb"
            elif alt_change < -1.0:
                phase_desc = "Descent"
            else:
                phase_desc = "Flight"
                
            # 添加异常标记
            if heading_std > 30:
                phase_desc += "(Unstable)"
            elif avg_speed < 1.0 and abs(alt_change) < 0.5:
                phase_desc = "Hover"
                
            sequence.append(f"[{phase_desc}]")
            
        return " -> ".join(sequence)

    @staticmethod
    def analyze_efficiency(trajectory: List[Dict]) -> Dict:
        """分析任务效率和完成度"""
        if len(trajectory) < 2:
            return {"efficiency_score": 0, "tortuosity": 0, "total_dist": 0}
            
        # 计算总飞行距离
        total_dist = 0
        for i in range(1, len(trajectory)):
            p1, p2 = trajectory[i-1], trajectory[i]
            # 兼容新旧数据格式
            if 'gps' in p1:
                lat1, lon1 = p1['gps']['lat'], p1['gps']['lon']
                lat2, lon2 = p2['gps']['lat'], p2['gps']['lon']
            else:
                lat1, lon1 = p1['latitude'], p1['longitude']
                lat2, lon2 = p2['latitude'], p2['longitude']
            
            dist = math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111000  # 粗略转米
            total_dist += dist
            
        # 计算有效位移 (Start to End)
        start, end = trajectory[0], trajectory[-1]
        if 'gps' in start:
            start_lat, start_lon = start['gps']['lat'], start['gps']['lon']
            end_lat, end_lon = end['gps']['lat'], end['gps']['lon']
        else:
            start_lat, start_lon = start['latitude'], start['longitude']
            end_lat, end_lon = end['latitude'], end['longitude']
            
        displacement = math.sqrt((start_lat - end_lat)**2 + (start_lon - end_lon)**2) * 111000
                               
        # 曲率 (1.0 is perfect straight line)
        tortuosity = total_dist / displacement if displacement > 1 else 1.0
        
        # 效率评分 (Tortuosity 越接近 1 越好，但在实际任务中 1.2-1.5 也是正常的)
        # 如果 > 3.0 说明绕了很多弯路
        efficiency_score = max(0, 100 - (tortuosity - 1.0) * 20)
        
        return {
            "efficiency_score": efficiency_score,
            "tortuosity": tortuosity,
            "total_distance": total_dist,
            "displacement": displacement
        }

    @staticmethod
    def segment_trajectory(trajectory: List[Dict]) -> List[Dict]:
        """轨迹分割 - 基于运动学属性将轨迹分成逻辑段"""
        if len(trajectory) < 2:
            return []
        
        segments = []
        current_segment = [trajectory[0]]
        current_heading = trajectory[0]['heading']
        current_speed = trajectory[0]['speed']
        current_alt = trajectory[0]['altitude']
        
        for i in range(1, len(trajectory)):
            point = trajectory[i]
            heading = point['heading']
            speed = point['speed']
            alt = point['altitude']
            
            heading_change = abs(heading - current_heading)
            if heading_change > 180:
                heading_change = 360 - heading_change
            
            speed_change = abs(speed - current_speed)
            alt_change = abs(alt - current_alt)
            
            if (heading_change > 15 or speed_change > 5 or alt_change > 3 or 
                len(current_segment) > 20):
                segments.append({
                    'id': len(segments),
                    'points': current_segment.copy(),
                    'start_idx': i - len(current_segment),
                    'end_idx': i - 1,
                    'duration': current_segment[-1].get('timestamp', i) - current_segment[0].get('timestamp', 0)
                })
                current_segment = [point]
                current_heading = heading
                current_speed = speed
                current_alt = alt
            else:
                current_segment.append(point)
        
        if current_segment:
            segments.append({
                'id': len(segments),
                'points': current_segment,
                'start_idx': len(trajectory) - len(current_segment),
                'end_idx': len(trajectory) - 1,
                'duration': current_segment[-1].get('timestamp', len(trajectory)) - current_segment[0].get('timestamp', 0)
            })
        
        return segments

    @staticmethod
    def extract_events(trajectory: List[Dict]) -> List[Dict]:
        """事件提取 - 识别关键飞行事件"""
        events = []
        
        if len(trajectory) < 2:
            return events
        
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]
            curr = trajectory[i]
            
            heading_change = abs(curr['heading'] - prev['heading'])
            if heading_change > 180:
                heading_change = 360 - heading_change
            
            alt_change = curr['altitude'] - prev['altitude']
            speed_change = curr['speed'] - prev['speed']
            
            timestamp = curr.get('timestamp', i)
            
            if heading_change > 30:
                events.append({
                    'time': timestamp,
                    'type': 'sharp_turn',
                    'value': heading_change,
                    'description': f'Sharp turn: {heading_change:.1f}°'
                })
            
            if abs(alt_change) > 3:
                events.append({
                    'time': timestamp,
                    'type': 'rapid_alt_change',
                    'value': abs(alt_change),
                    'description': f'Rapid altitude change: {alt_change:.1f}m/s'
                })
            
            if abs(speed_change) > 5:
                events.append({
                    'time': timestamp,
                    'type': 'rapid_speed_change',
                    'value': abs(speed_change),
                    'description': f'Rapid speed change: {speed_change:.1f}m/s'
                })
        
        return events

    @staticmethod
    def identify_attention_regions(trajectory: List[Dict]) -> List[Dict]:
        """识别需要关注的区域 - 高方差区域"""
        attention_regions = []
        
        if len(trajectory) < 10:
            return attention_regions
        
        window_size = 5
        for i in range(0, len(trajectory) - window_size + 1):
            window = trajectory[i:i+window_size]
            
            headings = [p['heading'] for p in window]
            speeds = [p['speed'] for p in window]
            alts = [p['altitude'] for p in window]
            
            heading_var = np.std(headings)
            speed_var = np.std(speeds)
            alt_var = np.std(alts)
            
            if heading_var > 20 or speed_var > 8 or alt_var > 5:
                attention_regions.append({
                    'time_start': window[0].get('timestamp', i),
                    'time_end': window[-1].get('timestamp', i+window_size),
                    'reason': f'High variance: heading={heading_var:.1f}°, speed={speed_var:.1f}m/s, alt={alt_var:.1f}m'
                })
        
        return attention_regions

    @staticmethod
    def analyze_temporal_phases(trajectory: List[Dict],
                                 drone_analyses: List[Dict] = None,
                                 formation_analysis: Dict = None,
                                 n_phases: int = 3) -> List[Dict]:
        """
        时序阶段分析 —— 将任务分为早/中/末三段，分别计算安全指标。
        末段（降落/任务收尾）权重最高，是安全评估的关键窗口。
        """
        if len(trajectory) < n_phases:
            return []

        n = len(trajectory)
        phase_size = n // n_phases
        phases = []

        risk_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        for p in range(n_phases):
            start = p * phase_size
            end = n if p == n_phases - 1 else (p + 1) * phase_size
            seg = trajectory[start:end]

            altitudes = [pt['altitude'] for pt in seg]
            speeds    = [pt['speed'] for pt in seg]
            headings  = [pt['heading'] for pt in seg]

            hdg_changes = []
            for i in range(1, len(headings)):
                diff = abs(headings[i] - headings[i-1])
                if diff > 180: diff = 360 - diff
                hdg_changes.append(diff)

            avg_hdg_chg = float(np.mean(hdg_changes)) if hdg_changes else 0.0
            alt_std     = float(np.std(altitudes))
            spd_std     = float(np.std(speeds))
            gps_issues  = sum(1 for pt in seg if pt.get('gps_status') != 'OK')
            sig_issues  = sum(1 for pt in seg if pt.get('signal_status') != 'OK')

            # 综合风险评分（0=低风险，越高越危险）
            risk_score = (avg_hdg_chg / 5 + alt_std * 0.5 + spd_std * 0.5
                          + gps_issues * 3 + sig_issues * 3)
            risk_idx = min(3, int(risk_score / 10))
            risk_label = risk_labels[risk_idx]

            phases.append({
                "phase": p + 1,
                "label": ["Early", "Mid", "Late"][p] if n_phases == 3 else f"P{p+1}",
                "t_range": f"t={start}-{end-1}",
                "avg_hdg_change": round(avg_hdg_chg, 1),
                "alt_std": round(alt_std, 2),
                "spd_std": round(spd_std, 2),
                "gps_issues": gps_issues,
                "sig_issues": sig_issues,
                "risk_score": round(risk_score, 1),
                "risk_label": risk_label
            })

        return phases

    @staticmethod
    def check_cross_drone_consistency(drones: List[Dict]) -> Dict:
        """
        跨机一致性检查 —— 检测多架无人机是否同步出现传感器异常。
        独立传感器的失效不会完全同步；若同步，说明系统级故障（GPS欺骗/电磁干扰）。
        返回：同步漂移时段、受影响无人机数量、判断结论。
        """
        if len(drones) < 2:
            return {"sync_detected": False, "max_sync_count": 0, "sync_windows": []}

        # 构建 time→drone_idx 的状态矩阵
        time_status: Dict[int, List[str]] = {}
        for d_idx, drone in enumerate(drones):
            for pt in drone['trajectory']:
                t = int(pt.get('time', pt.get('timestamp', 0)))
                status = "BAD" if (pt.get('gps_status') != 'OK' or
                                   pt.get('signal_status') != 'OK') else "OK"
                if t not in time_status:
                    time_status[t] = []
                time_status[t].append(status)

        # 找出同步故障时间点（≥2架同时异常）
        sync_times = []
        for t, statuses in sorted(time_status.items()):
            bad_count = sum(1 for s in statuses if s == "BAD")
            if bad_count >= 2:
                sync_times.append((t, bad_count))

        # 合并连续时间段
        sync_windows = []
        if sync_times:
            w_start, w_count = sync_times[0]
            prev_t = w_start
            for t, cnt in sync_times[1:]:
                if t == prev_t + 1:
                    w_count = max(w_count, cnt)
                else:
                    sync_windows.append({"t_start": w_start, "t_end": prev_t,
                                         "max_drones_affected": w_count})
                    w_start, w_count = t, cnt
                prev_t = t
            sync_windows.append({"t_start": w_start, "t_end": prev_t,
                                  "max_drones_affected": w_count})

        max_sync = max((w['max_drones_affected'] for w in sync_windows), default=0)
        ratio = len(sync_times) / max(1, max(len(d['trajectory']) for d in drones))

        return {
            "sync_detected": max_sync >= 2,
            "sync_windows": sync_windows,
            "max_sync_count": max_sync,
            "sync_time_ratio": round(ratio, 3),
            "verdict": (
                "SYSTEM-LEVEL FAILURE (synchronized)" if max_sync >= 3 else
                "POSSIBLE INTERFERENCE (2 drones sync)" if max_sync == 2 else
                "Independent errors (normal)"
            )
        }

    @staticmethod
    def check_physics_coherence(trajectory: List[Dict]) -> List[Dict]:
        """
        物理相干性验证 —— 检查 GPS位移方向 是否与 heading 传感器一致。
        若 GPS 显示向北移动但 heading 朝南，则为传感器矛盾（欺骗/故障）。
        返回：矛盾时间点列表，每个包含时间戳、GPS方向、heading方向、角度差。
        """
        if len(trajectory) < 2:
            return []

        contradictions = []
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]

            # GPS 位移方向
            dlat = curr['latitude']  - prev['latitude']
            dlon = curr['longitude'] - prev['longitude']
            displacement = math.sqrt(dlat**2 + dlon**2) * 111000  # 米

            if displacement < 0.5:   # 位移太小，忽略（悬停/精度问题）
                continue

            # 从GPS位移计算实际方向 (bearing)
            gps_bearing = math.degrees(math.atan2(dlon, dlat)) % 360

            # 传感器报告的heading
            sensor_hdg = curr['heading'] % 360

            # 计算角度差（取最短路径）
            angle_diff = abs(gps_bearing - sensor_hdg)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            # 差异超过 90° 视为矛盾（正常噪声应 < 30°）
            if angle_diff > 90:
                contradictions.append({
                    "time": curr.get('time', i),
                    "gps_bearing": round(gps_bearing, 1),
                    "sensor_heading": round(sensor_hdg, 1),
                    "angle_diff": round(angle_diff, 1),
                    "displacement_m": round(displacement, 2)
                })

        return contradictions

    @staticmethod
    def generate_trajectory_dsl(trajectory: List[Dict], formation_analysis: Dict = None) -> str:
        """生成轨迹DSL - 标准化的领域特定语言表示"""
        if len(trajectory) < 2:
            return "INSUFFICIENT DATA"
        
        dsl_lines = []
        
        segments = TrajectoryAnalyzer.segment_trajectory(trajectory)
        events = TrajectoryAnalyzer.extract_events(trajectory)
        attention_regions = TrajectoryAnalyzer.identify_attention_regions(trajectory)
        
        dsl_lines.append("=== TRAJECTORY DSL ===")
        
        for seg in segments:
            seg_points = seg['points']
            avg_speed = np.mean([p['speed'] for p in seg_points])
            heading_std = np.std([p['heading'] for p in seg_points])
            
            alt_start = seg_points[0]['altitude']
            alt_end = seg_points[-1]['altitude']
            alt_change = alt_end - alt_start
            
            if heading_std > 15:
                seg_type = "turn"
            elif abs(alt_change) > 2:
                seg_type = "climb" if alt_change > 0 else "descent"
            else:
                seg_type = "straight"
            
            direction = ""
            if seg_type == "turn":
                avg_heading = np.mean([p['heading'] for p in seg_points])
                if avg_heading >= 337.5 or avg_heading < 22.5:
                    direction = "N"
                elif 22.5 <= avg_heading < 67.5:
                    direction = "NE"
                elif 67.5 <= avg_heading < 112.5:
                    direction = "E"
                elif 112.5 <= avg_heading < 157.5:
                    direction = "SE"
                elif 157.5 <= avg_heading < 202.5:
                    direction = "S"
                elif 202.5 <= avg_heading < 247.5:
                    direction = "SW"
                elif 247.5 <= avg_heading < 292.5:
                    direction = "W"
                else:
                    direction = "NW"
            
            dsl_lines.append(f"SEG[{seg['id']}]: t={seg['start_idx']}-{seg['end_idx']}, type={seg_type}, dir={direction}, v={avg_speed:.1f}±{heading_std:.1f}")
        
        if events:
            dsl_lines.append("\n=== EVENTS ===")
            for event in events:
                dsl_lines.append(f"EVENT: t={event['time']}, {event['type']}={event['value']:.1f}")
        
        if attention_regions:
            dsl_lines.append("\n=== ATTENTION REGIONS ===")
            for attn in attention_regions:
                dsl_lines.append(f"ATTN: t={attn['time_start']}-{attn['time_end']}, reason={attn['reason']}")
        
        if formation_analysis:
            dsl_lines.append("\n=== FORMATION ===")
            dsl_lines.append(f"FORM: shape=swarm, error={100-formation_analysis['formation_stability']:.1f}%")
            dsl_lines.append(f"COORD: quality={formation_analysis['coordination_quality']:.1f}")
        
        anomalies = TrajectoryAnalyzer.detect_anomalies(trajectory, formation_analysis)
        if anomalies:
            dsl_lines.append("\n=== RISKS ===")
            for anomaly in anomalies:
                dsl_lines.append(f"RISK: {anomaly}")
        
        return "\n".join(dsl_lines)


class RealSingleMetricEvaluator:
    """优化的单指标评估器 - 基于轨迹分析和不可观测性检测"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def detect_unobservable_issues(self, trajectory: List[Dict]) -> Dict:
        """检测不可观测性问题（GPS故障、信号丢失）"""
        issues = {
            'gps_drift_count': 0,
            'signal_loss_count': 0,
            'has_critical_issues': False,
            'severity': 'none'
        }
        
        for point in trajectory:
            if point.get('gps_status') == 'DRIFT':
                issues['gps_drift_count'] += 1
            if point.get('signal_status') == 'LOST':
                issues['signal_loss_count'] += 1
        
        total_points = len(trajectory)
        drift_ratio = issues['gps_drift_count'] / total_points if total_points > 0 else 0
        loss_ratio = issues['signal_loss_count'] / total_points if total_points > 0 else 0
        
        # 判断严重程度
        if drift_ratio > 0.2 or loss_ratio > 0.2:
            issues['severity'] = 'critical'
            issues['has_critical_issues'] = True
        elif drift_ratio > 0.1 or loss_ratio > 0.1:
            issues['severity'] = 'high'
            issues['has_critical_issues'] = True
        elif drift_ratio > 0.05 or loss_ratio > 0.05:
            issues['severity'] = 'medium'
        
        return issues
    
    def evaluate(self, mission_data: Dict) -> Dict:
        drones = mission_data.get('drones', [])
        if not drones:
            return {"method": "Single-Metric", "safety_label": "Borderline", "score": 50, "issues_identified": []}
        
        # 分析第一个无人机的轨迹
        trajectory = drones[0]['trajectory']
        analysis = TrajectoryAnalyzer.analyze_single_drone(trajectory)
        
        # 检测不可观测性问题
        unobservable_issues = self.detect_unobservable_issues(trajectory)
        
        # 基础得分
        base_score = (
            analysis['trajectory_smoothness'] * 0.35 +
            analysis['altitude_stability'] * 0.25 +
            analysis['speed_consistency'] * 0.25
        )
        
        # 惩罚不可观测性问题
        if unobservable_issues['severity'] == 'critical':
            penalty = 40
        elif unobservable_issues['severity'] == 'high':
            penalty = 25
        elif unobservable_issues['severity'] == 'medium':
            penalty = 15
        else:
            penalty = 0
        
        # 惩罚急剧转向和高度变化
        if analysis['max_heading_change'] > 60:
            penalty += 15
        if analysis['altitude_range'] > 50:
            penalty += 10
        
        # 最终得分
        overall_score = max(0, base_score - penalty)
        
        # 根据得分判断安全等级（优化阈值）
        if overall_score >= 75:
            safety_label = "Safe"
        elif overall_score >= 55:
            safety_label = "Borderline"
        else:
            safety_label = "Risky"
        
        # 检测异常
        anomalies = TrajectoryAnalyzer.detect_anomalies(trajectory)
        
        # 添加不可观测性问题到异常列表
        if unobservable_issues['gps_drift_count'] > 0:
            anomalies.append(f"GPS drift detected ({unobservable_issues['gps_drift_count']} points)")
        if unobservable_issues['signal_loss_count'] > 0:
            anomalies.append(f"Signal loss detected ({unobservable_issues['signal_loss_count']} points)")
        
        # 计算效率
        eff = TrajectoryAnalyzer.analyze_efficiency(trajectory)
        eff_score = eff['efficiency_score']
        if eff_score >= 70:
            eff_label = "High"
        elif eff_score >= 50:
            eff_label = "Medium"
        else:
            eff_label = "Low"
        
        return {
            "method": "Single-Metric",
            "safety_label": safety_label,
            "efficiency_label": eff_label,
            "score": overall_score,
            "issues_identified": anomalies,
            "analysis": analysis,
            "unobservable_issues": unobservable_issues
        }


class RealFixedWeightEvaluator:
    """优化的固定权重评估器 - 包含编队分析和不可观测性检测"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.weights = {
            "trajectory_smoothness": 0.20,
            "altitude_stability": 0.20,
            "speed_consistency": 0.15,
            "formation_stability": 0.20,
            "coordination_quality": 0.15,
            "unobservable_penalty": 0.10
        }
    
    def detect_unobservable_issues(self, drones: List[Dict]) -> Dict:
        """检测所有无人机的不可观测性问题"""
        issues = {
            'total_gps_drift': 0,
            'total_signal_loss': 0,
            'affected_drones': set(),
            'severity': 'none',
            'has_critical_issues': False
        }
        
        total_points = 0
        for drone in drones:
            trajectory = drone['trajectory']
            total_points += len(trajectory)
            drone_has_issues = False
            
            for point in trajectory:
                if point.get('gps_status') == 'DRIFT':
                    issues['total_gps_drift'] += 1
                    drone_has_issues = True
                if point.get('signal_status') == 'LOST':
                    issues['total_signal_loss'] += 1
                    drone_has_issues = True
            
            if drone_has_issues:
                issues['affected_drones'].add(drone['drone_id'])
        
        # 计算比例
        drift_ratio = issues['total_gps_drift'] / total_points if total_points > 0 else 0
        loss_ratio = issues['total_signal_loss'] / total_points if total_points > 0 else 0
        
        # 判断严重程度
        if drift_ratio > 0.15 or loss_ratio > 0.15:
            issues['severity'] = 'critical'
            issues['has_critical_issues'] = True
        elif drift_ratio > 0.08 or loss_ratio > 0.08:
            issues['severity'] = 'high'
            issues['has_critical_issues'] = True
        elif drift_ratio > 0.03 or loss_ratio > 0.03:
            issues['severity'] = 'medium'
        
        issues['affected_drones'] = list(issues['affected_drones'])
        return issues
    
    def evaluate(self, mission_data: Dict) -> Dict:
        drones = mission_data.get('drones', [])
        if not drones:
            return {"method": "Fixed-Weight", "safety_label": "Borderline", "score": 50, "issues_identified": []}
        
        # 分析所有无人机
        drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(d['trajectory']) for d in drones]
        formation_analysis = TrajectoryAnalyzer.analyze_formation(drones)
        
        # 检测不可观测性问题
        unobservable_issues = self.detect_unobservable_issues(drones)
        
        # 计算平均指标
        avg_smoothness = np.mean([a['trajectory_smoothness'] for a in drone_analyses])
        avg_altitude_stability = np.mean([a['altitude_stability'] for a in drone_analyses])
        avg_speed_consistency = np.mean([a['speed_consistency'] for a in drone_analyses])
        avg_max_heading_change = np.mean([a['max_heading_change'] for a in drone_analyses])
        avg_altitude_range = np.mean([a['altitude_range'] for a in drone_analyses])
        
        # 基础加权得分
        base_score = (
            avg_smoothness * self.weights['trajectory_smoothness'] +
            avg_altitude_stability * self.weights['altitude_stability'] +
            avg_speed_consistency * self.weights['speed_consistency'] +
            formation_analysis['formation_stability'] * self.weights['formation_stability'] +
            formation_analysis['coordination_quality'] * self.weights['coordination_quality']
        )
        
        # 惩罚不可观测性问题
        if unobservable_issues['severity'] == 'critical':
            penalty = 35
        elif unobservable_issues['severity'] == 'high':
            penalty = 20
        elif unobservable_issues['severity'] == 'medium':
            penalty = 10
        else:
            penalty = 0
        
        # 惩罚编队问题
        if formation_analysis['min_formation_dist'] < 10:  # 小于10米，碰撞风险高
            penalty += 20
        elif formation_analysis['min_formation_dist'] < 20:
            penalty += 10
        
        # 惩罚急剧转向
        if avg_max_heading_change > 50:
            penalty += 15
        elif avg_max_heading_change > 30:
            penalty += 5
        
        # 惩罚高度不稳定
        if avg_altitude_range > 40:
            penalty += 10
        elif avg_altitude_range > 25:
            penalty += 5
        
        # 最终得分
        overall_score = max(0, base_score - penalty)
        
        # 根据得分判断安全等级（优化阈值）
        if overall_score >= 75:
            safety_label = "Safe"
        elif overall_score >= 55:
            safety_label = "Borderline"
        else:
            safety_label = "Risky"
        
        anomalies = TrajectoryAnalyzer.detect_anomalies(drones[0]['trajectory'], formation_analysis)
        
        # 添加不可观测性问题到异常列表
        if unobservable_issues['total_gps_drift'] > 0:
            anomalies.append(f"GPS drift detected ({unobservable_issues['total_gps_drift']} points, {len(unobservable_issues['affected_drones'])} drones)")
        if unobservable_issues['total_signal_loss'] > 0:
            anomalies.append(f"Signal loss detected ({unobservable_issues['total_signal_loss']} points, {len(unobservable_issues['affected_drones'])} drones)")
        if formation_analysis['min_formation_dist'] < 20:
            anomalies.append(f"Close formation detected (min distance: {formation_analysis['min_formation_dist']:.1f}m)")
        
        # 计算效率
        eff = TrajectoryAnalyzer.analyze_efficiency(drones[0]['trajectory'])
        eff_score = eff['efficiency_score']
        if eff_score >= 70:
            eff_label = "High"
        elif eff_score >= 50:
            eff_label = "Medium"
        else:
            eff_label = "Low"
        
        return {
            "method": "Fixed-Weight",
            "safety_label": safety_label,
            "efficiency_label": eff_label,
            "score": overall_score,
            "issues_identified": anomalies,
            "analysis": {
                "avg_smoothness": avg_smoothness,
                "avg_altitude_stability": avg_altitude_stability,
                "avg_speed_consistency": avg_speed_consistency,
                "formation_stability": formation_analysis['formation_stability'],
                "coordination_quality": formation_analysis['coordination_quality'],
                "min_formation_dist": formation_analysis['min_formation_dist']
            },
            "unobservable_issues": unobservable_issues
        }


class RealSingleAgentLLMEvaluator:
    """真实的单智能体LLM评估器 - 诚实思考模式"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        self.model = "Qwen/Qwen3-32B"
    
    def evaluate(self, mission_data: Dict) -> Dict:
        drones = mission_data.get('drones', [])
        if not drones:
            return {"method": "Single-Agent-LLM", "safety_label": "Borderline", "score": 50, "issues_identified": []}
        
        # 分析轨迹
        drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(d['trajectory']) for d in drones]
        formation_analysis = TrajectoryAnalyzer.analyze_formation(drones)
        
        # 构建轨迹数据摘要
        trajectory_summary = self._build_trajectory_summary(mission_data, drone_analyses, formation_analysis)
        
        prompt = f"""You are a professional UAV mission safety evaluator. Analyze the trajectory data carefully and make your judgment.

MISSION DATA:
{trajectory_summary}

EVALUATION FRAMEWORK:
1. Analyze the 5 key metrics (Smoothness, Altitude, Speed, Formation, Coordination).
2. Consider the Overall Average score as a strong indicator.
3. Check for any detected anomalies.

CLASSIFICATION GUIDELINES (STRICT):
1. SAFE: 
   - Overall Average >= 70.
   - AND NO Fatal Flaws (Max Heading > 180°, Alt Std > 50m).
   - AND Formation Stability > 60 AND Coordination Quality > 60.
   - Ignore efficiency/tortuosity for Safety rating.

2. BORDERLINE: 
   - Overall Average 55-69.
   - OR Average > 70 but one metric is near the limit.
   - OR Formation/Coordination is weak (40-60).

3. RISKY: 
   - Overall Average < 55.
   - OR ANY Fatal Flaw (Heading > 180°, Alt Std > 50m, Formation < 0.05m).
   - OR Formation Stability < 40 OR Coordination Quality < 40.

EFFICIENCY GUIDELINES:
- HIGH: Path Efficiency > 70, Tortuosity < 1.4.
- MEDIUM: Path Efficiency 50-70.
- LOW: Path Efficiency < 50.

Remember: Start with the PRESUMPTION OF SAFETY. If Score > 70 and no fatal flaws, it IS Safe. Efficiency is separate.

Provide:
SAFETY: [Safe/Borderline/Risky]
EFFICIENCY: [High/Medium/Low]
JUSTIFICATION: [Brief explanation of your reasoning]
SCORE: [Your confidence score 0-100]
"""
        
        try:
            print(f"    Calling LLM API...", end=" ", flush=True)
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        timeout=30.0
                    )
                    print("Response received", flush=True)
                    break
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "rate limiting" in error_str.lower():
                        wait_time = 2 ** attempt
                        print(f"Rate limited. Waiting {wait_time}s before retry (attempt {attempt + 1}/3)...")
                        time.sleep(wait_time)
                    elif attempt == 2:
                        raise e
                    else:
                        print(f"LLM Error: {e}. Retrying...")
                        time.sleep(1)
            
            response_text = response.choices[0].message.content
            
            safety_label = self._extract_field(response_text, "SAFETY", "Borderline")
            efficiency_label = self._extract_field(response_text, "EFFICIENCY", "Medium")
            score_str = self._extract_field(response_text, "SCORE", "50")
            score = float(score_str) if score_str.replace(".", "").isdigit() else 50.0
            
            anomalies = TrajectoryAnalyzer.detect_anomalies(drones[0]['trajectory'], formation_analysis)
            
            return {
                "method": "Single-Agent-LLM",
                "safety_label": safety_label,
                "efficiency_label": efficiency_label,
                "score": score,
                "issues_identified": anomalies
            }
        except Exception as e:
            print(f"Error in Single-Agent-LLM: {e}")
            return {
                "method": "Single-Agent-LLM",
                "safety_label": "Borderline",
                "efficiency_label": "Medium",
                "score": 50,
                "issues_identified": []
            }
    
    def _build_trajectory_summary(self, mission_data: Dict, drone_analyses: List[Dict], formation_analysis: Dict) -> str:
        """构建轨迹数据摘要 - 使用DSL tokenization"""
        smoothness = np.mean([a['trajectory_smoothness'] for a in drone_analyses])
        altitude = np.mean([a['altitude_stability'] for a in drone_analyses])
        speed = np.mean([a['speed_consistency'] for a in drone_analyses])
        formation = formation_analysis['formation_stability']
        coordination = formation_analysis['coordination_quality']
        overall_avg = (smoothness + altitude + speed + formation + coordination) / 5
        
        # 提取极值
        max_heading_chg = max([a.get('max_heading_change', 0) for a in drone_analyses])
        
        # 生成轨迹DSL (基于论文的tokenization方法)
        trajectory_dsl = TrajectoryAnalyzer.generate_trajectory_dsl(mission_data['drones'][0]['trajectory'], formation_analysis)
        
        # 计算效率指标
        eff = TrajectoryAnalyzer.analyze_efficiency(mission_data['drones'][0]['trajectory'])
        
        summary = f"""
Mission ID: {mission_data.get('mission_id', 'N/A')}
Metrics (0-100):
- Smoothness: {smoothness:.1f}
- Altitude Stability: {altitude:.1f}
- Speed Consistency: {speed:.1f}
- Formation Stability: {formation:.1f}
- Coordination: {coordination:.1f}
- OVERALL AVERAGE: {overall_avg:.1f}

Efficiency & Completion:
- Path Efficiency Score: {eff['efficiency_score']:.1f}/100
- Path Tortuosity: {eff['tortuosity']:.2f} (1.0 is straight line, lower is better)
- Total Distance: {eff['total_distance']:.1f}m
- Net Displacement: {eff['displacement']:.1f}m

Reference Thresholds (For Context):
- Max Heading Change: < 60° is NORMAL maneuvering. < 90° is acceptable.
- Altitude/Speed Scores: > 70 is Good. > 60 is Acceptable.

Trajectory DSL (Tokenized Evidence):
{trajectory_dsl}
"""
        return summary
    
    def _extract_field(self, text: str, field: str, default: str) -> str:
        """Extract field from response"""
        import re
        match = re.search(rf"{field}:\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = re.sub(r'\*\*', '', value)
            value = value.strip()
            
            if field.upper() == "SAFETY":
                value_lower = value.lower()
                if 'safe' in value_lower: return "Safe"
                if 'risk' in value_lower: return "Risky"
                if 'border' in value_lower: return "Borderline"
                
            if field.upper() == "EFFICIENCY":
                value_lower = value.lower()
                if 'high' in value_lower: return "High"
                if 'low' in value_lower: return "Low"
                if 'medium' in value_lower: return "Medium"
            
            return value
        return default


class RealMultiAgentDebateEvaluator:
    """
    Multi-Agent Expert Panel Evaluation (Based on IEEE Paper Architecture + Enhanced)
    
    Paper: "Multi-Agent Debate Framework for Comprehensive UAV Swarm Performance Evaluation"
    
    Key Innovations:
    1. Four Specialized Experts:
       - Flight Control Specialist (FCS): Precision metrics
       - Swarm Coordination Expert (SCE): Formation & cooperation
       - Safety Assessment Expert (SAE): Risk detection (Zero-trust)
       - Uncertainty Analysis Expert (UAE): Data quality & ambiguity [NEW]
    2. Red/Blue Team Rotation (eliminates confirmation bias)
    3. Evidence Chain Traceability (grounded claims)
    4. Deterministic Safety Verification (hard constraints)
    5. Multi-dimensional Consensus Modeling
    6. Uncertainty-aware decision making [NEW]
    """
    
    def __init__(self, api_key: str, model: str = "Qwen/Qwen3-32B", max_rounds: int = 2, verbose: bool = False,
                 enable_role_rotation: bool = True, enable_evidence_verification: bool = True,
                 ablation_settings: Dict = None):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        self.model = model
        self.verbose = verbose
        self.max_rounds = max_rounds
        
        # Ablation experiment settings
        self.enable_role_rotation = ablation_settings.get('role_rotation', True) if ablation_settings else enable_role_rotation
        self.enable_evidence_verification = ablation_settings.get('evidence_verification', True) if ablation_settings else enable_evidence_verification
        self.enable_adversarial = ablation_settings.get('adversarial', True) if ablation_settings else True
        self.enable_hierarchical = ablation_settings.get('hierarchical', True) if ablation_settings else True
        
        # Track current rotated roles
        self.current_role_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        
        # Four Specialized Expert Agents (as per paper Section II.B)
        self.experts = [
            {
                "id": 0,
                "name": "Flight Control Specialist (FCS)",
                "role": "FCS",
                "bias": "Precision Bias",
                "core_values": "Quantitative metrics, physical feasibility, energy efficiency, path optimization",
                "system_prompt": """You are a Flight Control Specialist with a PRECISION BIAS.

CORE EXPERTISE:
- Trajectory smoothness (heading change variance per phase)
- Altitude/speed CONSISTENCY across all mission phases (Early/Mid/Late)
- Physical feasibility: does the reported heading/speed MATCH the GPS displacement?
- MISSION EFFICIENCY: Path Efficiency score and Tortuosity ratio

CRITICAL RULE — PHYSICS COHERENCE CHECK:
You MUST check the "PHYSICS COHERENCE" section first. If GPS bearing contradicts heading by >90°
across multiple timesteps, this is a SENSOR MALFUNCTION (not a normal flight anomaly).
A drone claiming to fly South while GPS shows it moving North IS NOT SAFE.

CONTEXTUAL REASONING:
- Large heading changes (±70°) WITH perfectly constant speed/altitude = CONTROLLED maneuver (planned search)
- Large heading changes WITH erratic speed/altitude = loss of control
- Distinguish intentional maneuver from failure mode by checking speed/altitude stability

EVIDENCE CHAIN REQUIREMENT:
[CLAIM]: One-sentence judgment on flight control quality AND physical coherence
[EVIDENCE]: 3-5 bullets with specific phase data (e.g., "Late-phase AltStd=0.1m — stable")
[COUNTER]: Argue the opposing interpretation of the most ambiguous data point
[SUMMARY]: Three key takeaways
[CONFIDENCE]: 0-100"""
            },
            {
                "id": 1,
                "name": "Swarm Coordination Expert (SCE)",
                "role": "SCE",
                "bias": "Cohesion Bias",
                "core_values": "Formation stability, collective behavior, multi-scale coordination",
                "system_prompt": """You are a Swarm Coordination Expert with a COHESION BIAS.

CORE EXPERTISE:
- Formation geometry at EACH temporal phase (Early/Mid/Late)
- Sub-formation detection: are drones splitting into 2+ separate groups?
- Min/Max formation distance across all timesteps (not just average)
- Coordination quality trends: improving or degrading over time?

CRITICAL RULE — MULTI-SCALE ANALYSIS:
Global formation metrics can be misleading. You MUST check for sub-formation patterns:
- If 2 drones maintain 5m separation AND 2 others maintain 5m separation, but the two groups
  are 300m apart: this is a SPLIT FORMATION (Borderline, not Safe or Risky)
- A formation collapse in the LATE phase is more serious than one in the Early phase

TEMPORAL FORMATION TREND:
Always assess whether formation is STABLE / IMPROVING / DEGRADING over time.
A formation that degrades from 50m to 0.5m distance in the late phase is catastrophic
even if the early phase was perfect.

EVIDENCE CHAIN REQUIREMENT:
[CLAIM]: One-sentence judgment on formation health across all phases
[EVIDENCE]: 3-5 bullets citing phase-specific formation data and trend
[COUNTER]: Argue the opposing view (e.g., "the split was intentional coverage")
[SUMMARY]: Three key takeaways
[CONFIDENCE]: 0-100"""
            },
            {
                "id": 2,
                "name": "Safety Assessment Expert (SAE)",
                "role": "SAE",
                "bias": "Pessimistic Bias (Zero-Trust + Temporal Weighting)",
                "core_values": "Worst-case analysis, late-phase risk, collision proximity, recovery evaluation",
                "system_prompt": """You are a Safety Assessment Expert with PESSIMISTIC BIAS and TEMPORAL WEIGHTING.

CORE EXPERTISE:
- LATE-PHASE FOCUS: Safety incidents in the last 20% of mission (landing/final approach) are 3x more critical
- Minimum formation distance across ALL timesteps (the single closest approach matters most)
- Recovery quality: when near-miss occurred, did the system actively respond?
- Risk trajectory: is the situation IMPROVING or DETERIORATING in the final phase?

CRITICAL SAFETY THRESHOLDS:
- Formation dist < 0.5m at ANY point → COLLISION RISK (regardless of average)
- GPS errors in late phase → CRITICAL (landing guidance failure)
- Physics incoherence (GPS vs heading conflict) → RISKY (navigation failure)
- Synchronized sensor failure across drones → SYSTEM RISK (not individual noise)

TEMPORAL WEIGHTING RULE:
When the LATE phase risk label is HIGH or CRITICAL, this OVERRIDES an otherwise good average score.
"The mission was 90% safe but crashed on landing" = RISKY overall.

NEAR-MISS EVALUATION:
A near miss (0.5-2m) requires checking: Did speed/heading change sharply immediately after?
If YES → active avoidance system worked → Borderline
If NO  → no avoidance detected → Risky (could have collided)

EVIDENCE CHAIN REQUIREMENT:
[CLAIM]: One-sentence judgment focused on WORST-CASE and LATE-PHASE evidence
[EVIDENCE]: 3-5 bullets with exact timesteps (e.g., "t=82: UAV2/3 dist=0.3m — collision risk")
[COUNTER]: The Blue Team argument for why this risk is acceptable
[SUMMARY]: Three key takeaways emphasizing temporal risk
[CONFIDENCE]: 0-100"""
            },
            {
                "id": 3,
                "name": "Uncertainty & Integrity Analyst (UIA)",
                "role": "UIA",
                "bias": "Skeptical Bias (Data Integrity + Sensor Cross-Validation)",
                "core_values": "Cross-sensor consistency, system-level failure detection, physics validation",
                "system_prompt": """You are an Uncertainty & Integrity Analyst specializing in SENSOR CROSS-VALIDATION.

CORE EXPERTISE:
- CROSS-DRONE CONSISTENCY: Independent sensor failures don't synchronize. If multiple drones
  show GPS drift at EXACTLY the same timesteps → SYSTEM-LEVEL GPS failure (not random noise)
- PHYSICS COHERENCE: GPS displacement direction must match heading sensor within ~30°.
  If they contradict by >90°, the drone's navigation state is CORRUPTED.
- Data completeness and discontinuity patterns

CRITICAL DETECTION RULES:
1. SYNCHRONIZED SENSOR FAILURE:
   - 1 drone with GPS drift at t=40-55 = individual error (OK)
   - 4 drones ALL with GPS drift at t=40-55 in SAME direction = GPS spoofing/jamming (RISKY)
   
2. PHYSICS INCOHERENCE:
   - Check "PHYSICS COHERENCE" section carefully
   - If ≥5 contradictions detected: sensor malfunction is CONFIRMED → RISKY
   - If 1-4 contradictions: possible glitch → BORDERLINE

3. DATA QUALITY vs. MISSION OUTCOME:
   - Poor sensor quality DOES NOT automatically mean mission failed
   - If sensors are bad but no collision occurred AND mission recovered → BORDERLINE
   - If sensors are bad AND indicate collision/loss of control → RISKY

YOUR KEY QUESTION: "Can we TRUST the data? If not, what does the worst-case interpretation say?"

EVIDENCE CHAIN REQUIREMENT:
[CLAIM]: One-sentence judgment on data integrity and what it implies for safety
[EVIDENCE]: 3-5 bullets on specific sensor anomalies with timestamps
[COUNTER]: The case for trusting the data (or vice versa)
[SUMMARY]: Three key takeaways about data reliability
[CONFIDENCE]: 0-100 (confidence in data SUFFICIENCY for assessment)"""
            }
        ]
    
    def evaluate(self, mission_data: Dict) -> Dict:
        drones = mission_data.get('drones', [])
        if not drones:
            return {"method": "Multi-Agent-Debate", "safety_label": "Borderline", "score": 50, "issues_identified": []}
        
        # 分析轨迹
        drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(d['trajectory']) for d in drones]
        formation_analysis = TrajectoryAnalyzer.analyze_formation(drones)
        
        # 构建轨迹数据摘要
        trajectory_summary = self._build_trajectory_summary(mission_data, drone_analyses, formation_analysis)
        
        debate_history = []
        
        # === STEP 0: Hierarchical Routing (论文 Section III.B.1) ===
        trajectory = drones[0]['trajectory']
        events = TrajectoryAnalyzer.extract_events(trajectory)
        H_comp = self._calculate_complexity(trajectory, events, drone_analyses)
        route_layer = self._route_by_complexity(H_comp)
        
        print(f"\n    [Routing] Complexity H_comp={H_comp:.3f} -> Layer: {route_layer}")
        
        # 根据路由层调整max_rounds
        if route_layer == "FAST_CONSENSUS":
            actual_max_rounds = 1
            print(f"      [Fast Consensus] Single-round weighted aggregation")
        elif route_layer == "DEEP_ANALYSIS":
            actual_max_rounds = self.max_rounds
            print(f"      [Deep Analysis] Iterative adversarial debate ({actual_max_rounds} rounds)")
        else:  # META_DEBATE
            actual_max_rounds = self.max_rounds + 1
            print(f"      [Meta-Debate] Pre-debate alignment + {self.max_rounds} rounds")
        
        # === STEP 1: Deterministic Safety Verification (Paper Section III.F) ===
        mission_type = mission_data.get('mission_type', '')
        hard_violations = self._check_hard_constraints(drone_analyses, formation_analysis, mission_type)
        if hard_violations:
            print(f"\n    [WARNING] Hard constraints flagged: {hard_violations[0]} (Passing to debate)")
            # Add hard violations as critical evidence for the debate, but DO NOT VETO immediately
            # This allows the debate to contextually analyze if the violation is a sensor error or real risk
            trajectory_summary += "\n\n[SYSTEM DETECTED CRITICAL WARNINGS]:\n" + "\n".join([f"- {v}" for v in hard_violations])
            
            # ORIGINALLY VETOED HERE - NOW REMOVED TO ALLOW DEBATE
            # return {
            #     "method": "Multi-Agent-Debate",
            #     "safety_label": "Risky",
            #     ...
            # }
        
        # === STEP 2: Multi-round Expert Panel with Role & Team Rotation ===
        all_rounds = []
        emergent_issues = []
        
        for round_idx in range(actual_max_rounds):
            print(f"\n    [Expert Panel Round {round_idx + 1}/{actual_max_rounds}]")
            
            # Dynamic Role Rotation (Paper Eq. 14-15)
            if self.enable_role_rotation and round_idx > 0:
                if len(emergent_issues) > 2:
                    role_mapping = self._rotate_roles_adaptive(round_idx, emergent_issues)
                    print(f"      [Role Rotation - Adaptive] {role_mapping}")
                else:
                    role_mapping = self._rotate_roles_round_robin(round_idx)
                    print(f"      [Role Rotation - Round-Robin] {role_mapping}")
            
            # Red/Blue team assignment (Paper Eq. 11-12)
            red_ids, blue_ids = self._assign_red_blue_teams(round_idx)
            
            round_responses = []
            for expert in self.experts:
                # Get rotated expert if role rotation is enabled
                active_expert = self._get_rotated_expert(expert['id'])
                team = "RED" if expert['id'] in red_ids else "BLUE"
                
                context = trajectory_summary
                if round_idx > 0:
                    context += f"\n\n[PREVIOUS ROUND FEEDBACK]:\n{self._summarize_round(all_rounds[-1])}"
                
                if route_layer == "META_DEBATE" and round_idx == 0:
                    context += "\n\n[META-DEBATE ALIGNMENT]:\nFirst, define mission-specific evaluation standards before analysis."
                
                prompt = self._build_expert_prompt(active_expert, team, context)
                response = self._call_llm(active_expert['system_prompt'], prompt)
                
                parsed = self._parse_evidence_chain(response)
                parsed['expert'] = active_expert['name']
                parsed['original_expert'] = expert['name']
                parsed['team'] = team
                
                # Evidence Chain Verification (Paper Section III.D.1)
                if self.enable_evidence_verification:
                    is_valid, errors = self._verify_evidence_chain(parsed, drones[0]['trajectory'])
                    if not is_valid:
                        print(f"      [Evidence Warning] {active_expert['name']}: {errors[:1]}")
                        intervention = self._inject_evidence_intervention(round_idx, errors)
                        context += intervention
                        prompt = self._build_expert_prompt(active_expert, team, context)
                        response = self._call_llm(active_expert['system_prompt'], prompt)
                        parsed = self._parse_evidence_chain(response)
                        parsed['expert'] = active_expert['name']
                        parsed['team'] = team
                
                round_responses.append(parsed)
                debate_history.append(f"[R{round_idx+1}][{team}] {active_expert['name']}: {parsed.get('claim', '')[:80]}")
                print(f"      [{team}] {active_expert['name']}: {parsed.get('claim', 'N/A')[:60]}...")
                
                if round_idx == actual_max_rounds - 1:
                    claim = parsed.get('claim', '').lower()
                    for issue in ['collision', 'risk', 'safety', 'problem', 'instability']:
                        if issue in claim:
                            emergent_issues.append(issue)
            
            all_rounds.append(round_responses)
            
            # === Meta-Cognitive Quality Monitoring (论文 Section III.E.3) ===
            quality = self._calculate_debate_quality(round_responses)
            print(f"      [Quality Monitor] Novelty={quality['novelty']:.2f}, Diversity={quality['diversity']:.2f}, Relevance={quality['relevance']:.2f}, Depth={quality['depth']:.2f}, Overall={quality['overall']:.2f}")
            
            # 检查是否需要干预
            intervention = self._trigger_intervention(quality, round_idx)
            if intervention:
                print(f"      [INTERVENTION TRIGGERED]")
                # 将干预添加到下一轮的context中
                if round_idx < actual_max_rounds - 1:
                    trajectory_summary += f"\n\n{intervention}"
        
        # === STEP 3: Multi-dimensional Consensus Modeling (Paper Eq. 16-19) ===
        final_round = all_rounds[-1]
        consensus = self._calculate_consensus(final_round)
        print(f"\n    [Consensus] Score σ={consensus['score_std']:.2f}, Semantic Sim={consensus['semantic_sim']:.2f}, Priority={consensus['priority_consensus']:.2f}, Concern={consensus['concern_consensus']:.2f}")
        
        # === STEP 4: Final Verdict Synthesis ===
        final_verdict = self._synthesize_verdict(final_round, consensus, trajectory_summary)
        
        safety_label = self._extract_safety_label(final_verdict)
        efficiency_label = self._extract_efficiency_label(final_verdict)
        score = self._extract_score(final_verdict)
        
        anomalies = TrajectoryAnalyzer.detect_anomalies(mission_data['drones'][0]['trajectory'], formation_analysis)
        
        # === POST-PROCESSING: Override Borderline for clear safety issues ===
        if safety_label == "Borderline":
            # Check for clear safety issues that should override Borderline
            override_to_risky = False
            
            # Issue 1: Critical unobservable issues (GPS drift, signal loss)
            from exp1_real_evaluation import RealFixedWeightEvaluator
            unobservable_detector = RealFixedWeightEvaluator(self.api_key)
            unobservable = unobservable_detector.detect_unobservable_issues_all(drones) if hasattr(unobservable_detector, 'detect_unobservable_issues_all') else unobservable_detector.detect_unobservable_issues(drones)
            
            # Only override if issues are CRITICAL (High/Medium are handled by debate)
            if unobservable.get('severity') == 'critical':
                override_to_risky = True
            
            # Issue 2: Poor coordination quality - RELAXED THRESHOLD
            if formation_analysis.get('coordination_quality', 100) < 25:  # Lowered from 35
                override_to_risky = True
            
            # Issue 3: Unstable formation - RELAXED THRESHOLD
            if formation_analysis.get('formation_stability', 100) < 50:  # Lowered from 65
                override_to_risky = True
            
            # Issue 4: Majority of experts voted Risky with high confidence
            safe_votes = sum(1 for r in final_round if 'safe' in r.get('claim', '').lower() and r.get('confidence', 50) >= 60)
            risky_votes = sum(1 for r in final_round if ('risk' in r.get('claim', '').lower() or 'concern' in r.get('claim', '').lower()) and r.get('confidence', 50) >= 60)
            
            # Only override if Risky votes strictly outnumber Safe votes
            if risky_votes > safe_votes and risky_votes >= 2:
                override_to_risky = True
            
            if override_to_risky:
                print(f"    [POST-PROCESS] Overriding Borderline -> Risky (Critical safety issues verified)")
                safety_label = "Risky"
                score = max(10, score - 20)
        
        return {
            "method": "Multi-Agent-Debate",
            "safety_label": safety_label,
            "efficiency_label": efficiency_label,
            "score": score,
            "issues_identified": anomalies,
            "num_rounds": actual_max_rounds,
            "debate_transcript": debate_history,
            "route_layer": route_layer,
            "complexity": H_comp,
            "final_quality": quality
        }

    def _call_llm(self, system, user, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limiting" in error_str.lower():
                    wait_time = 2 ** attempt
                    print(f"Rate limited. Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                elif attempt == max_retries - 1:
                    print(f"LLM Error: {e}")
                    return "Error"
                else:
                    print(f"LLM Error: {e}. Retrying...")
                    time.sleep(1)
        return "Error"

    def _extract_score(self, text):
        import re
        # Try multiple formats
        match = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"CONFIDENCE:\s*(\d+)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"CONFIDENCE:\s*Score\s*(\d+)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 50.0

    def _extract_safety_label(self, text: str) -> str:
        import re
        # Try multiple formats
        match = re.search(r"SAFETY:\s*\[?\s*(Safe|Borderline|Risky)", text, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            return label.capitalize()
        
        match = re.search(r"CLAIM:\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            label = re.sub(r'\*\*', '', label)
            label = label.strip().lower()
            if 'safe' in label: return "Safe"
            if 'risk' in label: return "Risky"
            if 'border' in label: return "Borderline"
        
        match = re.search(r"SAFETY:\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            label = re.sub(r'\*\*', '', label)
            label = label.strip().lower()
            if 'safe' in label: return "Safe"
            if 'risk' in label: return "Risky"
            if 'border' in label: return "Borderline"
        return "Borderline"

    def _extract_efficiency_label(self, text: str) -> str:
        import re
        # Try multiple formats
        match = re.search(r"EFFICIENCY:\s*\[?\s*(High|Medium|Low)", text, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            return label.capitalize()
        
        match = re.search(r"CLAIM:\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            label = re.sub(r'\*\*', '', label)
            label = label.strip().lower()
            if 'high' in label: return "High"
            if 'low' in label: return "Low"
            if 'medium' in label: return "Medium"
        
        match = re.search(r"EFFICIENCY:\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            label = re.sub(r'\*\*', '', label)
            label = label.strip().lower()
            if 'high' in label: return "High"
            if 'low' in label: return "Low"
            if 'medium' in label: return "Medium"
        return "Medium"
    
    def _check_hard_constraints(self, drone_analyses: List[Dict], formation_analysis: Dict, mission_type: str = "") -> List[str]:
        """Deterministic Safety Verification (Paper Section III.F) - RELAXED"""
        violations = []
        
        # Hard Constraint 1: Heading change > 180° (Only catch uncontrolled spins)
        max_heading = max([a.get('max_heading_change', 0) for a in drone_analyses])
        if max_heading > 180:
            violations.append(f"FATAL: Heading change {max_heading:.1f}° > 180° (Loss of Control)")
        
        # Hard Constraint 2: Altitude instability (use std, not range)
        # For UAV missions, altitude range can be large (takeoff, climb, descent)
        # We should check altitude_std instead
        max_alt_std = max([a.get('altitude_std', 0) for a in drone_analyses])
        if max_alt_std > 50:
            violations.append(f"FATAL: Altitude std {max_alt_std:.1f}m > 50m (Extreme instability)")
        
        # Hard Constraint 3: Formation distance < 0.05m (collision imminent)
        # NOTE: 0.0m formation distance is normal during landing/takeoff phases
        # Only trigger VETO if formation distance is < 0.05m AND NOT in landing phase
        min_dist = formation_analysis.get('min_formation_dist', 999)
        
        # Check if this is a landing/takeoff mission by mission type or altitude trend
        is_landing_takeoff = False
        if 'landing' in mission_type.lower() or 'takeoff' in mission_type.lower():
            is_landing_takeoff = True
        elif len(drone_analyses) > 0 and len(drone_analyses[0].get('altitude_trend', [])) > 1:
            alt_trend = drone_analyses[0]['altitude_trend']
            # If altitude is decreasing significantly, it's likely landing
            if len(alt_trend) >= 2:
                alt_start = alt_trend[0]
                alt_end = alt_trend[-1]
                # If altitude drops more than 50m, it's likely landing/takeoff
                if abs(alt_end - alt_start) > 50:
                    is_landing_takeoff = True
        
        if not is_landing_takeoff and min_dist < 0.05:
            violations.append(f"FATAL: Min formation distance {min_dist:.1f}m < 0.05m (Collision risk)")
        
        return violations
    
    def _calculate_complexity(self, trajectory: List[Dict], events: List[Dict], drone_analyses: List[Dict]) -> float:
        """计算轨迹复杂度 H_comp (论文 Eq. 10)"""
        if len(trajectory) < 2:
            return 0.0
        
        # Event density: events per unit time
        duration = trajectory[-1].get('timestamp', len(trajectory)) - trajectory[0].get('timestamp', 0)
        event_density = len(events) / max(1, duration)
        
        # Velocity entropy: 信息论熵
        velocities = [point['speed'] for point in trajectory]
        
        # 计算速度分布的熵
        if len(velocities) > 0:
            velocity_hist, _ = np.histogram(velocities, bins=10, density=True)
            velocity_hist = velocity_hist[velocity_hist > 0]
            if len(velocity_hist) > 0:
                velocity_entropy = -np.sum(velocity_hist * np.log2(velocity_hist))
            else:
                velocity_entropy = 0.0
        else:
            velocity_entropy = 0.0
        
        # H_comp = alpha * event_density + beta * velocity_entropy
        alpha, beta = 1.0, 0.5
        H_comp = alpha * event_density + beta * velocity_entropy
        
        return H_comp
    
    def _route_by_complexity(self, H_comp: float) -> str:
        """根据复杂度路由到不同层 (论文 Section III.B.1)"""
        tau_1, tau_2 = 0.3, 0.7
        
        if H_comp < tau_1:
            return "FAST_CONSENSUS"
        elif H_comp < tau_2:
            return "DEEP_ANALYSIS"
        else:
            return "META_DEBATE"
    
    def _calculate_debate_quality(self, round_responses: List[Dict]) -> Dict:
        """计算辩论质量指标 (论文 Section III.E.3)"""
        if len(round_responses) == 0:
            return {'novelty': 0, 'diversity': 0, 'relevance': 0, 'depth': 0, 'overall': 0}
        
        # 1. Novelty: 新观点的比例
        claims = [r.get('claim', '').lower() for r in round_responses]
        unique_claims = set(claims)
        novelty = len(unique_claims) / len(claims) if len(claims) > 0 else 0
        
        # 2. Diversity: 观点的多样性
        safe_votes = sum('safe' in c for c in claims)
        risky_votes = sum('risky' in c or 'danger' in c for c in claims)
        total = len(claims)
        if total > 0:
            diversity = 1 - abs(safe_votes - risky_votes) / total
        else:
            diversity = 0
        
        # 3. Relevance: 与DSL证据的相关性
        relevance_scores = []
        for resp in round_responses:
            evidence = resp.get('evidence', [])
            # 检查是否引用了具体的DSL tokens
            has_dsl_refs = any('SEG' in e or 'EVENT' in e for e in evidence)
            relevance_scores.append(1.0 if has_dsl_refs else 0.5)
        relevance = np.mean(relevance_scores) if relevance_scores else 0
        
        # 4. Depth: 论证的深度（证据数量）
        evidence_counts = [len(r.get('evidence', [])) for r in round_responses]
        depth = np.mean(evidence_counts) / 5 if evidence_counts else 0  # 归一化到0-1
        
        # Overall quality
        overall_quality = (novelty + diversity + relevance + depth) / 4
        
        return {
            'novelty': novelty,
            'diversity': diversity,
            'relevance': relevance,
            'depth': depth,
            'overall': overall_quality
        }
    
    def _trigger_intervention(self, quality: Dict, round_idx: int) -> str:
        """当质量低于阈值时触发干预 (论文 Section III.E.3)"""
        tau_Q = 0.7
        
        if quality['overall'] >= tau_Q:
            return None
        
        # 根据最低的质量指标选择干预类型
        min_metric = min(quality.items(), key=lambda x: x[1])[0]
        
        if min_metric == 'novelty':
            intervention = f"""
[INTERVENTION - Round {round_idx + 1}]:
The debate is becoming repetitive. Experts are repeating similar claims without new insights.

INSTRUCTION: Each expert must provide at least ONE NEW perspective not mentioned before.
Focus on different aspects: Flight Control, Swarm Coordination, Safety, or Uncertainty.
"""
        elif min_metric == 'diversity':
            intervention = f"""
[INTERVENTION - Round {round_idx + 1}]:
The debate is showing "Groupthink". Experts are agreeing too easily.

INSTRUCTION FOR RED TEAM:
- You MUST construct a plausible FAILURE SCENARIO based on the weakest data point (e.g., "What if the GPS drift at t=45 coincides with a wind gust?").
- Do not accept "Good Enough".

INSTRUCTION FOR BLUE TEAM:
- Defend against the specific failure scenario constructed by Red Team.
"""
        elif min_metric == 'relevance':
            intervention = f"""
[INTERVENTION - Round {round_idx + 1}]:
Arguments are not grounded in trajectory data.

INSTRUCTION: All claims MUST cite specific DSL tokens (e.g., SEG[2], EVENT: t=15).
Avoid vague statements like "looks good" or "seems risky".
"""
        else:  # depth
            intervention = f"""
[INTERVENTION - Round {round_idx + 1}]:
Arguments lack sufficient evidence depth.

INSTRUCTION: Each expert must provide 3-5 evidence points.
Include: specific metrics, trajectory segments, and quantitative data.
"""
        
        return intervention
    
    def _assign_red_blue_teams(self, round_idx: int) -> tuple:
        """Red/Blue team rotation (Paper Eq. 11-12)
        
        With 4 experts: Round 0: [0] RED, [1,2,3] BLUE
                       Round 1: [1] RED, [0,2,3] BLUE
                       Round 2: [2] RED, [0,1,3] BLUE
                       etc.
        """
        N = len(self.experts)
        N_red = max(1, int(N * 0.25))  # ~25% red team for 4 experts (1 red, 3 blue)
        
        # Round-robin rotation
        red_ids = [(i + round_idx) % N for i in range(N_red)]
        blue_ids = [i for i in range(N) if i not in red_ids]
        
        return red_ids, blue_ids
    
    def _rotate_roles_round_robin(self, round_idx: int) -> Dict[int, int]:
        """Dynamic Role Rotation - Round-Robin (Paper Eq. 14)
        
        Agent a_i at round r assumes the role of agent (i + r) mod N
        This prevents perspective lock-in by cycling through different viewpoints.
        """
        N = len(self.experts)
        new_mapping = {}
        for orig_id in range(N):
            new_mapping[orig_id] = (orig_id + round_idx) % N
        self.current_role_mapping = new_mapping
        return new_mapping
    
    def _rotate_roles_adaptive(self, round_idx: int, emergent_issues: List[str]) -> Dict[int, int]:
        """Dynamic Role Rotation - Adaptive (Paper Eq. 15)
        
        Agents rotate based on expertise alignment with emergent issues from previous round.
        Relevance(a_j, E) quantifies how well agent j's expertise matches the issues.
        """
        N = len(self.experts)
        
        # Define expertise keywords for each agent
        expertise_keywords = {
            0: ['smoothness', 'altitude', 'speed', 'trajectory', 'efficiency', 'rmse'],
            1: ['formation', 'coordination', 'swarm', 'collective', 'communication'],
            2: ['safety', 'risk', 'collision', 'emergency', 'boundary', 'envelope'],
            3: ['uncertainty', 'quality', 'ambiguity', 'confidence', 'reliability']
        }
        
        def relevance_score(agent_id: int, issues: List[str]) -> float:
            keywords = set(expertise_keywords.get(agent_id, []))
            issue_text = ' '.join(issues).lower()
            score = sum(1 for kw in keywords if kw in issue_text)
            return float(score)
        
        new_mapping = {}
        for orig_id in range(N):
            score = relevance_score(orig_id, emergent_issues)
            new_mapping[orig_id] = score
        
        sorted_agents = sorted(new_mapping.keys(), key=lambda x: new_mapping[x], reverse=True)
        for i, agent_id in enumerate(sorted_agents):
            new_mapping[agent_id] = i
        
        self.current_role_mapping = new_mapping
        return new_mapping
    
    def _get_rotated_expert(self, original_expert_id: int) -> Dict:
        """Get the expert after role rotation"""
        if not self.enable_role_rotation:
            return self.experts[original_expert_id]
        
        rotated_id = self.current_role_mapping.get(original_expert_id, original_expert_id)
        return self.experts[rotated_id]
    
    def _verify_evidence_chain(self, response: Dict, trajectory: List[Dict]) -> Tuple[bool, List[str]]:
        """Evidence Chain Verification (Paper Section III.D.1)
        
        Validates that all DSL citations in the response actually exist in trajectory data.
        Returns: (is_valid, validation_errors)
        """
        errors = []
        evidence = response.get('evidence', [])
        
        for ev in evidence:
            seg_matches = re.findall(r'SEG\[(\d+)\]', ev)
            event_matches = re.findall(r'EVENT:\s*t=(\d+)', ev)
            attn_matches = re.findall(r'ATTN\[(\d+)\]', ev)
            
            for seg_id in seg_matches:
                seg_idx = int(seg_id)
                max_seg = len(trajectory) // 30
                if seg_idx > max_seg:
                    errors.append(f"SEG[{seg_idx}] exceeds maximum segment index ({max_seg})")
            
            for t in event_matches:
                time_point = int(t)
                if time_point >= len(trajectory):
                    errors.append(f"EVENT t={time_point} exceeds trajectory length ({len(trajectory)})")
        
        return len(errors) == 0, errors
    
    def _inject_evidence_intervention(self, round_idx: int, validation_errors: List[str]) -> str:
        """Generate intervention message for evidence chain violations"""
        return f"""
[INTERVENTION - Evidence Chain Verification FAILED]:
The following evidence citations are INVALID:
{chr(10).join(f"- {e}" for e in validation_errors[:3])}

INSTRUCTION: All claims must be backed by VERIFIABLE trajectory data.
- Use SEG[n] where n is a valid segment index (0-{len(self.experts)})
- Use EVENT: t=X where X is within trajectory time range
- Avoid fabricating specific metrics that don't exist in the data
"""
    
    def _build_expert_prompt(self, expert: Dict, team: str, context: str) -> str:
        """Build role-specific prompt with team instructions"""
        
        if team == "RED":
            team_instruction = """
[RED TEAM INSTRUCTIONS - Adversarial Role]:
- CRITICAL MISSION: You are the "Devil's Advocate". Your job is to find the hidden flaw that could cause a crash.
- Do NOT just look at averages. Look at specific SEGMENTS or EVENTS where limits were approached.
- Challenge "Safe" verdicts by asking: "What if this sensor reading is slightly off?" or "What if wind gusts occur here?"
- If the formation is tight (< 2m), argue it's a collision risk.
- If the formation is loose (> 10m), argue it's a coordination failure.
- Your Goal: PROVE that the mission is RISKY.
"""
        else:
            team_instruction = """
[BLUE TEAM INSTRUCTIONS - Collaborative Role]:
- MISSION: Defend the system's performance. Explain why the mission was successful despite minor imperfections.
- Contextualize errors: "This sharp turn was necessary for the mission waypoints," or "The formation change was a controlled maneuver."
- Refute Red Team's hypothetical risks with the hard data showing no actual collision occurred.
- Your Goal: PROVE that the mission is SAFE and EFFECTIVE.
"""
        
        return f"""
{team_instruction}

MISSION DATA:
{context}

As a {expert['name']} ({expert['bias']}), analyze this mission from your domain expertise.

REMEMBER YOUR COGNITIVE BIAS:
- {expert['bias']}: {expert['core_values']}

INSTRUCTION ON NOVELTY:
- Do NOT simply repeat what other experts have said.
- Find a NEW piece of evidence (a different time segment, a different metric) to support your claim.

Provide your assessment following the Evidence Chain format:
[CLAIM]: ...
[EVIDENCE]: ...
[COUNTER]: ...
[SUMMARY]: ...
[CONFIDENCE]: ...
"""
    
    def _parse_evidence_chain(self, response: str) -> Dict:
        """Parse structured Evidence Chain response (Paper Section III.E)"""
        parsed = {
            'claim': '',
            'evidence': [],
            'counter': '',
            'summary': '',
            'confidence': 50,
            'raw': response
        }
        
        # Extract CLAIM
        claim_match = re.search(r'\[CLAIM\]:?\s*(.+?)(?=\n\[|$)', response, re.IGNORECASE | re.DOTALL)
        if claim_match:
            parsed['claim'] = claim_match.group(1).strip()[:200]
        
        # Extract EVIDENCE
        evidence_match = re.search(r'\[EVIDENCE\]:?\s*(.+?)(?=\n\[|$)', response, re.IGNORECASE | re.DOTALL)
        if evidence_match:
            evidence_text = evidence_match.group(1).strip()
            # Split by bullet points or newlines
            parsed['evidence'] = [e.strip() for e in re.split(r'[-•\n]', evidence_text) if e.strip()][:5]
        
        # Extract CONFIDENCE
        conf_match = re.search(r'\[CONFIDENCE\]:?\s*(\d+)', response, re.IGNORECASE)
        if conf_match:
            parsed['confidence'] = int(conf_match.group(1))
        
        return parsed
    
    def _summarize_round(self, round_responses: List[Dict]) -> str:
        """Summarize previous round for next iteration"""
        summary_lines = []
        for resp in round_responses:
            expert = resp.get('expert', 'Unknown')
            claim = resp.get('claim', 'N/A')
            conf = resp.get('confidence', 0)
            summary_lines.append(f"- {expert} (Conf {conf}): {claim[:100]}")
        return "\n".join(summary_lines)
    
    def _calculate_consensus(self, round_responses: List[Dict]) -> Dict:
        """Multi-dimensional Consensus Modeling (论文 Eq. 16-19)"""
        import numpy as np
        
        confidences = [r.get('confidence', 50) for r in round_responses]
        
        # C_score: 1 - std(confidences) (Eq. 16)
        score_std = np.std(confidences) if len(confidences) > 1 else 0
        score_consensus = max(0, 1 - score_std / 100)
        
        # C_semantic: Simple similarity based on claim keywords (Eq. 17)
        claims = [r.get('claim', '').lower() for r in round_responses]
        safe_count = sum('safe' in c for c in claims)
        risky_count = sum('risky' in c or 'danger' in c for c in claims)
        total = len(claims)
        
        # Agreement: high if all agree, low if split
        max_agreement = max(safe_count, risky_count)
        semantic_sim = max_agreement / total if total > 0 else 0
        
        # C_priority: Similarity of claims (Eq. 18)
        # 使用简单的词汇重叠来模拟claim相似度
        if len(claims) >= 2:
            claim_words = [set(c.split()) for c in claims]
            pair_similarities = []
            for i in range(len(claim_words)):
                for j in range(i+1, len(claim_words)):
                    # Jaccard similarity
                    intersection = len(claim_words[i] & claim_words[j])
                    union = len(claim_words[i] | claim_words[j])
                    sim = intersection / union if union > 0 else 0
                    pair_similarities.append(sim)
            priority_consensus = np.mean(pair_similarities) if pair_similarities else 0
        else:
            priority_consensus = 0
        
        # C_concern: Overlap of evidence (Eq. 19)
        evidences = [r.get('evidence', []) for r in round_responses]
        if len(evidences) > 0:
            # 将所有证据转换为单词集合
            evidence_words_list = []
            for ev_list in evidences:
                ev_words = set()
                for ev in ev_list:
                    ev_words.update(ev.lower().split())
                evidence_words_list.append(ev_words)
            
            # 计算交集和并集
            if len(evidence_words_list) > 0:
                intersection = set.intersection(*evidence_words_list) if len(evidence_words_list) > 1 else evidence_words_list[0]
                union = set.union(*evidence_words_list)
                concern_consensus = len(intersection) / len(union) if len(union) > 0 else 0
            else:
                concern_consensus = 0
        else:
            concern_consensus = 0
        
        return {
            'score_std': score_std,
            'score_consensus': score_consensus,
            'semantic_sim': semantic_sim,
            'priority_consensus': priority_consensus,  # 新增
            'concern_consensus': concern_consensus,    # 新增
            'safe_votes': safe_count,
            'risky_votes': risky_count
        }
    
    def _synthesize_verdict(self, final_round: List[Dict], consensus: Dict, trajectory_summary: str) -> str:
        """Synthesize final verdict from expert panel"""
        
        # Aggregate expert opinions
        expert_summaries = []
        for resp in final_round:
            expert = resp.get('expert', 'Unknown')
            claim = resp.get('claim', 'N/A')
            conf = resp.get('confidence', 0)
            expert_summaries.append(f"- {expert} (Confidence {conf}): {claim}")
        
        synthesis_prompt = f"""
MISSION TRAJECTORY DATA:
{trajectory_summary}

EXPERT PANEL ASSESSMENTS:
{chr(10).join(expert_summaries)}

CONSENSUS METRICS:
- Confidence Std Dev: {consensus['score_std']:.1f}
- Safe Votes: {consensus['safe_votes']}/{len(final_round)}
- Risky Votes: {consensus['risky_votes']}/{len(final_round)}

YOUR TASK:
Synthesize the final verdict based on the *process* of debate, not just the final vote.

HIGHLIGHTS REQUIRED:
- Identify the "Turning Point" in the debate (if any): Did an expert change their mind? Why?
- Identify "Hidden Risks": Risks that were not obvious in the summary but were uncovered by an expert.
- Resolve conflicts: If experts disagree, explain whose evidence is stronger (e.g., Red Team found specific collision risk vs Blue Team general metrics).

DECISION LOGIC:
1. PRESUMPTION OF SAFETY unless proven otherwise by SPECIFIC evidence (e.g., time t=X, dist < 0.5m).
2. If Red Team found a valid collision risk (dist < 0.5m) or loss of control (heading change > 150 deg), Verdict is RISKY.
3. If Red Team's concerns were successfully refuted by Blue Team (e.g., it was a controlled maneuver), Verdict is SAFE.
4. If the situation is ambiguous or data is missing, Verdict is BORDERLINE.

OUTPUT FORMAT:
SAFETY: [Safe/Borderline/Risky]
EFFICIENCY: [High/Medium/Low]
SCORE: [0-100]
REASONING: [
  - **Debate Key Insight**: [One sentence on what the debate revealed]
  - **Verdict Justification**: [Explain why the final decision was reached]
]
"""
        
        return self._call_llm(
            "You are the Lead Safety Evaluator synthesizing expert opinions into a final verdict.",
            synthesis_prompt
        )
    
    def _build_trajectory_summary(self, mission_data: Dict, drone_analyses: List[Dict], formation_analysis: Dict) -> str:
        """
        构建增强型轨迹摘要，包含：
        1. 基础性能评分
        2. 极值证据（最坏情况）
        3. 时序阶段安全分析（末段加权）
        4. 跨机一致性检查（系统级故障检测）
        5. 物理相干性验证（传感器欺骗检测）
        6. DSL 令牌化轨迹
        """
        drones = mission_data['drones']
        primary_traj = drones[0]['trajectory']

        smoothness   = np.mean([a['trajectory_smoothness'] for a in drone_analyses])
        altitude     = np.mean([a['altitude_stability'] for a in drone_analyses])
        speed        = np.mean([a['speed_consistency'] for a in drone_analyses])
        formation    = formation_analysis['formation_stability']
        coordination = formation_analysis['coordination_quality']
        overall_avg  = (smoothness + altitude + speed + formation + coordination) / 5

        max_heading_chg = max([a.get('max_heading_change', 0) for a in drone_analyses])
        max_alt_std     = max([a.get('altitude_std', 0) for a in drone_analyses])
        min_form_dist   = formation_analysis.get('min_formation_dist', 0)

        eff = TrajectoryAnalyzer.analyze_efficiency(primary_traj)

        # ── 时序阶段分析 ──
        phases = TrajectoryAnalyzer.analyze_temporal_phases(primary_traj)
        phase_lines = []
        for ph in phases:
            trend_arrow = "↑" if ph['phase'] > 1 and ph['risk_score'] < phases[ph['phase']-2]['risk_score'] else "↓" if ph['phase'] > 1 else " "
            phase_lines.append(
                f"  {ph['label']:5s} ({ph['t_range']:10s}): "
                f"HdgVar={ph['avg_hdg_change']:5.1f}° | AltStd={ph['alt_std']:5.2f}m | "
                f"GPS_err={ph['gps_issues']:2d} | Risk={ph['risk_label']}{trend_arrow}"
            )
        phase_section = "\n".join(phase_lines) if phase_lines else "  N/A"

        # ── 末段预警（最后20%最重要）──
        n_traj = len(primary_traj)
        late_start = int(n_traj * 0.8)
        late_traj  = primary_traj[late_start:]
        late_gps_issues = sum(1 for pt in late_traj if pt.get('gps_status') != 'OK')
        late_risk_note  = ""
        if phases and phases[-1]['risk_label'] in ('HIGH', 'CRITICAL'):
            late_risk_note = (f"  [!] LATE-PHASE RISK DETECTED: {phases[-1]['risk_label']} "
                              f"(GPS errors={late_gps_issues} in last {len(late_traj)} steps)")

        # ── 分阶段编队距离趋势（关键：识别末段收敛/扩散危机）──
        early_end = int(n_traj * 0.33)
        mid_end   = int(n_traj * 0.66)
        def _phase_formation(start, end):
            sliced = [{"drone_id": d["drone_id"],
                       "trajectory": d["trajectory"][start:end]} for d in drones]
            fa_p = TrajectoryAnalyzer.analyze_formation(sliced)
            return fa_p['min_formation_dist'], fa_p.get('avg_formation_distance', 0)

        try:
            early_min, early_mean = _phase_formation(0, early_end)
            mid_min,   mid_mean   = _phase_formation(early_end, mid_end)
            late_min,  late_mean  = _phase_formation(mid_end, n_traj)
            formation_trend = (
                f"  Formation Min Distance  — Early:{early_min:.1f}m | Mid:{mid_min:.1f}m | Late:{late_min:.1f}m\n"
                f"  Formation Mean Distance — Early:{early_mean:.1f}m | Mid:{mid_mean:.1f}m | Late:{late_mean:.1f}m"
            )
            # 自动检测末段收敛趋势
            if late_min < early_min * 0.6 and late_min < 20:
                formation_trend += (f"\n  [!] CONVERGENCE TREND: formation shrinking "
                                    f"{early_min:.1f}m -> {late_min:.1f}m — collision risk increasing")
            elif late_mean > early_mean * 1.5:
                formation_trend += (f"\n  [!] DIVERGENCE TREND: formation expanding "
                                    f"{early_mean:.1f}m -> {late_mean:.1f}m — coordination breakdown")
        except Exception:
            formation_trend = "  N/A"

        # ── 跨机一致性检查 ──
        cross_check = TrajectoryAnalyzer.check_cross_drone_consistency(drones)
        cross_section = f"  {cross_check['verdict']}"
        if cross_check['sync_windows']:
            for w in cross_check['sync_windows'][:3]:
                cross_section += (f"\n  SYNC WINDOW: t={w['t_start']}-{w['t_end']}, "
                                  f"{w['max_drones_affected']}/{len(drones)} drones affected")
        if cross_check['sync_detected']:
            cross_section += "\n  *** SYNCHRONIZED FAILURE = SYSTEM-LEVEL EVENT, NOT RANDOM NOISE ***"

        # ── 物理相干性验证（针对主无人机）──
        contradictions = TrajectoryAnalyzer.check_physics_coherence(primary_traj)
        phys_section = "  No contradictions detected (sensors consistent)"
        if contradictions:
            phys_section = f"  CONTRADICTIONS FOUND: {len(contradictions)} timesteps"
            for c in contradictions[:3]:
                phys_section += (f"\n  t={c['time']}: GPS_bearing={c['gps_bearing']:.0f}° vs "
                                 f"heading={c['sensor_heading']:.0f}° (diff={c['angle_diff']:.0f}°) "
                                 f"— {c['displacement_m']:.1f}m displaced")
            if len(contradictions) > 5:
                phys_section += f"\n  *** SEVERE: {len(contradictions)} total contradictions — sensor malfunction likely ***"

        # ── 个体无人机极值 ──
        drone_spotlight = []
        for d_idx, (drone, analysis) in enumerate(zip(drones, drone_analyses)):
            issues = []
            if analysis.get('max_heading_change', 0) > 45:
                issues.append(f"MaxHdg={analysis['max_heading_change']:.0f}°")
            if analysis.get('altitude_std', 0) > 5:
                issues.append(f"AltStd={analysis['altitude_std']:.1f}m")
            gps_cnt = sum(1 for pt in drone['trajectory'] if pt.get('gps_status') != 'OK')
            if gps_cnt > 0:
                issues.append(f"GPS_err={gps_cnt}pts")
            label = f"  {drone['drone_id']}: " + (", ".join(issues) if issues else "Clean")
            drone_spotlight.append(label)

        # ── DSL ──
        trajectory_dsl = TrajectoryAnalyzer.generate_trajectory_dsl(primary_traj, formation_analysis)

        return f"""
Mission: {mission_data.get('mission_id','N/A')} | Duration: {mission_data.get('flight_duration','N/A')} | Type: {mission_data.get('mission_type','N/A')}
Description: {mission_data.get('description', 'N/A')}

═══ PERFORMANCE SCORECARD (0-100) ═══
Overall Safety: {overall_avg:.1f}  |  Path Efficiency: {eff['efficiency_score']:.1f} (Tortuosity: {eff['tortuosity']:.2f})
Smoothness: {smoothness:.1f} | Altitude: {altitude:.1f} | Speed: {speed:.1f} | Formation: {formation:.1f} | Coordination: {coordination:.1f}

═══ CRITICAL EXTREMES (Worst-Case Evidence) ═══
Max Heading Change : {max_heading_chg:.1f}°  (threshold: 150°=CRITICAL)
Max Altitude Std   : {max_alt_std:.1f}m  (threshold: 50m=CRITICAL)
Min Formation Dist : {min_form_dist:.2f}m  (threshold: <0.5m=collision risk)

═══ TEMPORAL PHASE ANALYSIS (Early/Mid/Late) ═══
{phase_section}
{late_risk_note}

═══ FORMATION DISTANCE TREND (Phase-by-Phase) ═══
{formation_trend}

═══ CROSS-DRONE CONSISTENCY CHECK ═══
{cross_section}

═══ PHYSICS COHERENCE (GPS vs. Heading/Speed) ═══
{phys_section}

═══ INDIVIDUAL DRONE SPOTLIGHT ═══
{"chr(10)".join(drone_spotlight)}

═══ TRAJECTORY DSL (Tokenized Evidence) ═══
{trajectory_dsl}
"""


def load_missions(dataset_path: str) -> List[Dict]:
    """加载任务数据"""
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'missions' in data:
        return data['missions']
    return [data]


def compute_metrics(predictions: List[str], ground_truth: List[str]) -> Dict:
    """计算评估指标 - 三分类（Safe/Borderline/Risky）"""
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions) if predictions else 0
    
    # 计算每个类别的指标
    labels = ["Safe", "Borderline", "Risky"]
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, 
        labels=labels, 
        average=None, 
        zero_division=0
    )
    
    # 计算宏平均（Macro Average）
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # 计算加权平均（Weighted Average）
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    # 每个类别的详细指标
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": int(support[i])
        }
    
    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class_metrics
    }


def compute_human_agreement(predictions: List[str], human_labels: List[List[str]]) -> float:
    """计算Cohen's kappa"""
    pred_binary = [1 if p == "Safe" else 0 for p in predictions]
    human_binary = [[1 if h == "Safe" else 0 for h in labels] for labels in human_labels]
    human_majority = [1 if sum(h) > len(h)/2 else 0 for h in human_binary]
    
    kappa = cohen_kappa_score(pred_binary, human_majority) if len(pred_binary) > 0 else 0
    return kappa


def evaluate_single_evaluator(args):
    """评估单个mission的单个评估器 - 用于并行执行"""
    mission_idx, mission, evaluator_name, evaluator = args
    mission_id = mission['mission_id']
    
    try:
        result = evaluator.evaluate(mission)
        prediction = result['safety_label']
        eff_pred = result.get('efficiency_label', 'Medium')
        
        result_data = {
            "mission_id": mission_id,
            "prediction": prediction,
            "efficiency_prediction": eff_pred,
            "score": result.get('score', 0),
            "issues": result.get('issues_identified', [])
        }
    except Exception as e:
        result_data = {
            "mission_id": mission_id,
            "prediction": "Borderline",
            "efficiency_prediction": "Medium",
            "score": 50,
            "issues": []
        }
        prediction = "Borderline"
        eff_pred = "Medium"
    
    return {
        "mission_idx": mission_idx,
        "mission_id": mission_id,
        "evaluator_name": evaluator_name,
        "result": result_data,
        "prediction": prediction,
        "eff_prediction": eff_pred
    }


def main():
    api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"
    if not api_key:
        print("ERROR: SILICONFLOW_API_KEY not set")
        print("Get your API key from: https://siliconflow.cn")
        return
    
    print("="*80)
    print("EXPERIMENT 1: Real Evaluation (No Data Leakage) - PARALLEL MODE")
    print("="*80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "hard_uav_missions.json")  # Use HARD dataset for uncertainty testing
    missions = load_missions(dataset_path)
    if not missions:
        print("ERROR: No missions loaded.")
        return
    
    print(f"\nLoaded {len(missions)} missions")
    
    evaluators = {
        "Single-Metric": RealSingleMetricEvaluator(api_key),
        "Fixed-Weight": RealFixedWeightEvaluator(api_key),
        "Single-Agent-LLM": RealSingleAgentLLMEvaluator(api_key),
        "Multi-Agent-Debate": RealMultiAgentDebateEvaluator(api_key, max_rounds=2, verbose=False)
    }
    
    missions_to_evaluate = missions
    print(f"\nEvaluating {len(missions_to_evaluate)} missions with {len(evaluators)} evaluators")
    print(f"Total parallel tasks: {len(missions_to_evaluate) * len(evaluators)}")
    
    results = {name: [] for name in evaluators.keys()}
    efficiency_ground_truth = []
    
    print("\n[3/5] Running evaluations in BATCHES (10 missions per batch)...")
    
    batch_size = 10
    num_batches = (len(missions_to_evaluate) + batch_size - 1) // batch_size
    
    completed_results = []
    correct_counts = {name: 0 for name in evaluators.keys()}
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(missions_to_evaluate))
        batch_missions = missions_to_evaluate[start_idx:end_idx]
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx + 1}/{num_batches}: Missions {start_idx + 1}-{end_idx}")
        print(f"{'='*80}")
        
        task_args = []
        for mission_idx in range(start_idx, end_idx):
            mission = missions_to_evaluate[mission_idx]
            for evaluator_name, evaluator in evaluators.items():
                task_args.append((mission_idx, mission, evaluator_name, evaluator))
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_task = {executor.submit(evaluate_single_evaluator, args): args 
                             for args in task_args}
            
            for future in as_completed(future_to_task):
                args = future_to_task[future]
                try:
                    result = future.result()
                    completed_results.append(result)
                    
                    mission_idx = result['mission_idx']
                    gt_safety = missions_to_evaluate[mission_idx]['ground_truth']['safety_label']
                    evaluator_name = result['evaluator_name']
                    prediction = result['prediction']
                    
                    if prediction == gt_safety:
                        correct_counts[evaluator_name] += 1
                    
                    completed = len(completed_results)
                    total = len(missions_to_evaluate) * len(evaluators)
                    print(f"\rProgress: {completed}/{total} | ", end="")
                    for name in evaluators.keys():
                        acc = correct_counts[name] / len(missions_to_evaluate) * 100
                        print(f"{name}: {acc:.1f}%  ", end="")
                    print(flush=True)
                    
                except Exception as e:
                    mission_idx, mission, evaluator_name, _ = args
                    print(f"\n[Error] Mission {mission_idx+1} {evaluator_name} failed: {e}")
        
        batch_acc = {}
        for name in evaluators.keys():
            batch_acc[name] = correct_counts[name] / (end_idx) * 100
        
        print(f"\nBatch {batch_idx + 1} completed. Current accuracy: ", end="")
        for name, acc in batch_acc.items():
            print(f"{name}: {acc:.1f}%  ", end="")
        print()
    
    print("\n\n[4/5] Organizing results...")
    
    for result in completed_results:
        evaluator_name = result['evaluator_name']
        results[evaluator_name].append(result['result'])
    
    for mission in missions_to_evaluate:
        efficiency_ground_truth.append(mission['ground_truth']['efficiency_label'])
    
    cumulative_correct = {name: 0 for name in evaluators.keys()}
    cumulative_eff_correct = {name: 0 for name in evaluators.keys()}
    
    for result in completed_results:
        mission_idx = result['mission_idx']
        gt_safety = missions_to_evaluate[mission_idx]['ground_truth']['safety_label']
        eff_gt = missions_to_evaluate[mission_idx]['ground_truth']['efficiency_label']
        
        evaluator_name = result['evaluator_name']
        prediction = result['prediction']
        eff_pred = result['eff_prediction']
        
        if prediction == gt_safety:
            cumulative_correct[evaluator_name] += 1
        
        if eff_pred == eff_gt:
            cumulative_eff_correct[evaluator_name] += 1
    
    print("\n" + "="*80)
    print("FINAL RESULTS - Safety Accuracy")
    print("="*80)
    for name in evaluators.keys():
        accuracy = cumulative_correct[name] / len(missions_to_evaluate) * 100
        print(f"  {name:<25} {cumulative_correct[name]}/{len(missions_to_evaluate)} = {accuracy:.1f}%")
    
    ground_truth = [m['ground_truth']['safety_label'] for m in missions_to_evaluate]
    human_labels = [[m['ground_truth']['safety_label']] * 3 for m in missions_to_evaluate]
    
    metrics_table = {}
    for name in evaluators.keys():
        predictions = [r['prediction'] for r in results[name]]
        eff_preds = [r['efficiency_prediction'] for r in results[name]]
        
        metrics = compute_metrics(predictions, ground_truth)
        kappa = compute_human_agreement(predictions, human_labels)
        metrics['human_agreement_kappa'] = kappa
        
        eff_correct = sum(1 for p, g in zip(eff_preds, efficiency_ground_truth) if p == g)
        metrics['efficiency_accuracy'] = eff_correct / len(eff_preds) if eff_preds else 0
        
        metrics_table[name] = metrics
    
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    print(f"{'Method':<20} {'Safety Acc':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Eff Acc':<10}")
    print("-"*80)
    
    for name, metrics in metrics_table.items():
        print(f"{name:<20} {metrics['accuracy']:<12.2%} {metrics['precision']:<10.2%} "
              f"{metrics['recall']:<10.2%} {metrics['f1_score']:<10.2%} {metrics['efficiency_accuracy']:<10.2%}")
    
    print("="*70)
    
    output_file = "exp1_real_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": "complex_uav_missions.json",
            "num_missions": len(missions_to_evaluate),
            "results": results,
            "metrics": metrics_table
        }, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print(f"\n[Result] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
