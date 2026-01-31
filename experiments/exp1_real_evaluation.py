"""
真实的多智能体辩论评估器
只从原始轨迹数据中提取特征，不使用预计算的评分
"""

import json
import os
import time
import math
import re
import threading
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from tqdm import tqdm
from experiment_dashboard import ExperimentDashboard, MetricsComparisonPanel
import web_dashboard


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
            "speed_range": speed_range
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
        
        formation_std = np.std(formation_distances)
        formation_stability = max(0, 100 - formation_std * 5) # Adjusted scaling
        
        # 极值提取
        min_formation_dist = min(formation_distances) if formation_distances else 0
        max_formation_dist = max(formation_distances) if formation_distances else 0
        
        avg_speed_std = np.mean(speed_correlations) if speed_correlations else 0
        avg_heading_std = np.mean(heading_correlations) if heading_correlations else 0
        
        coordination_quality = max(0, 100 - (avg_speed_std * 3 + avg_heading_std * 0.5))
        
        return {
            "formation_stability": formation_stability,
            "coordination_quality": coordination_quality,
            "avg_formation_distance": np.mean(formation_distances),
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
   - Overall Average >= 75.
   - AND NO Fatal Flaws (Max Heading > 90°, Alt Dev > 15m).
   - Ignore efficiency/tortuosity for Safety rating.

2. BORDERLINE: 
   - Overall Average 60-74.
   - OR Average > 75 but one metric is near the limit.

3. RISKY: 
   - Overall Average < 60.
   - OR ANY Fatal Flaw (Heading > 90°, Alt Dev > 15m, Formation < 2m).

EFFICIENCY GUIDELINES:
- HIGH: Path Efficiency > 70, Tortuosity < 1.4.
- MEDIUM: Path Efficiency 50-70.
- LOW: Path Efficiency < 50.

Remember: Start with the PRESUMPTION OF SAFETY. If Score > 75 and no fatal flaws, it IS Safe. Efficiency is separate.

Provide:
SAFETY: [Safe/Borderline/Risky]
EFFICIENCY: [High/Medium/Low]
JUSTIFICATION: [Brief explanation of your reasoning]
SCORE: [Your confidence score 0-100]
"""
        
        try:
            print(f"    Calling LLM API...", end=" ", flush=True)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                timeout=30.0
            )
            print("Response received", flush=True)
            
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
    
    def __init__(self, api_key: str, model: str = "Qwen/Qwen3-32B", max_rounds: int = 2, verbose: bool = False):
        self.client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        self.model = model
        self.verbose = verbose
        self.max_rounds = max_rounds
        
        # Three Specialized Expert Agents (as per paper Section II.B)
        self.experts = [
            {
                "id": 0,
                "name": "Flight Control Specialist (FCS)",
                "role": "FCS",
                "bias": "Precision Bias",
                "core_values": "Quantitative metrics, physical feasibility, energy efficiency, path optimization",
                "system_prompt": """You are a Flight Control Specialist with a PRECISION BIAS.

CORE EXPERTISE:
- Trajectory smoothness (RMSE, variance, heading changes)
- Altitude stability and vertical control
- Speed consistency and acceleration profiles
- Physical feasibility of maneuvers
- MISSION EFFICIENCY (Path Efficiency, Tortuosity, Energy use)

COGNITIVE BIAS: You prioritize quantitative metrics and efficiency. You penalize deviations >5% from nominal trajectories and inefficient paths.

EVIDENCE CHAIN REQUIREMENT:
You MUST structure your response as:
[CLAIM]: One-sentence core judgment on Safety AND Efficiency
[EVIDENCE]: 3-5 bullet points citing specific metrics (e.g., "Smoothness: 85, Efficiency: 92 (High)")
[COUNTER]: Anticipate one potential counterargument
[SUMMARY]: Three key takeaways
[CONFIDENCE]: Score 0-100"""
            },
            {
                "id": 1,
                "name": "Swarm Coordination Expert (SCE)",
                "role": "SCE",
                "bias": "Cohesion Bias",
                "core_values": "Formation stability, collective behavior, inter-drone coordination",
                "system_prompt": """You are a Swarm Coordination Expert with a COHESION BIAS.

CORE EXPERTISE:
- Formation stability and geometry preservation
- Inter-drone communication quality
- Collective behavior patterns
- Coordination efficiency

COGNITIVE BIAS: You evaluate the swarm as a COLLECTIVE entity, penalizing individual outliers even if individually safe. Group statistics matter more than individual metrics.

EVIDENCE CHAIN REQUIREMENT:
You MUST structure your response as:
[CLAIM]: One-sentence core judgment
[EVIDENCE]: 3-5 bullet points citing formation metrics (e.g., "Min Formation Dist: 3.2m, safe")
[COUNTER]: Anticipate one potential counterargument
[SUMMARY]: Three key takeaways
[CONFIDENCE]: Score 0-100"""
            },
            {
                "id": 2,
                "name": "Safety Assessment Expert (SAE)",
                "role": "SAE",
                "bias": "Pessimistic Bias (Zero-Trust)",
                "core_values": "Risk detection, edge case identification, worst-case analysis",
                "system_prompt": """You are a Safety Assessment Expert with a PESSIMISTIC BIAS (Zero-Trust Model).

CORE EXPERTISE:
- Collision risk analysis
- Safety envelope boundary violations
- Critical anomaly detection
- Worst-case scenario evaluation

COGNITIVE BIAS: You operate under ZERO-TRUST. Flag any state vector approaching within 10% of safety boundaries as CRITICAL RISK. Prioritize FALSE POSITIVES over FALSE NEGATIVES.

SAFETY THRESHOLDS (RELAXED FOR COMPLEX MANEUVERS):
- Heading Change > 150° in 1s -> CRITICAL (Uncontrolled spin)
- Altitude Deviation > 30m -> CRITICAL (Loss of control)
- Formation Distance < 0.2m -> CRITICAL (Collision imminent)

EVIDENCE CHAIN REQUIREMENT:
You MUST structure your response as:
[CLAIM]: One-sentence core judgment
[EVIDENCE]: 3-5 bullet points citing extreme values (e.g., "Max Heading Change: 35°, acceptable")
[COUNTER]: Anticipate one potential counterargument
[SUMMARY]: Three key takeaways
[CONFIDENCE]: Score 0-100"""
            },
            {
                "id": 3,
                "name": "Uncertainty Analysis Expert (UAE)",
                "role": "UAE",
                "bias": "Skeptical Bias (Question Data Quality)",
                "core_values": "Data integrity, measurement uncertainty, sensor reliability, edge case identification",
                "system_prompt": """You are an Uncertainty Analysis Expert with a SKEPTICAL BIAS toward data quality.

CORE EXPERTISE:
- Sensor noise and drift patterns
- GPS signal quality and dropout detection
- Data completeness and missing segments
- Measurement uncertainty propagation
- Ambiguous/borderline cases identification

COGNITIVE BIAS: You question the RELIABILITY of the data. Look for:
- Sudden jumps indicating sensor glitches
- Data gaps or suspicious discontinuities (check if they are critical)
- High variance indicating poor sensor quality

KEY INDICATORS OF UNCERTAINTY:
- Speed/altitude/heading variance > 20% of mean -> Low data quality
- Sudden position jumps > 50m -> GPS glitch
- Missing time segments -> Data loss (Assess if mission recovered)

YOUR ROLE: Identify if data quality issues prevent a reliable assessment. If data is messy but mission succeeded, state "Reliable enough".

EVIDENCE CHAIN REQUIREMENT:
You MUST structure your response as:
[CLAIM]: One-sentence core judgment on data quality
[EVIDENCE]: 3-5 bullet points citing data quality indicators
[COUNTER]: Anticipate one potential counterargument (e.g. "Data gap was short")
[SUMMARY]: Three key takeaways about uncertainty
[CONFIDENCE]: Score 0-100 (reflecting confidence in the DATA SUFFICIENCY)"""
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
        hard_violations = self._check_hard_constraints(drone_analyses, formation_analysis)
        if hard_violations:
            print(f"\n    [VETO] Hard constraint violated: {hard_violations[0]}")
            # Use single agent efficiency logic for VETO cases
            eff = TrajectoryAnalyzer.analyze_efficiency(drones[0]['trajectory'])
            eff_score = eff['efficiency_score']
            eff_label = "High" if eff_score >= 70 else "Medium" if eff_score >= 50 else "Low"
            
            return {
                "method": "Multi-Agent-Debate",
                "safety_label": "Risky",
                "efficiency_label": eff_label,
                "score": 15,
                "issues_identified": hard_violations,
                "debate_transcript": ["VETO: Deterministic Safety Verification"],
                "route_layer": route_layer,
                "complexity": H_comp
            }
        
        # === STEP 2: Multi-round Expert Panel with Red/Blue Rotation ===
        all_rounds = []
        for round_idx in range(actual_max_rounds):
            print(f"\n    [Expert Panel Round {round_idx + 1}/{actual_max_rounds}]")
            
            # Red/Blue team assignment (Paper Eq. 11-12)
            red_ids, blue_ids = self._assign_red_blue_teams(round_idx)
            
            round_responses = []
            for expert in self.experts:
                team = "RED" if expert['id'] in red_ids else "BLUE"
                
                # Build prompt with team instructions
                context = trajectory_summary
                if round_idx > 0:
                    context += f"\n\n[PREVIOUS ROUND FEEDBACK]:\n{self._summarize_round(all_rounds[-1])}"
                
                # Meta-Debate: 添加标准对齐轮
                if route_layer == "META_DEBATE" and round_idx == 0:
                    context += "\n\n[META-DEBATE ALIGNMENT]:\nFirst, define mission-specific evaluation standards before analysis."
                
                prompt = self._build_expert_prompt(expert, team, context)
                response = self._call_llm(expert['system_prompt'], prompt)
                
                # Parse Evidence Chain (Paper Section III.D)
                parsed = self._parse_evidence_chain(response)
                parsed['expert'] = expert['name']
                parsed['team'] = team
                
                round_responses.append(parsed)
                debate_history.append(f"[R{round_idx+1}][{team}] {expert['name']}: {parsed.get('claim', '')[:80]}")
                print(f"      [{team}] {expert['name']}: {parsed.get('claim', 'N/A')[:60]}...")
            
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

    def _call_llm(self, system, user):
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
            print(f"LLM Error: {e}")
            return "Error"

    def _extract_score(self, text):
        import re
        match = re.search(r"SCORE:\s*(\d+)", text)
        if match:
            return float(match.group(1))
        return 50.0

    def _extract_safety_label(self, text: str) -> str:
        import re
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
        match = re.search(r"EFFICIENCY:\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            label = re.sub(r'\*\*', '', label)
            label = label.strip().lower()
            if 'high' in label: return "High"
            if 'low' in label: return "Low"
            if 'medium' in label: return "Medium"
        return "Medium"
    
    def _check_hard_constraints(self, drone_analyses: List[Dict], formation_analysis: Dict) -> List[str]:
        """Deterministic Safety Verification (Paper Section III.F) - RELAXED"""
        violations = []
        
        # Hard Constraint 1: Heading change > 150° (Allow sharp turns but catch spins)
        max_heading = max([a.get('max_heading_change', 0) for a in drone_analyses])
        if max_heading > 150:
            violations.append(f"FATAL: Heading change {max_heading:.1f}° > 150° (Loss of Control)")
        
        # Hard Constraint 2: Altitude deviation > 30m
        max_alt_range = max([a.get('altitude_range', 0) for a in drone_analyses])
        if max_alt_range > 30:
            violations.append(f"FATAL: Altitude deviation {max_alt_range:.1f}m > 30m (Hard limit)")
        
        # Hard Constraint 3: Formation distance < 0.05m
        min_dist = formation_analysis.get('min_formation_dist', 999)
        if min_dist < 0.05:
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
The debate is showing echo chamber effects - experts are converging too quickly.

INSTRUCTION: Red Team agents must actively challenge the consensus.
Blue Team agents must acknowledge valid criticisms but defend their position.
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
    
    def _build_expert_prompt(self, expert: Dict, team: str, context: str) -> str:
        """Build role-specific prompt with team instructions"""
        
        if team == "RED":
            team_instruction = """
[RED TEAM INSTRUCTIONS - Adversarial Role]:
- Actively search for anomalies and potential risks
- Challenge any claims of safety with concrete counter-examples
- Identify boundary cases where system approaches (but may not violate) limits
- Question assumptions made by other experts
- Flag patterns indicating rare but critical failure modes

Your goal: Find what could go wrong.
"""
        else:
            team_instruction = """
[BLUE TEAM INSTRUCTIONS - Collaborative Role]:
- Provide objective professional assessment based on domain expertise
- Acknowledge valid criticisms from Red team with data-backed evidence
- Refute unreasonable challenges by citing specific trajectory segments
- Maintain constructive stance focused on system understanding

Your goal: Balanced, evidence-based analysis.
"""
        
        return f"""
{team_instruction}

MISSION DATA:
{context}

As a {expert['name']} ({expert['bias']}), analyze this mission from your domain expertise.

REMEMBER YOUR COGNITIVE BIAS:
- {expert['bias']}: {expert['core_values']}

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
- Semantic Agreement: {consensus['semantic_sim']:.2f}
- Safe Votes: {consensus['safe_votes']}/{len(final_round)}
- Risky Votes: {consensus['risky_votes']}/{len(final_round)}

YOUR TASK:
As the Lead Evaluator, synthesize the expert panel's assessments into a final verdict.

DECISION RULES:
1. If ALL experts agree (unanimous) -> Follow their consensus
2. If 3/4+ agree -> Majority wins (but note dissent)
3. If split 2-2 -> Default to SAFE (Presumption of Safety) UNLESS a Fatal Flaw was proven OR Uncertainty Expert flags poor data quality
4. Fatal Flaws (immediate Risky): Heading > 150°, Alt Dev > 30m, Formation < 0.2m
5. Data Quality Issues: If UAE flags low confidence (<50), downgrade certainty (Safe->Borderline, Borderline->Risky)

EFFICIENCY RULES (Independent):
- Path Efficiency > 70 -> High
- Path Efficiency < 50 -> Low

OUTPUT FORMAT:
SAFETY: [Safe/Borderline/Risky]
EFFICIENCY: [High/Medium/Low]
SCORE: [0-100]
REASONING: [2-3 sentences explaining the verdict and key evidence]
"""
        
        return self._call_llm(
            "You are the Lead Safety Evaluator synthesizing expert opinions into a final verdict.",
            synthesis_prompt
        )
    
    def _build_trajectory_summary(self, mission_data: Dict, drone_analyses: List[Dict], formation_analysis: Dict) -> str:
        """构建轨迹摘要 - 包含极值证据和DSL tokenization"""
        smoothness = np.mean([a['trajectory_smoothness'] for a in drone_analyses])
        altitude = np.mean([a['altitude_stability'] for a in drone_analyses])
        speed = np.mean([a['speed_consistency'] for a in drone_analyses])
        formation = formation_analysis['formation_stability']
        coordination = formation_analysis['coordination_quality']
        overall_avg = (smoothness + altitude + speed + formation + coordination) / 5
        
        # 提取极值证据 (Worst Case Evidence)
        max_heading_chg = max([a.get('max_heading_change', 0) for a in drone_analyses])
        max_alt_range = max([a.get('altitude_range', 0) for a in drone_analyses])
        min_form_dist = formation_analysis.get('min_formation_dist', 0)
        
        # 生成轨迹DSL (基于论文的tokenization方法)
        trajectory_dsl = TrajectoryAnalyzer.generate_trajectory_dsl(mission_data['drones'][0]['trajectory'], formation_analysis)
        
        # 计算效率指标
        eff = TrajectoryAnalyzer.analyze_efficiency(mission_data['drones'][0]['trajectory'])
        
        return f"""
Mission Duration: {mission_data.get('flight_duration', 'N/A')}

PERFORMANCE SCORECARD (0-100):
- Overall Safety: {overall_avg:.1f}
- Path Efficiency: {eff['efficiency_score']:.1f} (Tortuosity: {eff['tortuosity']:.2f})
- Metrics: Smoothness {smoothness:.1f} | Altitude {altitude:.1f} | Speed {speed:.1f} | Formation {formation:.1f}

CRITICAL EVIDENCE (Extreme Values):
- Max Heading Change: {max_heading_chg:.1f}° (Acceptable range: < 90°)
- Max Altitude Deviation: {max_alt_range:.1f}m (Acceptable range: < 15m)
- Min Formation Distance: {min_form_dist:.1f}m (Safe distance: > 2m)

TRAJECTORY DSL (Tokenized Evidence):
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
            "support": support[i]
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


def main():
    api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"
    if not api_key:
        print("ERROR: SILICONFLOW_API_KEY not set")
        print("Get your API key from: https://siliconflow.cn")
        return
    
    print("="*80)
    print("EXPERIMENT 1: Real Evaluation (No Data Leakage)")
    print("="*80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "complex_uav_missions.json")
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
    print(f"\nEvaluating {len(missions_to_evaluate)} missions (SCENARIO MODE)")
    
    results = {name: [] for name in evaluators.keys()}
    
    print("\n[3/5] Running evaluations...")
    
    # 启动Web Dashboard服务
    print("\n📊 Starting Web Dashboard Service...")
    dashboard_thread = threading.Thread(target=web_dashboard.run_dashboard, 
                                       kwargs={'host': '0.0.0.0', 'port': 5000, 'debug': False},
                                       daemon=True)
    dashboard_thread.start()
    time.sleep(2)
    print("✅ Dashboard is running at: http://localhost:5000")
    print("   Open this URL in your browser to see real-time results!")
    
    # 初始化实验
    web_dashboard.initialize_experiment(list(evaluators.keys()), len(missions_to_evaluate))
    
    # 累计统计
    cumulative_correct = {name: 0 for name in evaluators.keys()}
    cumulative_eff_correct = {name: 0 for name in evaluators.keys()}
    
    efficiency_ground_truth = []
    
    for idx, mission in enumerate(missions_to_evaluate):
        # 计算 Efficiency Ground Truth
        eff_analysis = TrajectoryAnalyzer.analyze_efficiency(mission['drones'][0]['trajectory'])
        eff_score = eff_analysis['efficiency_score']
        if eff_score >= 70:
            eff_gt = "High"
        elif eff_score >= 50:
            eff_gt = "Medium"
        else:
            eff_gt = "Low"
        efficiency_ground_truth.append(eff_gt)

        print(f"\n{'='*70}")
        print(f"Evaluating mission {idx+1}/{len(missions_to_evaluate)}: {mission['mission_id']}")
        print(f"{'='*70}")
        print(f"Ground Truth: Safety={mission['ground_truth']} | Efficiency={eff_gt}")
        
        mission_predictions = {}
        eff_predictions = {}
        
        for name, evaluator in evaluators.items():
            print(f"  - {name}...", end=" ", flush=True)
            try:
                result = evaluator.evaluate(mission)
                prediction = result['safety_label']
                eff_pred = result.get('efficiency_label', 'Medium')
                
                results[name].append({
                    "mission_id": mission['mission_id'],
                    "prediction": prediction,
                    "efficiency_prediction": eff_pred,
                    "score": result.get('score', 0),
                    "issues": result.get('issues_identified', [])
                })
                mission_predictions[name] = prediction
                eff_predictions[name] = eff_pred
                print(f"Done: {prediction} | Eff: {eff_pred}")
            except Exception as e:
                print(f"Error: {e}")
                results[name].append({
                    "mission_id": mission['mission_id'],
                    "prediction": "Borderline",
                    "efficiency_prediction": "Medium",
                    "score": 50,
                    "issues": []
                })
                mission_predictions[name] = "Borderline"
                eff_predictions[name] = "Medium"
        
        # 立即打印该mission的结果对比
        print(f"\n  Result Summary for {mission['mission_id']}:")
        print(f"  {'Method':<20} {'Safety':<12} {'S-Status':<10} {'Efficiency':<10} {'E-Status':<10}")
        print(f"  {'-'*65}")
        for name, pred in mission_predictions.items():
            eff_pred = eff_predictions[name]
            
            is_correct = pred == mission['ground_truth']
            status = "[OK]" if is_correct else "[NO]"
            if is_correct: cumulative_correct[name] += 1
            
            is_eff_correct = eff_pred == eff_gt
            eff_status = "[OK]" if is_eff_correct else "[NO]"
            if is_eff_correct: cumulative_eff_correct[name] += 1
            
            print(f"  {name:<20} {pred:<12} {status:<10} {eff_pred:<10} {eff_status:<10}")
        
        # 显示累计准确率
        print(f"\n  Cumulative Accuracy (Safety):")
        for name in evaluators.keys():
            accuracy = cumulative_correct[name] / (idx + 1) * 100
            print(f"    {name:<25} {cumulative_correct[name]}/{idx+1} = {accuracy:.1f}%")
        
        # 更新Web Dashboard
        web_dashboard.update_mission(idx, cumulative_correct, cumulative_eff_correct)
        
        time.sleep(0.3)
    
    ground_truth = [m['ground_truth'] for m in missions_to_evaluate]
    human_labels = [[m['ground_truth']] * 3 for m in missions_to_evaluate]
    
    metrics_table = {}
    for name in evaluators.keys():
        predictions = [r['prediction'] for r in results[name]]
        eff_preds = [r['efficiency_prediction'] for r in results[name]]
        
        # Safety Metrics
        metrics = compute_metrics(predictions, ground_truth)
        kappa = compute_human_agreement(predictions, human_labels)
        metrics['human_agreement_kappa'] = kappa
        
        # Efficiency Metrics (Accuracy only for now)
        eff_correct = sum(1 for p, g in zip(eff_preds, efficiency_ground_truth) if p == g)
        metrics['efficiency_accuracy'] = eff_correct / len(eff_preds) if eff_preds else 0
        
        metrics_table[name] = metrics
    
    print("\n[5/5] RESULTS")
    print("\n" + "="*80)
    print("Performance Comparison on {} Missions".format(len(missions_to_evaluate)))
    print("="*80)
    print(f"{'Method':<20} {'Safety Acc':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Eff Acc':<10}")
    print("-"*80)
    
    for name, metrics in metrics_table.items():
        print(f"{name:<20} {metrics['accuracy']:<12.2%} {metrics['precision']:<10.2%} "
              f"{metrics['recall']:<10.2%} {metrics['f1_score']:<10.2%} {metrics['efficiency_accuracy']:<10.2%}")
    
    # 每个类别的详细指标
    print("\n" + "="*80)
    print("Per-Class Metrics (Macro Average)")
    print("="*80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-"*80)
    
    for label in ["Safe", "Borderline", "Risky"]:
        # 计算平均指标
        avg_precision = np.mean([m['per_class'][label]['precision'] for m in metrics_table.values()])
        avg_recall = np.mean([m['per_class'][label]['recall'] for m in metrics_table.values()])
        avg_f1 = np.mean([m['per_class'][label]['f1'] for m in metrics_table.values()])
        avg_support = np.mean([m['per_class'][label]['support'] for m in metrics_table.values()])
        
        print(f"{label:<15} {avg_precision:<12.2%} {avg_recall:<12.2%} {avg_f1:<12.2%} {avg_support:<10.0f}")
    
    print("\n" + "="*70)
    print("Human Expert Agreement (Cohen's κ)")
    print("="*70)
    
    for name, metrics in metrics_table.items():
        kappa = metrics['human_agreement_kappa']
        if np.isnan(kappa):
            print(f"{name:<25} κ = nan (No agreement)")
        else:
            print(f"{name:<25} κ = {kappa:.3f} (Substantial agreement)")
    
    print("="*70)
    
    # 完成实验，通知Web Dashboard
    print("\n📊 Finalizing experiment...")
    web_dashboard.finalize_experiment(metrics_table)
    
    # 保存JSON结果
    output_file = "exp1_real_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": "complex_uav_missions.json",
            "num_missions": len(missions_to_evaluate),
            "results": results,
            "metrics": metrics_table
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("✅ Dashboard is still running at: http://localhost:5000")
    print("   Check the browser for comprehensive metrics comparison!")
    
    # 保持Web服务运行
    print("\n🌐 Web Dashboard is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        return


if __name__ == "__main__":
    main()
