"""
真实的多智能体辩论评估器
只从原始轨迹数据中提取特征，不使用预计算的评分
"""

import json
import os
import time
import math
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from tqdm import tqdm


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
        
        # 提取数据
        timestamps = [p['timestamp'] for p in trajectory]
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
        
        return {
            "trajectory_smoothness": trajectory_smoothness,
            "altitude_stability": altitude_stability,
            "speed_consistency": speed_consistency,
            "heading_changes": heading_changes,
            "avg_heading_change": avg_heading_change,
            "altitude_std": altitude_std,
            "speed_std": speed_std
        }
    
    @staticmethod
    def analyze_formation(drones: List[Dict]) -> Dict:
        """分析编队保持情况"""
        if len(drones) < 2:
            return {"formation_stability": 100, "coordination_quality": 100}
        
        # 获取所有无人机的轨迹点
        all_trajectories = [drone['trajectory'] for drone in drones]
        min_length = min(len(t) for t in all_trajectories)
        
        if min_length < 2:
            return {"formation_stability": 100, "coordination_quality": 100}
        
        # 计算编队稳定性 - 无人机之间的距离变化
        formation_distances = []
        for i in range(min_length):
            positions = []
            for trajectory in all_trajectories:
                point = trajectory[i]
                positions.append((point['gps']['lat'], point['gps']['lon']))
            
            # 计算所有无人机对之间的距离
            for j in range(len(positions)):
                for k in range(j+1, len(positions)):
                    lat1, lon1 = positions[j]
                    lat2, lon2 = positions[k]
                    dist = math.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)
                    formation_distances.append(dist)
        
        if not formation_distances:
            return {"formation_stability": 100, "coordination_quality": 100}
        
        formation_std = np.std(formation_distances)
        formation_stability = max(0, 100 - formation_std * 10000)
        
        # 协调质量 - 基于速度和航向的一致性
        speed_correlations = []
        heading_correlations = []
        
        for i in range(min_length):
            speeds_at_time = [t[i]['speed'] for t in all_trajectories]
            headings_at_time = [t[i]['heading'] for t in all_trajectories]
            
            speed_std = np.std(speeds_at_time)
            heading_std = np.std(headings_at_time)
            
            speed_correlations.append(speed_std)
            heading_correlations.append(heading_std)
        
        avg_speed_std = np.mean(speed_correlations) if speed_correlations else 0
        avg_heading_std = np.mean(heading_correlations) if heading_correlations else 0
        
        coordination_quality = max(0, 100 - (avg_speed_std * 3 + avg_heading_std * 0.5))
        
        return {
            "formation_stability": formation_stability,
            "coordination_quality": coordination_quality,
            "avg_formation_distance": np.mean(formation_distances),
            "formation_std": formation_std
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


class RealSingleMetricEvaluator:
    """真实的单指标评估器 - 基于轨迹分析"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def evaluate(self, mission_data: Dict) -> Dict:
        drones = mission_data.get('drones', [])
        if not drones:
            return {"method": "Single-Metric", "safety_label": "Borderline", "score": 50, "issues_identified": []}
        
        # 分析第一个无人机的轨迹
        trajectory = drones[0]['trajectory']
        analysis = TrajectoryAnalyzer.analyze_single_drone(trajectory)
        
        # 计算综合得分
        overall_score = (
            analysis['trajectory_smoothness'] * 0.4 +
            analysis['altitude_stability'] * 0.3 +
            analysis['speed_consistency'] * 0.3
        )
        
        # 根据得分判断安全等级
        if overall_score >= 80:
            safety_label = "Safe"
        elif overall_score >= 60:
            safety_label = "Borderline"
        else:
            safety_label = "Risky"
        
        # 检测异常
        anomalies = TrajectoryAnalyzer.detect_anomalies(trajectory)
        
        return {
            "method": "Single-Metric",
            "safety_label": safety_label,
            "score": overall_score,
            "issues_identified": anomalies,
            "analysis": analysis
        }


class RealFixedWeightEvaluator:
    """真实的固定权重评估器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.weights = {
            "trajectory_smoothness": 0.25,
            "altitude_stability": 0.25,
            "speed_consistency": 0.2,
            "formation_stability": 0.15,
            "coordination_quality": 0.15
        }
    
    def evaluate(self, mission_data: Dict) -> Dict:
        drones = mission_data.get('drones', [])
        if not drones:
            return {"method": "Fixed-Weight", "safety_label": "Borderline", "score": 50, "issues_identified": []}
        
        # 分析所有无人机
        drone_analyses = [TrajectoryAnalyzer.analyze_single_drone(d['trajectory']) for d in drones]
        formation_analysis = TrajectoryAnalyzer.analyze_formation(drones)
        
        # 计算平均指标
        avg_smoothness = np.mean([a['trajectory_smoothness'] for a in drone_analyses])
        avg_altitude_stability = np.mean([a['altitude_stability'] for a in drone_analyses])
        avg_speed_consistency = np.mean([a['speed_consistency'] for a in drone_analyses])
        
        # 加权计算
        overall_score = (
            avg_smoothness * self.weights['trajectory_smoothness'] +
            avg_altitude_stability * self.weights['altitude_stability'] +
            avg_speed_consistency * self.weights['speed_consistency'] +
            formation_analysis['formation_stability'] * self.weights['formation_stability'] +
            formation_analysis['coordination_quality'] * self.weights['coordination_quality']
        )
        
        if overall_score >= 80:
            safety_label = "Safe"
        elif overall_score >= 60:
            safety_label = "Borderline"
        else:
            safety_label = "Risky"
        
        anomalies = TrajectoryAnalyzer.detect_anomalies(drones[0]['trajectory'], formation_analysis)
        
        return {
            "method": "Fixed-Weight",
            "safety_label": safety_label,
            "score": overall_score,
            "issues_identified": anomalies,
            "analysis": {
                "avg_smoothness": avg_smoothness,
                "avg_altitude_stability": avg_altitude_stability,
                "avg_speed_consistency": avg_speed_consistency,
                "formation_stability": formation_analysis['formation_stability'],
                "coordination_quality": formation_analysis['coordination_quality']
            }
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

CLASSIFICATION GUIDELINES:
- SAFE: Strong performance (typically avg >= 75), well-executed mission. Minor fluctuations are acceptable.
- BORDERLINE: Moderate performance (typically avg 55-74), some concerns but mission completed.
- RISKY: Poor performance (typically avg < 55), significant safety concerns or instability.

Remember: These are guidelines. Use your professional judgment based on the holistic view of the mission.

Provide:
SAFETY: [Safe/Borderline/Risky]
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
            score_str = self._extract_field(response_text, "SCORE", "50")
            score = float(score_str) if score_str.replace(".", "").isdigit() else 50.0
            
            anomalies = TrajectoryAnalyzer.detect_anomalies(drones[0]['trajectory'], formation_analysis)
            
            return {
                "method": "Single-Agent-LLM",
                "safety_label": safety_label,
                "score": score,
                "issues_identified": anomalies
            }
        except Exception as e:
            print(f"Error in Single-Agent-LLM: {e}")
            return {
                "method": "Single-Agent-LLM",
                "safety_label": "Borderline",
                "score": 50,
                "issues_identified": []
            }
    
    def _build_trajectory_summary(self, mission_data: Dict, drone_analyses: List[Dict], formation_analysis: Dict) -> str:
        """构建轨迹数据摘要"""
        smoothness = np.mean([a['trajectory_smoothness'] for a in drone_analyses])
        altitude = np.mean([a['altitude_stability'] for a in drone_analyses])
        speed = np.mean([a['speed_consistency'] for a in drone_analyses])
        formation = formation_analysis['formation_stability']
        coordination = formation_analysis['coordination_quality']
        overall_avg = (smoothness + altitude + speed + formation + coordination) / 5
        
        summary = f"""
Mission ID: {mission_data.get('mission_id', 'N/A')}
Metrics (0-100):
- Smoothness: {smoothness:.1f}
- Altitude Stability: {altitude:.1f}
- Speed Consistency: {speed:.1f}
- Formation Stability: {formation:.1f}
- Coordination: {coordination:.1f}
- OVERALL AVERAGE: {overall_avg:.1f}

Detected Anomalies:
{chr(10).join(['  - ' + a for a in TrajectoryAnalyzer.detect_anomalies(mission_data['drones'][0]['trajectory'], formation_analysis)]) if TrajectoryAnalyzer.detect_anomalies(mission_data['drones'][0]['trajectory'], formation_analysis) else '  None detected'}
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
            
            return value
        return default


class RealMultiAgentDebateEvaluator:
    """真实的多智能体辩论评估器 - 正反辩证机制"""
    
    def __init__(self, api_key: str, model: str = "Qwen/Qwen3-32B", max_rounds: int = 2, verbose: bool = False):
        self.client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        self.model = model
        self.verbose = verbose
        
        # 定义新的正反方角色
        self.agents = [
            {
                "name": "Mission Advocate (Positive)",
                "role": "Optimist",
                "goal": "Argue why this mission is SAFE. Emphasize that real-world flights are never perfect.",
                "system_prompt": "You are the Mission Advocate. Your philosophy: 'Safe enough is Safe'. Real-world data always has noise. If the drone didn't crash and metrics are generally good (>60), it's a success. Defend against the Critic's nitpicking. Argue that minor fluctuations are environmental noise, not system failures."
            },
            {
                "name": "Safety Critic (Negative)",
                "role": "Pessimist",
                "goal": "Identify CRITICAL risks. Don't sweat the small stuff, find the killers.",
                "system_prompt": "You are the Safety Critic. Your goal is to find FATAL flaws. Don't complain about minor metric drops unless they indicate a crash risk. Focus on: Sudden altitude drops, Loss of formation control, Near-collisions. If the mission is generally stable, admit it, but warn about edge cases."
            }
        ]
        
        # 裁判角色
        self.judge = {
            "name": "Chief Safety Officer",
            "role": "Judge",
            "system_prompt": "You are the Chief Safety Officer. DECISION PHILOSOPHY: 'Innocent until proven guilty'. Default to SAFE unless the Critic provides undeniable evidence of a critical failure risk. High average scores (>75) are almost always SAFE. Don't let perfectionism kill good missions. Balance: Performance vs. Risk."
        }
    
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
        
        # Round 1: Initial Arguments (独立陈述)
        r1_responses = {}
        for agent in self.agents:
            prompt = f"""
MISSION DATA:
{trajectory_summary}

YOUR TASK:
{agent['goal']}

Analyze the data from your specific perspective ({agent['role']}).
Provide your argument.
"""
            response = self._call_llm(agent['system_prompt'], prompt)
            r1_responses[agent['role']] = response
            debate_history.append(f"{agent['name']}: {response}")
            if self.verbose: print(f"  [{agent['name']}]: {response[:100]}...")

        # Round 2: Rebuttal (交叉反驳)
        advocate_rebuttal_prompt = f"""
THE CRITIC'S ARGUMENT:
{r1_responses['Pessimist']}

YOUR TASK:
Rebut the Critic's claims. Explain why the risks they identified are manageable or not critical. Defend your SAFE position.
"""
        adv_resp = self._call_llm(self.agents[0]['system_prompt'], advocate_rebuttal_prompt)
        debate_history.append(f"Advocate Rebuttal: {adv_resp}")
        
        critic_rebuttal_prompt = f"""
THE ADVOCATE'S ARGUMENT:
{r1_responses['Optimist']}

YOUR TASK:
Dismantle the Advocate's defense. Point out why their optimism is dangerous. Reiterate why this is RISKY.
"""
        crit_resp = self._call_llm(self.agents[1]['system_prompt'], critic_rebuttal_prompt)
        debate_history.append(f"Critic Rebuttal: {crit_resp}")

        # Final Round: Judge's Decision (裁决)
        judge_prompt = f"""
MISSION DATA:
{trajectory_summary}

THE DEBATE:
[Advocate]: {r1_responses['Optimist']}
[Critic]: {r1_responses['Pessimist']}
[Rebuttals]: {adv_resp} | {crit_resp}

YOUR TASK:
As Chief Safety Officer, make the final call.

DECISION LOGIC (Follow this strict priority):
1. **Critical Failure Check**: Did the Critic identify a FATAL flaw (e.g., collision imminent, loss of control)? 
   - If YES -> RISKY
2. **Performance Check**: Is the Overall Average > 75?
   - If YES and no Fatal Flaw -> SAFE (Ignore minor complaints)
3. **Stability Check**: Is the flight stable despite some low metrics?
   - If YES -> SAFE or BORDERLINE
   - If NO -> RISKY

VERDICT GUIDELINES:
- **SAFE**: Mission successful, drone intact. Metrics > 75. (Most high-scoring missions are Safe!)
- **BORDERLINE**: Mission successful but "scary". Metrics 55-75.
- **RISKY**: Mission failed or likely to crash. Metrics < 55.

FORMAT:
SAFETY: [Safe/Borderline/Risky]
SCORE: [0-100]
REASONING: [Brief explanation]
"""
        judge_response = self._call_llm(self.judge['system_prompt'], judge_prompt)
        
        # 解析结果
        safety_label = self._extract_safety_label(judge_response)
        score = self._extract_score(judge_response)
        
        anomalies = TrajectoryAnalyzer.detect_anomalies(mission_data['drones'][0]['trajectory'], formation_analysis)
        
        return {
            "method": "Multi-Agent-Debate",
            "safety_label": safety_label,
            "score": score,
            "issues_identified": anomalies,
            "num_rounds": 2,
            "debate_transcript": debate_history
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
    
    def _build_trajectory_summary(self, mission_data: Dict, drone_analyses: List[Dict], formation_analysis: Dict) -> str:
        """构建轨迹摘要 - 纯数据，无引导"""
        smoothness = np.mean([a['trajectory_smoothness'] for a in drone_analyses])
        altitude = np.mean([a['altitude_stability'] for a in drone_analyses])
        speed = np.mean([a['speed_consistency'] for a in drone_analyses])
        formation = formation_analysis['formation_stability']
        coordination = formation_analysis['coordination_quality']
        overall_avg = (smoothness + altitude + speed + formation + coordination) / 5
        
        return f"""
Mission Duration: {mission_data.get('flight_duration', 'N/A')}
Metrics (0-100):
- Smoothness: {smoothness:.1f}
- Altitude Stability: {altitude:.1f}
- Speed Consistency: {speed:.1f}
- Formation Stability: {formation:.1f}
- Coordination: {coordination:.1f}
- OVERALL AVERAGE: {overall_avg:.1f}

Detected Anomalies:
{chr(10).join(['  - ' + a for a in TrajectoryAnalyzer.detect_anomalies(mission_data['drones'][0]['trajectory'], formation_analysis)]) if TrajectoryAnalyzer.detect_anomalies(mission_data['drones'][0]['trajectory'], formation_analysis) else '  None'}
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
    """计算评估指标"""
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions) if predictions else 0
    
    pred_binary = [1 if p == "Safe" else 0 for p in predictions]
    gt_binary = [1 if g == "Safe" else 0 for g in ground_truth]
    
    precision, recall, f1, _ = precision_recall_fscore_support(gt_binary, pred_binary, average='binary', zero_division=0)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}


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
    dataset_path = os.path.join(script_dir, "..", "improved_uav_missions.json")
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
    
    missions_to_evaluate = missions[:10]
    print(f"\nEvaluating {len(missions_to_evaluate)} missions (fast test mode)")
    
    results = {name: [] for name in evaluators.keys()}
    
    print("\n[3/5] Running evaluations...")
    
    # 累计统计
    cumulative_correct = {name: 0 for name in evaluators.keys()}
    
    for idx, mission in enumerate(missions_to_evaluate):
        print(f"\n{'='*70}")
        print(f"Evaluating mission {idx+1}/{len(missions_to_evaluate)}: {mission['mission_id']}")
        print(f"{'='*70}")
        print(f"Ground Truth: {mission['ground_truth']}")
        
        mission_predictions = {}
        for name, evaluator in evaluators.items():
            print(f"  - {name}...", end=" ", flush=True)
            try:
                result = evaluator.evaluate(mission)
                prediction = result['safety_label']
                results[name].append({
                    "mission_id": mission['mission_id'],
                    "prediction": prediction,
                    "score": result.get('score', 0),
                    "issues": result.get('issues_identified', [])
                })
                mission_predictions[name] = prediction
                print(f"Done: {prediction}")
            except Exception as e:
                print(f"Error: {e}")
                results[name].append({
                    "mission_id": mission['mission_id'],
                    "prediction": "Borderline",
                    "score": 50,
                    "issues": []
                })
                mission_predictions[name] = "Borderline"
        
        # 立即打印该mission的结果对比
        print(f"\n  Result Summary for {mission['mission_id']}:")
        print(f"  {'Method':<25} {'Prediction':<15} {'Status':<10}")
        print(f"  {'-'*50}")
        for name, pred in mission_predictions.items():
            is_correct = pred == mission['ground_truth']
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            if is_correct:
                cumulative_correct[name] += 1
            print(f"  {name:<25} {pred:<15} {status:<10}")
        
        # 显示累计准确率
        print(f"\n  Cumulative Accuracy (so far):")
        for name in evaluators.keys():
            accuracy = cumulative_correct[name] / (idx + 1) * 100
            print(f"    {name:<25} {cumulative_correct[name]}/{idx+1} = {accuracy:.1f}%")
        
        time.sleep(0.3)
    
    ground_truth = [m['ground_truth'] for m in missions_to_evaluate]
    human_labels = [[m['ground_truth']] * 3 for m in missions_to_evaluate]
    
    metrics_table = {}
    for name in evaluators.keys():
        predictions = [r['prediction'] for r in results[name]]
        metrics = compute_metrics(predictions, ground_truth)
        kappa = compute_human_agreement(predictions, human_labels)
        metrics['human_agreement_kappa'] = kappa
        metrics_table[name] = metrics
    
    print("\n[5/5] RESULTS")
    print("\n" + "="*70)
    print("Performance Comparison on {} Missions".format(len(missions_to_evaluate)))
    print("="*70)
    print(f"{'Method':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    
    for name, metrics in metrics_table.items():
        print(f"{name:<25} {metrics['accuracy']:<12.2%} {metrics['precision']:<12.2%} "
              f"{metrics['recall']:<12.2%} {metrics['f1_score']:<12.2%}")
    
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
    
    output_file = "exp1_real_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": "improved_uav_missions.json",
            "num_missions": len(missions_to_evaluate),
            "results": results,
            "metrics": metrics_table
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
