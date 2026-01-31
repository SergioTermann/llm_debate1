"""
实验监控和可视化工具 - 类似TensorBoard的实时曲线显示
"""

import json
import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from datetime import datetime
import numpy as np


class ExperimentMonitor:
    """实验监控器 - 实时显示评估结果"""
    
    def __init__(self, results_file: str = "exp1_real_results.json"):
        self.results_file = results_file
        self.data = {
            'timestamps': [],
            'methods': [],
            'metrics': {
                'accuracy': {},
                'precision': {},
                'recall': {},
                'f1_score': {},
                'efficiency_accuracy': {}
            },
            'per_class_metrics': {}
        }
        self.current_mission = 0
        self.total_missions = 0
        
    def load_results(self):
        """加载最新的结果文件"""
        if not os.path.exists(self.results_file):
            return False
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        self.total_missions = results_data.get('num_missions', 0)
        self.data['methods'] = list(results_data.get('results', {}).keys())
        
        # 加载每个方法的结果
        for method in self.data['methods']:
            method_results = results_data['results'][method]
            
            # 计算累积指标
            for i in range(len(method_results)):
                self._update_cumulative_metrics(method, i, method_results[i])
        
        return True
    
    def _update_cumulative_metrics(self, method: str, idx: int, result: dict):
        """更新累积指标"""
        if method not in self.data['metrics']['accuracy']:
            self.data['metrics']['accuracy'][method] = []
            self.data['metrics']['precision'][method] = []
            self.data['metrics']['recall'][method] = []
            self.data['metrics']['f1_score'][method] = []
            self.data['metrics']['efficiency_accuracy'][method] = []
        
        # 这里需要根据ground truth计算累积指标
        # 暂时存储预测结果，后续统一计算
        pass
    
    def create_dashboard(self, save_path: str = "experiment_dashboard.png"):
        """创建实验仪表板"""
        if not self.load_results():
            print("No results file found!")
            return
        
        num_methods = len(self.data['methods'])
        
        # 创建大图
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 累积准确率曲线
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_cumulative_accuracy(ax1)
        ax1.set_title('Cumulative Safety Accuracy', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Mission Number')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 累积效率准确率曲线
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_cumulative_efficiency(ax2)
        ax2.set_title('Cumulative Efficiency Accuracy', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Mission Number')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        # 3. 每个类别的F1分数对比
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_per_class_f1(ax3)
        ax3.set_title('Per-Class F1 Score Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylabel('F1 Score')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. 方法性能雷达图
        ax4 = fig.add_subplot(gs[2, 0], projection='polar')
        self._plot_radar_chart(ax4)
        ax4.set_title('Overall Performance (Radar)', fontsize=12, fontweight='bold', pad=20)
        
        # 5. 性能对比条形图
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_bar_comparison(ax5)
        ax5.set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Score (%)')
        ax5.legend(loc='lower right')
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to: {save_path}")
        plt.show()
    
    def _plot_cumulative_accuracy(self, ax):
        """绘制累积准确率曲线"""
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.data['methods'])))
        
        for i, method in enumerate(self.data['methods']):
            # 模拟累积准确率数据（实际应该从结果文件计算）
            missions = np.arange(1, self.total_missions + 1)
            
            # 根据方法特性生成模拟曲线
            if 'Single-Metric' in method:
                # 传统方法：准确率较低
                accuracy = np.linspace(0, 50, self.total_missions)
            elif 'Fixed-Weight' in method:
                # 固定权重：中等准确率
                accuracy = np.linspace(0, 60, self.total_missions)
            elif 'Single-Agent-LLM' in method:
                # 单智能体LLM：较高准确率
                accuracy = np.linspace(0, 70, self.total_missions)
            else:  # Multi-Agent-Debate
                # 多智能体辩论：中等准确率
                accuracy = np.linspace(0, 60, self.total_missions)
            
            ax.plot(missions, accuracy, marker='o', linewidth=2, 
                   markersize=4, color=colors[i], label=method, alpha=0.8)
    
    def _plot_cumulative_efficiency(self, ax):
        """绘制累积效率准确率曲线"""
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.data['methods'])))
        
        for i, method in enumerate(self.data['methods']):
            missions = np.arange(1, self.total_missions + 1)
            
            # 根据方法特性生成模拟曲线
            if 'Single-Metric' in method or 'Fixed-Weight' in method:
                # 传统方法：效率准确率高
                efficiency = np.linspace(100, 100, self.total_missions)
            elif 'Single-Agent-LLM' in method:
                # 单智能体LLM：中等效率准确率
                efficiency = np.linspace(60, 70, self.total_missions)
            else:  # Multi-Agent-Debate
                # 多智能体辩论：高效率准确率
                efficiency = np.linspace(100, 100, self.total_missions)
            
            ax.plot(missions, efficiency, marker='s', linewidth=2, 
                   markersize=4, color=colors[i], label=method, alpha=0.8)
    
    def _plot_per_class_f1(self, ax):
        """绘制每个类别的F1分数对比"""
        methods = self.data['methods']
        classes = ['Safe', 'Borderline', 'Risky']
        
        x = np.arange(len(classes))
        width = 0.2
        
        # 模拟数据（实际应该从结果文件读取）
        for i, method in enumerate(methods):
            if 'Single-Metric' in method:
                f1_scores = [0.0, 0.0, 0.0]
            elif 'Fixed-Weight' in method:
                f1_scores = [0.0, 0.0, 0.0]
            elif 'Single-Agent-LLM' in method:
                f1_scores = [0.0, 0.0, 0.0]
            else:  # Multi-Agent-Debate
                f1_scores = [0.0, 0.0, 0.0]
            
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, f1_scores, width, label=method, alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1)
    
    def _plot_radar_chart(self, ax):
        """绘制雷达图"""
        methods = self.data['methods']
        categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'Efficiency']
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, method in enumerate(methods):
            # 模拟数据
            if 'Single-Metric' in method:
                values = [50, 0, 0, 0, 100]
            elif 'Fixed-Weight' in method:
                values = [60, 0, 0, 0, 100]
            elif 'Single-Agent-LLM' in method:
                values = [70, 0, 0, 0, 70]
            else:  # Multi-Agent-Debate
                values = [60, 0, 0, 0, 100]
            
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
    
    def _plot_bar_comparison(self, ax):
        """绘制性能对比条形图"""
        methods = self.data['methods']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Efficiency']
        
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, method in enumerate(methods):
            # 模拟数据
            if 'Single-Metric' in method:
                values = [50, 0, 0, 0, 100]
            elif 'Fixed-Weight' in method:
                values = [60, 0, 0, 0, 100]
            elif 'Single-Agent-LLM' in method:
                values = [70, 0, 0, 0, 70]
            else:  # Multi-Agent-Debate
                values = [60, 0, 0, 0, 100]
            
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, values, width, label=method, alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylim(0, 100)


class RealTimeMonitor:
    """实时监控器 - 在实验运行时实时更新"""
    
    def __init__(self, results_file: str = "exp1_real_results.json"):
        self.results_file = results_file
        self.mission_results = {
            'Single-Metric': [],
            'Fixed-Weight': [],
            'Single-Agent-LLM': [],
            'Multi-Agent-Debate': []
        }
        self.ground_truth = []
        self.current_mission = 0
        
        # 创建实时图表
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 10))
        self.fig.suptitle('Real-Time Experiment Monitor', fontsize=16, fontweight='bold')
        
        # 初始化数据线
        self.lines = {}
        self.scatters = {}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (ax_row, ax_col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            ax = self.axes[ax_row, ax_col]
            method = ['Single-Metric', 'Fixed-Weight', 'Single-Agent-LLM', 'Multi-Agent-Debate'][i]
            color = colors[i]
            
            # 准确率线
            line, = ax.plot([], [], 'o-', linewidth=2, color=color, label=f'{method} Acc', markersize=6)
            self.lines[f'{method}_acc'] = line
            
            # 效率准确率线
            line, = ax.plot([], [], 's-', linewidth=2, color=color, linestyle='--', label=f'{method} Eff', markersize=6)
            self.lines[f'{method}_eff'] = line
            
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)
            
            if ax_col == 0:
                ax.set_ylabel('Accuracy (%)', fontsize=10)
            if ax_row == 1:
                ax.set_xlabel('Mission Number', fontsize=10)
        
        plt.tight_layout()
        
    def update(self, mission_id: str, results: dict, ground_truth: str):
        """更新实时数据"""
        self.current_mission += 1
        self.ground_truth.append(ground_truth)
        
        for method, result in results.items():
            self.mission_results[method].append(result['prediction'])
        
        # 更新图表
        self._update_charts()
        
        # 保存到文件
        self._save_to_file()
        
        print(f"\n[Monitor] Updated: Mission {self.current_mission}/{len(self.mission_results['Single-Metric'])}")
        print(f"  Ground Truth: {ground_truth}")
        for method, result in results.items():
            print(f"  {method}: {result['prediction']}")
    
    def _update_charts(self):
        """更新图表"""
        missions = np.arange(1, self.current_mission + 1)
        
        for method in ['Single-Metric', 'Fixed-Weight', 'Single-Agent-LLM', 'Multi-Agent-Debate']:
            predictions = self.mission_results[method]
            
            # 计算累积准确率
            correct = sum(1 for p, g in zip(predictions, self.ground_truth) if p == g)
            accuracy = correct / len(predictions) * 100 if predictions else 0
            
            # 计算累积效率准确率
            eff_predictions = [r.get('efficiency_prediction', 'Medium') for r in self.mission_results[method]]
            # 简化：假设效率准确率为100%（实际需要根据ground truth计算）
            eff_accuracy = 100
            
            # 更新准确率线
            self.lines[f'{method}_acc'].set_data(missions, [accuracy] * len(missions))
            self.lines[f'{method}_eff'].set_data(missions, [eff_accuracy] * len(missions))
        
        plt.draw()
        plt.pause(0.01)
    
    def _save_to_file(self):
        """保存到结果文件"""
        results_data = {
            'dataset': 'complex_uav_missions.json',
            'num_missions': self.current_mission,
            'results': {
                method: [{'mission_id': f'MISSION-{i+1:04d}', 'prediction': pred} 
                         for i, pred in enumerate(preds)]
                for method, preds in self.mission_results.items()
            },
            'ground_truth': self.ground_truth
        }
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    def close(self):
        """关闭监控器"""
        plt.ioff()
        plt.close(self.fig)
        print("\n[Monitor] Closed")


def create_comparison_dashboard(results_file: str = "exp1_real_results.json"):
    """创建对比仪表板"""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    methods = list(results_data.get('results', {}).keys())
    metrics_data = results_data.get('metrics', {})
    
    # 创建大图
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. 准确率对比
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [metrics_data[m]['accuracy'] for m in methods]
    bars1 = ax1.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Safety Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1%}', 
                ha='center', va='bottom', fontsize=10)
    
    # 2. 效率准确率对比
    ax2 = fig.add_subplot(gs[0, 1])
    eff_accs = [metrics_data[m]['efficiency_accuracy'] for m in methods]
    bars2 = ax2.bar(methods, eff_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Efficiency Accuracy Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1%}', 
                ha='center', va='bottom', fontsize=10)
    
    # 3. 每个类别的详细指标
    ax3 = fig.add_subplot(gs[1, :])
    classes = ['Safe', 'Borderline', 'Risky']
    x = np.arange(len(classes))
    width = 0.2
    
    for i, method in enumerate(methods):
        per_class = metrics_data[method].get('per_class', {})
        f1_scores = [per_class.get(c, {}).get('f1', 0) for c in classes]
        offset = (i - 1.5) * width
        bars = ax3.bar(x + offset, f1_scores, width, label=method, alpha=0.8)
    
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Per-Class F1 Score', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1)
    
    # 4. 雷达图
    ax4 = fig.add_subplot(gs[2, 0], projection='polar')
    categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'Efficiency']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, method in enumerate(methods):
        metrics = metrics_data[method]
        values = [
            metrics['accuracy'] * 100,
            metrics['precision'] * 100,
            metrics['recall'] * 100,
            metrics['f1_score'] * 100,
            metrics['efficiency_accuracy'] * 100
        ]
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
        ax4.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 100)
    ax4.set_yticks([0, 20, 40, 60, 80, 100])
    ax4.set_title('Overall Performance (Radar)', fontsize=12, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.3, 1.1))
    
    # 5. 方法排名表
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # 创建排名数据
    ranking_data = []
    for method in methods:
        metrics = metrics_data[method]
        ranking_data.append({
            'Method': method,
            'Acc': f"{metrics['accuracy']:.1%}",
            'Prec': f"{metrics['precision']:.1%}",
            'Rec': f"{metrics['recall']:.1%}",
            'F1': f"{metrics['f1_score']:.1%}",
            'Eff': f"{metrics['efficiency_accuracy']:.1%}"
        })
    
    # 按准确率排序
    ranking_data.sort(key=lambda x: float(x['Acc'].rstrip('%')), reverse=True)
    
    # 绘制表格
    table_data = [[d['Method'], d['Acc'], d['Prec'], d['Rec'], d['F1'], d['Eff']] 
                  for d in ranking_data]
    table = ax5.table(cellText=table_data, cellLoc='center',
                     colLabels=['Method', 'Accuracy', 'Precision', 'Recall', 'F1', 'Efficiency'],
                     bbox=[0, 0, 1, 1], cellLoc='center')
    table.auto_set_font_size(False)
    table.scale(1, 2)
    ax5.set_title('Method Ranking', fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig('experiment_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"Dashboard saved to: experiment_dashboard.png")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment Monitor and Visualization')
    parser.add_argument('--mode', type=str, default='dashboard',
                       choices=['dashboard', 'monitor'],
                       help='Mode: dashboard (static) or monitor (real-time)')
    parser.add_argument('--results', type=str, default='exp1_real_results.json',
                       help='Results file path')
    
    args = parser.parse_args()
    
    if args.mode == 'dashboard':
        create_comparison_dashboard(args.results)
    elif args.mode == 'monitor':
        print("Real-time monitor mode requires integration with evaluation script")
        print("Use dashboard mode to visualize results after evaluation")
