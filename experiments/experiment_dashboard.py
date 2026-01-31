"""
实验实时可视化面板 - 类似TensorBoard
支持实时显示准确率曲线、多维度指标对比
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Dict, List
import time


class ExperimentDashboard:
    """实验实时可视化面板"""
    
    def __init__(self, methods: List[str]):
        self.methods = methods
        self.num_methods = len(methods)
        
        # 数据存储
        self.safety_history = {name: [] for name in methods}
        self.efficiency_history = {name: [] for name in methods}
        self.mission_indices = []
        
        # 颜色配置
        self.colors = {
            'Single-Metric': '#1f77b4',
            'Fixed-Weight': '#ff7f0e',
            'Single-Agent-LLM': '#2ca02c',
            'Multi-Agent-Debate': '#d62728'
        }
        
        # 创建图形
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('UAV Safety Evaluation Dashboard - Real-time Monitoring', 
                         fontsize=16, fontweight='bold')
        
        # 使用GridSpec布局
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # 1. 安全性准确率曲线（占据主要位置）
        self.ax_safety = self.fig.add_subplot(gs[0:2, 0:2])
        self.setup_safety_plot()
        
        # 2. 效率准确率曲线
        self.ax_efficiency = self.fig.add_subplot(gs[0, 2])
        self.setup_efficiency_plot()
        
        # 3. 当前准确率对比柱状图
        self.ax_bar = self.fig.add_subplot(gs[1, 2])
        self.setup_bar_plot()
        
        # 4. 累计正确数表格
        self.ax_table = self.fig.add_subplot(gs[2, :])
        self.setup_table()
        
        # 初始化线条
        self.safety_lines = {}
        self.efficiency_lines = {}
        
        for method in methods:
            color = self.colors.get(method, plt.cm.tab10(self.methods.index(method)))
            
            # 安全性准确率线条
            line_safety, = self.ax_safety.plot([], [], marker='o', 
                                              label=method, color=color,
                                              linewidth=2, markersize=4)
            self.safety_lines[method] = line_safety
            
            # 效率准确率线条
            line_eff, = self.ax_efficiency.plot([], [], marker='s',
                                               label=method, color=color,
                                               linewidth=2, markersize=4)
            self.efficiency_lines[method] = line_eff
        
        self.ax_safety.legend(loc='lower right', fontsize=9)
        self.ax_efficiency.legend(loc='lower right', fontsize=8)
        
        plt.ion()
        plt.show(block=False)
    
    def setup_safety_plot(self):
        """设置安全性准确率曲线图"""
        self.ax_safety.set_title('Safety Accuracy Over Missions', fontsize=12, fontweight='bold')
        self.ax_safety.set_xlabel('Mission Number', fontsize=10)
        self.ax_safety.set_ylabel('Accuracy (%)', fontsize=10)
        self.ax_safety.set_ylim(0, 105)
        self.ax_safety.grid(True, alpha=0.3, linestyle='--')
        self.ax_safety.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random Baseline')
    
    def setup_efficiency_plot(self):
        """设置效率准确率曲线图"""
        self.ax_efficiency.set_title('Efficiency Accuracy Over Missions', fontsize=11, fontweight='bold')
        self.ax_efficiency.set_xlabel('Mission', fontsize=9)
        self.ax_efficiency.set_ylabel('Acc (%)', fontsize=9)
        self.ax_efficiency.set_ylim(0, 105)
        self.ax_efficiency.grid(True, alpha=0.3, linestyle='--')
    
    def setup_bar_plot(self):
        """设置柱状图"""
        self.ax_bar.set_title('Current Accuracy Comparison', fontsize=11, fontweight='bold')
        self.ax_bar.set_ylabel('Accuracy (%)', fontsize=9)
        self.ax_bar.set_ylim(0, 105)
        self.ax_bar.grid(True, alpha=0.3, axis='y')
    
    def setup_table(self):
        """设置数据表格"""
        self.ax_table.axis('off')
        self.ax_table.set_title('Cumulative Results Summary', fontsize=12, fontweight='bold', pad=10)
    
    def update(self, mission_idx: int, safety_correct: Dict[str, int], 
               eff_correct: Dict[str, int]):
        """更新图表数据"""
        self.mission_indices.append(mission_idx + 1)
        
        # 计算准确率
        for method in self.methods:
            safety_acc = (safety_correct[method] / (mission_idx + 1)) * 100
            eff_acc = (eff_correct[method] / (mission_idx + 1)) * 100
            
            self.safety_history[method].append(safety_acc)
            self.efficiency_history[method].append(eff_acc)
            
            # 更新线条
            self.safety_lines[method].set_data(self.mission_indices, self.safety_history[method])
            self.efficiency_lines[method].set_data(self.mission_indices, self.efficiency_history[method])
        
        # 更新X轴范围
        self.ax_safety.set_xlim(0.5, max(10, mission_idx + 1.5))
        self.ax_efficiency.set_xlim(0.5, max(10, mission_idx + 1.5))
        
        # 更新柱状图
        self.ax_bar.clear()
        self.setup_bar_plot()
        
        current_safety_acc = [self.safety_history[method][-1] for method in self.methods]
        colors = [self.colors.get(method, plt.cm.tab10(i)) for i, method in enumerate(self.methods)]
        
        bars = self.ax_bar.bar(self.methods, current_safety_acc, color=colors, alpha=0.7, edgecolor='black')
        
        # 在柱子上显示数值
        for bar, acc in zip(bars, current_safety_acc):
            height = bar.get_height()
            self.ax_bar.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        
        self.ax_bar.set_xticklabels(self.methods, rotation=45, ha='right', fontsize=8)
        
        # 更新表格
        self.update_table(mission_idx, safety_correct, eff_correct)
        
        # 刷新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.1)
    
    def update_table(self, mission_idx: int, safety_correct: Dict[str, int], 
                     eff_correct: Dict[str, int]):
        """更新数据表格"""
        self.ax_table.clear()
        self.ax_table.axis('off')
        self.ax_table.set_title(f'Cumulative Results Summary (Mission {mission_idx + 1})', 
                               fontsize=12, fontweight='bold', pad=10)
        
        # 准备表格数据
        table_data = []
        headers = ['Method', 'Safety Correct', 'Safety Acc', 'Eff Correct', 'Eff Acc']
        
        for method in self.methods:
            s_corr = safety_correct[method]
            s_acc = (s_corr / (mission_idx + 1)) * 100
            e_corr = eff_correct[method]
            e_acc = (e_corr / (mission_idx + 1)) * 100
            
            table_data.append([
                method,
                f'{s_corr}/{mission_idx + 1}',
                f'{s_acc:.1f}%',
                f'{e_corr}/{mission_idx + 1}',
                f'{e_acc:.1f}%'
            ])
        
        # 创建表格
        table = self.ax_table.table(cellText=table_data, colLabels=headers,
                                   cellLoc='center', loc='center',
                                   colColours=['#f0f0f0'] * 5)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 设置列宽
        for i in range(5):
            table[(0, i)].set_facecolor('#4a90e2')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    def save_final_results(self, output_path: str = 'experiment_dashboard.png'):
        """保存最终结果图"""
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to: {output_path}")
    
    def close(self):
        """关闭可视化面板"""
        plt.close(self.fig)


class MetricsComparisonPanel:
    """多维度指标对比面板 - 实验结束后显示"""
    
    def __init__(self, methods: List[str]):
        self.methods = methods
        self.colors = {
            'Single-Metric': '#1f77b4',
            'Fixed-Weight': '#ff7f0e',
            'Single-Agent-LLM': '#2ca02c',
            'Multi-Agent-Debate': '#d62728'
        }
        
        self.fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('Comprehensive Metrics Comparison', fontsize=16, fontweight='bold')
        
        self.ax_precision = axes[0, 0]
        self.ax_recall = axes[0, 1]
        self.ax_f1 = axes[1, 0]
        self.ax_kappa = axes[1, 1]
        
        self.setup_axes()
    
    def setup_axes(self):
        """设置各子图"""
        # Precision
        self.ax_precision.set_title('Precision Comparison', fontsize=12, fontweight='bold')
        self.ax_precision.set_ylabel('Precision', fontsize=10)
        self.ax_precision.set_ylim(0, 1.1)
        self.ax_precision.grid(True, alpha=0.3)
        
        # Recall
        self.ax_recall.set_title('Recall Comparison', fontsize=12, fontweight='bold')
        self.ax_recall.set_ylabel('Recall', fontsize=10)
        self.ax_recall.set_ylim(0, 1.1)
        self.ax_recall.grid(True, alpha=0.3)
        
        # F1 Score
        self.ax_f1.set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
        self.ax_f1.set_ylabel('F1 Score', fontsize=10)
        self.ax_f1.set_ylim(0, 1.1)
        self.ax_f1.grid(True, alpha=0.3)
        
        # Kappa
        self.ax_kappa.set_title("Human Agreement (Cohen's κ)", fontsize=12, fontweight='bold')
        self.ax_kappa.set_ylabel("κ Score", fontsize=10)
        self.ax_kappa.set_ylim(0, 1.1)
        self.ax_kappa.grid(True, alpha=0.3)
    
    def plot_metrics(self, metrics_table: Dict[str, Dict]):
        """绘制指标对比图"""
        x = np.arange(len(self.methods))
        width = 0.25
        
        # 为每个类别创建柱状图
        classes = ['Safe', 'Borderline', 'Risky']
        class_colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        for idx, cls in enumerate(classes):
            precisions = [metrics_table[m]['per_class'][cls]['precision'] for m in self.methods]
            recalls = [metrics_table[m]['per_class'][cls]['recall'] for m in self.methods]
            f1s = [metrics_table[m]['per_class'][cls]['f1'] for m in self.methods]
            
            offset = (idx - 1) * width
            
            self.ax_precision.bar(x + offset, precisions, width, label=cls, 
                                 color=class_colors[idx], alpha=0.8)
            self.ax_recall.bar(x + offset, recalls, width, label=cls, 
                              color=class_colors[idx], alpha=0.8)
            self.ax_f1.bar(x + offset, f1s, width, label=cls, 
                          color=class_colors[idx], alpha=0.8)
        
        # 设置X轴标签
        for ax in [self.ax_precision, self.ax_recall, self.ax_f1]:
            ax.set_xticks(x)
            ax.set_xticklabels(self.methods, rotation=15, ha='right', fontsize=9)
            ax.legend(fontsize=8, loc='lower right')
        
        # 绘制Kappa分数
        kappas = [metrics_table[m]['human_agreement_kappa'] for m in self.methods]
        colors = [self.colors.get(m, plt.cm.tab10(i)) for i, m in enumerate(self.methods)]
        bars = self.ax_kappa.bar(x, kappas, color=colors, alpha=0.7, edgecolor='black')
        
        for bar, kappa in zip(bars, kappas):
            height = bar.get_height()
            if not np.isnan(kappa):
                self.ax_kappa.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                 f'{kappa:.3f}', ha='center', va='bottom', fontsize=9)
        
        self.ax_kappa.set_xticks(x)
        self.ax_kappa.set_xticklabels(self.methods, rotation=15, ha='right', fontsize=9)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)
    
    def save(self, output_path: str = 'metrics_comparison.png'):
        """保存指标对比图"""
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {output_path}")
