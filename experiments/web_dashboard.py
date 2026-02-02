"""
实验实时可视化Web服务
基于Flask + WebSocket实现实时数据推送
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import json
from typing import Dict, List
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'uav-safety-dashboard-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局状态
experiment_state = {
    'is_running': False,
    'current_mission': 0,
    'total_missions': 0,
    'methods': [],
    'safety_history': {},
    'efficiency_history': {},
    'cumulative_correct': {},
    'cumulative_eff_correct': {},
    'final_metrics': None
}


@app.route('/')
def index():
    """主页面"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """获取实验状态"""
    return jsonify(experiment_state)


@app.route('/api/reset')
def reset_experiment():
    """重置实验状态"""
    global experiment_state
    experiment_state = {
        'is_running': False,
        'current_mission': 0,
        'total_missions': 0,
        'methods': [],
        'safety_history': {},
        'efficiency_history': {},
        'cumulative_correct': {},
        'cumulative_eff_correct': {},
        'final_metrics': None
    }
    socketio.emit('reset')
    return jsonify({'status': 'reset'})


@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    print(f"Client connected: {request.sid}")
    # 不在这里emit，等待客户端请求状态
    # emit('connected', {'data': 'Connected to UAV Safety Dashboard'})
    # emit('state_update', experiment_state)


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开"""
    print(f"Client disconnected: {request.sid}")


def initialize_experiment(methods: List[str], total_missions: int):
    """初始化实验"""
    global experiment_state
    experiment_state['methods'] = methods
    experiment_state['total_missions'] = total_missions
    experiment_state['is_running'] = True
    experiment_state['current_mission'] = 0
    
    for method in methods:
        experiment_state['safety_history'][method] = []
        experiment_state['efficiency_history'][method] = []
        experiment_state['cumulative_correct'][method] = 0
        experiment_state['cumulative_eff_correct'][method] = 0
    
    print(f"[Dashboard] Initializing experiment: methods={methods}, total={total_missions}")
    socketio.emit('experiment_initialized', {
        'methods': methods,
        'total_missions': total_missions
    })


def update_mission(mission_idx: int, safety_correct: Dict[str, int], 
                   eff_correct: Dict[str, int]):
    """更新任务数据"""
    global experiment_state
    experiment_state['current_mission'] = mission_idx + 1
    experiment_state['cumulative_correct'] = safety_correct
    experiment_state['cumulative_eff_correct'] = eff_correct
    
    # 计算准确率
    for method in experiment_state['methods']:
        safety_acc = (safety_correct[method] / (mission_idx + 1)) * 100
        eff_acc = (eff_correct[method] / (mission_idx + 1)) * 100
        
        experiment_state['safety_history'][method].append(safety_acc)
        experiment_state['efficiency_history'][method].append(eff_acc)
    
    # 推送更新到所有客户端
    print(f"[Dashboard] Updating mission {mission_idx + 1}")
    socketio.emit('mission_update', {
        'mission_idx': mission_idx,
        'safety_history': experiment_state['safety_history'],
        'efficiency_history': experiment_state['efficiency_history'],
        'cumulative_correct': safety_correct,
        'cumulative_eff_correct': eff_correct
    })


def finalize_experiment(metrics_table: Dict):
    """完成实验"""
    global experiment_state
    experiment_state['is_running'] = False
    experiment_state['final_metrics'] = metrics_table
    
    print(f"[Dashboard] Finalizing experiment")
    socketio.emit('experiment_completed', {
        'final_metrics': metrics_table,
        'total_missions': experiment_state['total_missions']
    })


def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """启动Web服务"""
    print(f"[Dashboard] Starting UAV Safety Dashboard...")
    print(f"   Access at: http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_dashboard(debug=True)
