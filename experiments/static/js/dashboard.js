// UAV Safety Dashboard - Frontend JavaScript

const socket = io('http://localhost:5000', {
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: 10
});

let safetyChart, efficiencyChart, barChart;
let precisionChart, recallChart, f1Chart, kappaChart;
let currentMethods = [];
let missionLabels = [];

const colors = {
    'Single-Metric': '#1f77b4',
    'Fixed-Weight': '#ff7f0e',
    'Single-Agent-LLM': '#2ca02c',
    'Multi-Agent-Debate': '#d62728'
};

const classColors = {
    'Safe': '#2ecc71',
    'Borderline': '#f39c12',
    'Risky': '#e74c3c'
};

// Socket event handlers
socket.on('connect', () => {
    console.log('Connected to dashboard server');
    document.getElementById('connection-status').textContent = 'Connected';
    document.getElementById('connection-status').className = 'status-badge connected';
});

socket.on('disconnect', (reason) => {
    console.log('Disconnected from dashboard server:', reason);
    document.getElementById('connection-status').textContent = 'Disconnected';
    document.getElementById('connection-status').className = 'status-badge disconnected';
});

socket.on('connect_error', (error) => {
    console.error('Connection error:', error);
    document.getElementById('connection-status').textContent = 'Connection Error';
    document.getElementById('connection-status').className = 'status-badge disconnected';
});

socket.on('error', (error) => {
    console.error('Socket error:', error);
});

// Initialize charts
function initCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 300
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 105,
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                }
            },
            x: {
                grid: {
                    display: false
                }
            }
        },
        plugins: {
            legend: {
                position: 'bottom'
            }
        }
    };

    // Safety Accuracy Chart
    safetyChart = new Chart(document.getElementById('safetyChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                title: {
                    display: true,
                    text: 'Safety Accuracy (%)'
                }
            }
        }
    });

    // Efficiency Accuracy Chart
    efficiencyChart = new Chart(document.getElementById('efficiencyChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                title: {
                    display: true,
                    text: 'Efficiency Accuracy (%)'
                }
            }
        }
    });

    // Bar Chart for Current Accuracy
    barChart = new Chart(document.getElementById('barChart'), {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Current Safety Accuracy (%)',
                data: [],
                backgroundColor: []
            }]
        },
        options: {
            ...chartOptions,
            indexAxis: 'y',
            plugins: {
                ...chartOptions.plugins,
                legend: {
                    display: false
                }
            }
        }
    });

    // Final Metrics Charts
    initFinalMetricsCharts();
}

function initFinalMetricsCharts() {
    const barOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                max: 1.1
            }
        },
        plugins: {
            legend: {
                position: 'bottom'
            }
        }
    };

    precisionChart = new Chart(document.getElementById('precisionChart'), {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: barOptions
    });

    recallChart = new Chart(document.getElementById('recallChart'), {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: barOptions
    });

    f1Chart = new Chart(document.getElementById('f1Chart'), {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: barOptions
    });

    kappaChart = new Chart(document.getElementById('kappaChart'), {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: {
            ...barOptions,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Update charts with new data
function updateCharts(safetyHistory, efficiencyHistory, methods) {
    missionLabels = Array.from({ length: Object.values(safetyHistory)[0]?.length || 0 }, (_, i) => i + 1);

    // Update Safety Chart
    safetyChart.data.labels = missionLabels;
    safetyChart.data.datasets = methods.map(method => ({
        label: method,
        data: safetyHistory[method],
        borderColor: colors[method] || '#999',
        backgroundColor: colors[method] || '#999',
        tension: 0.3,
        pointRadius: 4,
        pointHoverRadius: 6
    }));
    safetyChart.update();

    // Update Efficiency Chart
    efficiencyChart.data.labels = missionLabels;
    efficiencyChart.data.datasets = methods.map(method => ({
        label: method,
        data: efficiencyHistory[method],
        borderColor: colors[method] || '#999',
        backgroundColor: colors[method] || '#999',
        tension: 0.3,
        pointRadius: 4,
        pointHoverRadius: 6
    }));
    efficiencyChart.update();

    // Update Bar Chart
    const currentSafetyAcc = methods.map(method => 
        safetyHistory[method][safetyHistory[method].length - 1] || 0
    );
    barChart.data.labels = methods;
    barChart.data.datasets[0].data = currentSafetyAcc;
    barChart.data.datasets[0].backgroundColor = methods.map(m => colors[m] || '#999');
    barChart.update();
}

// Update results table
function updateResultsTable(cumulativeCorrect, cumulativeEffCorrect, currentMission, methods) {
    const tbody = document.getElementById('resultsBody');
    tbody.innerHTML = '';

    methods.forEach(method => {
        const sCorrect = cumulativeCorrect[method];
        const sAcc = ((sCorrect / (currentMission + 1)) * 100).toFixed(1);
        const eCorrect = cumulativeEffCorrect[method];
        const eAcc = ((eCorrect / (currentMission + 1)) * 100).toFixed(1);

        const row = document.createElement('tr');
        row.innerHTML = `
            <td style="font-weight: 600; color: ${colors[method]}">${method}</td>
            <td>${sCorrect}/${currentMission + 1}</td>
            <td style="font-weight: 600">${sAcc}%</td>
            <td>${eCorrect}/${currentMission + 1}</td>
            <td style="font-weight: 600">${eAcc}%</td>
        `;
        tbody.appendChild(row);
    });
}

// Update info panel
function updateInfoPanel(currentMission, totalMissions) {
    document.getElementById('currentMission').textContent = currentMission;
    document.getElementById('totalMissions').textContent = totalMissions;
    const progress = totalMissions > 0 ? ((currentMission / totalMissions) * 100).toFixed(1) : 0;
    document.getElementById('progress').textContent = `${progress}%`;
}

// Update final metrics
function updateFinalMetrics(metricsTable, methods) {
    const finalMetricsDiv = document.getElementById('finalMetrics');
    finalMetricsDiv.style.display = 'block';

    const classes = ['Safe', 'Borderline', 'Risky'];

    // Precision Chart
    precisionChart.data.labels = methods;
    precisionChart.data.datasets = classes.map(cls => ({
        label: cls,
        data: methods.map(m => metricsTable[m]?.per_class?.[cls]?.precision || 0),
        backgroundColor: classColors[cls]
    }));
    precisionChart.update();

    // Recall Chart
    recallChart.data.labels = methods;
    recallChart.data.datasets = classes.map(cls => ({
        label: cls,
        data: methods.map(m => metricsTable[m]?.per_class?.[cls]?.recall || 0),
        backgroundColor: classColors[cls]
    }));
    recallChart.update();

    // F1 Chart
    f1Chart.data.labels = methods;
    f1Chart.data.datasets = classes.map(cls => ({
        label: cls,
        data: methods.map(m => metricsTable[m]?.per_class?.[cls]?.f1 || 0),
        backgroundColor: classColors[cls]
    }));
    f1Chart.update();

    // Kappa Chart
    kappaChart.data.labels = methods;
    kappaChart.data.datasets = [{
        data: methods.map(m => metricsTable[m]?.human_agreement_kappa || 0),
        backgroundColor: methods.map(m => colors[m] || '#999'),
        borderColor: '#333',
        borderWidth: 1
    }];
    kappaChart.update();

    // Scroll to final metrics
    finalMetricsDiv.scrollIntoView({ behavior: 'smooth' });
}

// Socket event handlers for experiment updates
socket.on('experiment_initialized', (data) => {
    console.log('Experiment initialized:', data);
    currentMethods = data.methods;
    document.getElementById('totalMissions').textContent = data.total_missions;
    document.getElementById('experiment-status').textContent = 'Running';
    document.getElementById('experiment-status').className = 'status-badge running';
});

socket.on('mission_update', (data) => {
    console.log('Mission update:', data);
    const { mission_idx, safety_history, efficiency_history, cumulative_correct, cumulative_eff_correct } = data;
    
    updateCharts(safety_history, efficiency_history, currentMethods);
    updateResultsTable(cumulative_correct, cumulative_eff_correct, mission_idx, currentMethods);
    updateInfoPanel(mission_idx + 1, document.getElementById('totalMissions').textContent);
});

socket.on('experiment_completed', (data) => {
    console.log('Experiment completed:', data);
    document.getElementById('experiment-status').textContent = 'Completed';
    document.getElementById('experiment-status').className = 'status-badge completed';
    updateFinalMetrics(data.final_metrics, currentMethods);
});

socket.on('reset', () => {
    console.log('Dashboard reset');
    location.reload();
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
});
