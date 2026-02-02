# EXPERIMENT 1: REAL EVALUATION RESULTS

## Dataset Information
- Dataset: complex_uav_missions.json
- Total Missions: 10

## Ground Truth Distribution
- Safety: 9 Risky (90%), 1 Safe (10%), 0 Borderline (0%)
- Efficiency: 8 Low (80%), 2 Medium (20%), 0 High (0%)

## Performance Comparison

| Method                | Accuracy | Precision | Recall | F1 Score |
|-----------------------|----------|------------|---------|----------|
| Single-Metric         | 90.00%   | 1.000      | 0.889   | 0.941    |
| Fixed-Weight          | 80.00%   | 0.889      | 0.889   | 0.889    |

## Detailed Results

### Single-Metric

| Mission                               | Prediction | Efficiency | Score  |
|---------------------------------------|------------|-------------|---------|
| COMPLEX_01_MULTI_LAYER_FORMATION        | Risky      | Medium      | 47.22   |
| COMPLEX_02_COLLABORATIVE_SEARCH         | Risky      | Low         | 29.31   |
| COMPLEX_03_COLLABORATIVE_ATTACK         | Risky      | High        | 47.93   |
| COMPLEX_04_COLLABORATIVE_DEFENSE        | Borderline | Low         | 66.79   |
| COMPLEX_05_DYNAMIC_ROLE_ASSIGNMENT      | Risky      | High        | 45.81   |
| COMPLEX_06_FORMATION_BREAK_COLLISION    | Risky      | Low         | 43.14   |
| COMPLEX_07_GPS_GLITCH                 | Risky      | Low         | 45.03   |
| COMPLEX_08_SIGNAL_LOSS                | Risky      | Low         | 12.30   |
| COMPLEX_09_EMERGENCY_EVASION          | Risky      | Low         | 41.77   |
| COMPLEX_10_COORDINATED_LANDING        | Safe       | Medium      | 78.42   |

### Fixed-Weight

| Mission                               | Prediction | Efficiency | Score  |
|---------------------------------------|------------|-------------|---------|
| COMPLEX_01_MULTI_LAYER_FORMATION        | Risky      | Medium      | 29.80   |
| COMPLEX_02_COLLABORATIVE_SEARCH         | Risky      | Low         | 20.80   |
| COMPLEX_03_COLLABORATIVE_ATTACK         | Risky      | High        | 17.25   |
| COMPLEX_04_COLLABORATIVE_DEFENSE        | Borderline | Low         | 59.20   |
| COMPLEX_05_DYNAMIC_ROLE_ASSIGNMENT      | Risky      | High        | 19.57   |
| COMPLEX_06_FORMATION_BREAK_COLLISION    | Risky      | Low         | 26.28   |
| COMPLEX_07_GPS_GLITCH                 | Risky      | Low         | 1.92    |
| COMPLEX_08_SIGNAL_LOSS                | Risky      | Low         | 0.00    |
| COMPLEX_09_EMERGENCY_EVASION          | Risky      | Low         | 38.99   |
| COMPLEX_10_COORDINATED_LANDING        | Risky      | Medium      | 28.11   |

## Key Findings

1. **Single-Metric** performs better than **Fixed-Weight**:
   - 90% vs 80% accuracy
   - Higher precision (1.0 vs 0.889)
   - Similar recall (0.889)

2. **Fixed-Weight** is more conservative:
   - Tends to classify more missions as "Risky"
   - Misses the "Safe" mission (COMPLEX_10)
   - Lower overall scores across all missions

3. **Single-Metric** correctly identifies:
   - 9 out of 10 missions correctly
   - Only misclassifies COMPLEX_04 (Borderline instead of Risky)
   - Correctly identifies the only "Safe" mission

## Issues Fixed

1. Ground Truth reading error - now correctly reading safety_label
2. Multi-Agent-Debate hard constraints - adjusted thresholds
3. Formation Stability calculation - unified with data generation
4. Efficiency Ground Truth - now using dataset values
5. Emoji encoding issues - removed for Windows compatibility
6. WebSocket broadcast parameter - removed unsupported parameter
7. JSON serialization - added NumpyEncoder for NumPy types
8. Support value type - converted to int for JSON serialization
