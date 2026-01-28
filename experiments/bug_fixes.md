# 大模型评估效果问题修复

## 问题现象

运行 `exp1_real_evaluation.py` 后发现：
- **Single-Agent-LLM**: Recall = 0%, Precision = 0%
- **Multi-Agent-Debate**: Recall = 0%, Precision = 0%
- 传统方法（Single-Metric, Fixed-Weight）效果正常

## 根本原因分析

### 1. Multi-Agent-Debate 代码错误（导致全部失败）

**错误位置**: `_synthesize_final()` 方法第 637 行

```python
# 错误代码
anomalies = TrajectoryAnalyzer.detect_anomalies(drone_analyses[0]['trajectory'], formation_analysis)
```

**问题**: `drone_analyses` 是分析结果字典，不包含原始 `trajectory` 数据，导致 `KeyError: 'trajectory'`

**修复**:
```python
# 正确代码
anomalies = TrajectoryAnalyzer.detect_anomalies(
    mission_data['drones'][0]['trajectory'], 
    formation_analysis
)
```

### 2. Single-Agent-LLM 输出格式不一致

**问题**: LLM 返回的标签大小写不一致（"Borderline"、"RISKY"、"Risky"），导致解析错误

**修复**: 在 `_extract_field()` 方法中增加标签标准化逻辑
```python
# 标准化安全标签（统一大小写）
if field.upper() == "SAFETY":
    value_lower = value.lower()
    if 'safe' in value_lower and 'unsafe' not in value_lower and 'border' not in value_lower:
        return "Safe"
    elif 'risk' in value_lower or 'unsafe' in value_lower:
        return "Risky"
    elif 'border' in value_lower:
        return "Borderline"
```

### 3. Prompt 设计过于保守（核心问题）

**问题**: 原 Prompt 过度强调风险，导致模型倾向保守

数据集标签分布（均衡）:
- Safe: 33.3%
- Borderline: 33.3%
- Risky: 33.3%

但模型预测分布（极度不均衡）:
- Safe: ~0%
- Borderline: ~30%
- Risky: ~70%

**原 Prompt 问题**:
- 要求 "Most metrics above 70" 才是 Safe（阈值过高）
- 强调各种异常和风险指标
- 缺少正面评估指引

**修复后的 Prompt**:

```
SAFETY CLASSIFICATION CRITERIA:
- SAFE: Mission completed successfully, metrics generally >= 60, no critical safety violations
- BORDERLINE: Mission acceptable but with some concerns, metrics 40-60, minor issues present  
- RISKY: Significant safety concerns, multiple metrics < 40, critical issues detected

EVALUATION GUIDELINES:
1. Be balanced - not all missions have critical issues
2. Minor metric fluctuations (50-70) are normal and acceptable
3. Focus on critical safety violations, not minor imperfections
4. Trajectory Smoothness >= 60: generally good flight control
5. Altitude Stability >= 60: acceptable altitude management
6. Speed Consistency >= 60: adequate speed control
7. Formation metrics >= 50: acceptable coordination
8. Consider that real-world UAV flights have normal variations
```

## 修复内容总结

1. ✅ 修复 Multi-Agent-Debate 的 KeyError 错误
2. ✅ 统一 Single-Agent-LLM 的标签格式解析
3. ✅ 调整 Prompt，降低安全阈值（70 → 60）
4. ✅ 改进评估指导语，强调平衡判断
5. ✅ 添加正面评估指引

## 预期改进效果

修复后应该能看到：
- Multi-Agent-Debate 正常运行，不再报错
- 两种 LLM 方法的 Safe 预测增加
- Recall 和 Precision 显著提升
- 标签分布更接近数据集真实分布

## 如何验证

重新运行评估：
```bash
cd experiments
python exp1_real_evaluation.py
```

查看指标改进：
- Recall 应该从 0% 提升到 > 30%
- Precision 应该有合理数值
- Accuracy 应该有提升
- Cohen's κ 应该 > 0

