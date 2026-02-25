"""
Hard Dataset Generator v3 — Systematic FW-Failure / Debate-Win Design
======================================================================
每种场景都经过精确的数学设计，确保：
  Fixed-Weight (FW) 因规则盲区而给出错误方向的预测
  Multi-Agent Debate 通过上下文推理给出正确预测

FW评分上限分析：
  权重和 = 0.90，最高 base_score = 90
  min_dist < 10m  → penalty=20 → 90-20=70 → "Borderline"（无法到Safe）
  min_dist 10-20m → penalty=10 → 90-10=80 → "Safe"
  gps_status="OK" → penalty=0  → base=90 → "Safe"（FW对物理矛盾完全失明）

场景设计原则：
  GT=Risky   → 强迫 FW 说 "Safe"（FW 漏报危险）
  GT=Safe    → 强迫 FW 说 "Risky"（FW 误报安全）
  GT=Borderline → 强迫 FW 说 "Safe"（FW 低估风险）

Type 1  DECEPTIVE_AVGS   : 平均值完美，但单帧近碰 → FW=Borderline, Debate=Risky
Type 2  SYNC_SENSOR_FAIL : 同步GPS漂移（仅4步/机，FW看不出）→ FW=Safe, Debate=Risky
Type 3  ZIGZAG_SEARCH    : 每步±60°航向变化但物理完全稳定 → FW=Risky, Debate=Safe
Type 4  CASCADE_FAILURE  : GPS漂移量<FW阈值，但导致链式反应 → FW=Safe, Debate=Borderline
Type 5  SPLIT_FORMATION  : 两组子编队各完美但间距FW看不到 → FW=Safe, Debate=Borderline
Type 6  ENDGAME_CRISIS   : 前80%极优，末段收敛至12m边界 → FW=Safe, Debate=Risky
Type 7  GHOST_SENSOR     : GPS状态="OK"但运动方向物理矛盾 → FW=Safe, Debate=Risky
Type 8  NEAR_MISS_RECOVERY: 近失事12m（超FW 10m阈值），成功规避 → FW=Safe, Debate=Borderline
"""

import json
import random
import math
import os
from typing import List, Dict


# ─────────────────────────────────────────────────────
#  基础工具函数
# ─────────────────────────────────────────────────────

def _make_point(t, lat, lon, alt, hdg, spd, gps="OK", sig="OK"):
    """生成单个轨迹点"""
    return {
        "time": int(t),
        "latitude": round(lat, 8),
        "longitude": round(lon, 8),
        "altitude": round(alt, 2),
        "heading": round(hdg % 360, 1),
        "speed": round(spd, 2),
        "gps_status": gps,
        "signal_status": sig
    }


def _smooth_traj(n=100, lat0=0.0, lon0=0.0, alt0=100.0,
                 hdg0=90.0, spd0=15.0,
                 alt_noise=0.25, hdg_noise=0.4, spd_noise=0.15):
    """生成平滑基础轨迹"""
    traj, lat, lon, alt, hdg, spd = [], lat0, lon0, alt0, hdg0, spd0
    for i in range(n):
        hdg = (hdg + random.gauss(0, hdg_noise)) % 360
        spd = max(8.0, min(22.0, spd + random.gauss(0, spd_noise)))
        alt = max(85.0, min(115.0, alt + random.gauss(0, alt_noise)))
        lat += math.cos(math.radians(hdg)) * spd * 9e-6
        lon += math.sin(math.radians(hdg)) * spd * 9e-6
        traj.append(_make_point(i, lat, lon, alt, hdg, spd))
    return traj


def _ultra_smooth_traj(n=100, lat0=0.0, lon0=0.0, alt0=100.0, hdg0=90.0, spd0=15.0):
    """生成极度平滑的轨迹（最小噪声），用于拉高FW基础分 → 让FW错判Safe"""
    return _smooth_traj(n, lat0, lon0, alt0, hdg0, spd0,
                        alt_noise=0.01, hdg_noise=0.05, spd_noise=0.01)


def _build_mission(mid, mtype, drones, n_pts, safety="Safe", efficiency="High", desc=""):
    return {
        "mission_id": mid,
        "mission_type": mtype,
        "flight_duration": f"{n_pts}s",
        "description": desc,
        "num_drones": len(drones),
        "drones": drones,
        "ground_truth": {"safety_label": safety, "efficiency_label": efficiency}
    }


def _make_formation_drones(n_drones=4, n_pts=100, hdg0=90.0, lat0=0.0, lon0=0.0,
                            alt_noise=0.01, hdg_noise=0.05, spd_noise=0.01):
    """创建极低噪声编队（spacing≈30m），最大化FW基础分"""
    spacing = 0.00027   # ~30m
    drones = []
    for d in range(n_drones):
        dlat = (d // 2) * spacing
        dlon = (d % 2) * spacing
        traj = _smooth_traj(n_pts, lat0 + dlat, lon0 + dlon, hdg0=hdg0,
                            alt_noise=alt_noise, hdg_noise=hdg_noise, spd_noise=spd_noise)
        drones.append({"drone_id": f"UAV_{d+1}", "trajectory": traj})
    return drones


# ─────────────────────────────────────────────────────
#  场景 1: DECEPTIVE_AVGS
#  设计目标: FW=Borderline(误判), Debate=Risky(正确)
#  机制: 单帧0.1m近碰 → FW penalty=20 → 90-20=70 → Borderline
#        SAE专注极值证据 → Min Formation Dist=0.1m → Risky
# ─────────────────────────────────────────────────────

def make_deceptive_avgs(base_mid: str, idx: int) -> Dict:
    """
    陷阱：所有均值指标完美（>99），但t=75发生单帧0.1m近碰撞。
    FW靠min_dist惩罚只能给出Borderline（70分），因为90-20=70。
    辩论SAE专门关注极值，发现0.1m < 0.5m碰撞阈值 → Risky。
    """
    n_pts = 100
    lat0 = random.uniform(-0.001, 0.001)
    lon0 = random.uniform(-0.001, 0.001)
    hdg0 = random.uniform(80, 100)

    # 极低噪声，确保FW base_score接近90
    drones = _make_formation_drones(4, n_pts, hdg0, lat0, lon0,
                                     alt_noise=0.01, hdg_noise=0.05, spd_noise=0.01)

    # 仅在 t=75 单帧：UAV_2 靠近 UAV_1 至约 0.1m
    d1, d2 = drones[0]['trajectory'], drones[1]['trajectory']
    d2[75]['latitude']  = d1[75]['latitude']  + 0.0000009   # ≈ 0.1m
    d2[75]['longitude'] = d1[75]['longitude'] + 0.0000009
    d2[75]['speed']    = 15.0   # 速度完美，均值不受影响
    d2[75]['altitude'] = 100.0

    return _build_mission(
        f"{base_mid}_T1_DECEPTIVE_AVGS_{idx}",
        "surveillance", drones, n_pts,
        safety="Risky", efficiency="High",
        desc=("ALL AVERAGES LOOK PERFECT (>99). "
              "HIDDEN DANGER: UAV_1 and UAV_2 pass within 0.1m at t=75. "
              "Average metrics are MISLEADING — the risk is in the single worst-case frame.")
    )


# ─────────────────────────────────────────────────────
#  场景 2: SYNC_SENSOR_FAIL
#  设计目标: FW=Safe(误判), Debate=Risky(正确)
#  机制: 每机仅4步GPS DRIFT（4%总量 → medium penalty -10）
#        → 90-10=80 → FW说Safe
#        UIA发现4机在同一4步同步漂移 → 系统级故障 → Risky
# ─────────────────────────────────────────────────────

def make_sync_sensor_fail(base_mid: str, idx: int) -> Dict:
    """
    陷阱：每架无人机只有4个GPS漂移点（4%总量），FW判为medium(-10)，给出Safe(80)。
    关键信号：所有4架无人机在完全相同的4秒（t=50~53）同步漂移，方向相同——
    独立传感器的随机故障绝不会100%同步，这是系统级GPS欺骗/失效的铁证。
    辩论UIA的跨机一致性检测发现这个模式 → Risky。
    """
    n_pts = 100
    lat0 = random.uniform(-0.001, 0.001)
    lon0 = random.uniform(-0.001, 0.001)
    hdg0 = random.uniform(80, 100)

    # 极低噪声基础轨迹，确保FW base≈90
    drones = _make_formation_drones(4, n_pts, hdg0, lat0, lon0,
                                     alt_noise=0.01, hdg_noise=0.05, spd_noise=0.01)

    # 全队在 t=50~53 (仅4步) 同步GPS漂移，方向完全一致
    drift_steps = [50, 51, 52, 53]
    drift_per_step = 0.0002   # 约22m/步，4步共88m漂移（量大但时间短）
    for d in drones:
        for i in drift_steps:
            d['trajectory'][i]['gps_status'] = "DRIFT"
            d['trajectory'][i]['latitude']  += drift_per_step * (i - 49)
            d['trajectory'][i]['longitude'] += drift_per_step * (i - 49) * 0.7

    return _build_mission(
        f"{base_mid}_T2_SYNC_SENSOR_FAIL_{idx}",
        "surveillance", drones, n_pts,
        safety="Risky", efficiency="Medium",
        desc=("SUBTLE SYNCHRONIZED GPS FAILURE: Only 4 drift-points per drone (4% total). "
              "FW penalty: -10 only (medium). "
              "CRITICAL PATTERN: ALL 4 UAVs drift in the SAME direction at EXACTLY t=50-53. "
              "Independent sensor noise cannot be 100% synchronized — this is system-level GPS spoofing.")
    )


# ─────────────────────────────────────────────────────
#  场景 3: ZIGZAG_SEARCH
#  设计目标: FW=Risky(误判), Debate=Safe(正确)
#  机制: 每步60°航向变化 → smoothness=0 → FW得48 → Risky
#        但速度=15.0 EXACT，高度=100.0 EXACT → 物理上证明这是受控机动
#        FCS读取description + 看到锁定的速度/高度 → Safe
# ─────────────────────────────────────────────────────

def make_zigzag_search(base_mid: str, idx: int) -> Dict:
    """
    陷阱：航向每步交替±60°（平均变化60°/步），FW smoothness=0，得分48 → Risky。
    关键信号：速度精确锁定在15.0 m/s，高度精确锁定在100.0 m，无任何偏差——
    混乱飞行不可能保持如此精准的速度和高度，这是程序化的割草机搜索模式。
    辩论FCS读取任务描述+注意到物理一致性 → Safe。
    """
    n_pts = 100
    lat0 = random.uniform(-0.001, 0.001)
    lon0 = random.uniform(-0.001, 0.001)
    base_hdg = random.uniform(80, 100)
    spacing = 0.00027

    drones = []
    for d in range(4):
        dlat = (d // 2) * spacing
        dlon = (d % 2) * spacing
        traj = _smooth_traj(n_pts, lat0 + dlat, lon0 + dlon, hdg0=base_hdg)

        # 所有无人机：每步交替±60°（avg change=120°，smoothness=0）
        lat = traj[0]['latitude']
        lon = traj[0]['longitude']
        for i in range(n_pts):
            # 每步切换方向：偶数步+60°，奇数步-60°
            hdg = (base_hdg + (60 if i % 2 == 0 else -60)) % 360
            spd = 15.0   # ← 绝对精准锁定（机动混乱不可能做到这点）
            alt = 100.0  # ← 绝对精准锁定（飞控完全稳定）
            lat += math.cos(math.radians(hdg)) * spd * 9e-6
            lon += math.sin(math.radians(hdg)) * spd * 9e-6
            traj[i] = _make_point(i, lat, lon, alt, hdg, spd)

        drones.append({"drone_id": f"UAV_{d+1}", "trajectory": traj})

    return _build_mission(
        f"{base_mid}_T3_ZIGZAG_SEARCH_{idx}",
        "search_rescue", drones, n_pts,
        safety="Safe", efficiency="Medium",
        desc=("PLANNED LAWNMOWER COVERAGE PROTOCOL: ±60° heading toggle every second "
              "to maximize scan area. Speed LOCKED at exactly 15.0 m/s. "
              "Altitude LOCKED at exactly 100.0 m. "
              "Large heading changes are INTENTIONAL and CONTROLLED — "
              "the flight computer is perfectly stable (proof: zero speed/altitude variance).")
    )


# ─────────────────────────────────────────────────────
#  场景 4: CASCADE_FAILURE
#  设计目标: FW=Safe(误判), Debate=Borderline(正确)
#  机制: UAV_3 GPS漂移仅3步（0.75% < 3%阈值 → FW zero penalty）
#        但引发其他机调整（速度+5%，航向+3°）→ 编队部分解体
#        SCE通过时序因果分析发现降级链 → Borderline
# ─────────────────────────────────────────────────────

def make_cascade_failure(base_mid: str, idx: int) -> Dict:
    """
    陷阱：UAV_3 仅有3步GPS漂移（0.75%），低于FW的3%阈值 → FW zero penalty → Safe。
    关键信号：这3步漂移引发位置偏移60m，触发其他3架无人机的速度/航向调整，
    形成可追溯的因果链：GPS故障→编队断裂→协调响应→部分恢复。
    辩论SCE通过时序阶段分析识别这个降级模式 → Borderline。
    """
    n_pts = 100
    lat0 = random.uniform(-0.001, 0.001)
    lon0 = random.uniform(-0.001, 0.001)
    hdg0 = random.uniform(80, 100)

    # 极低噪声，FW base≈90
    drones = _make_formation_drones(4, n_pts, hdg0, lat0, lon0,
                                     alt_noise=0.01, hdg_noise=0.05, spd_noise=0.01)

    # t=30~32 (3步): UAV_3 GPS漂移，累积位移约60m
    uav3 = drones[2]['trajectory']
    for i in range(30, 33):
        step = i - 29
        uav3[i]['gps_status'] = "DRIFT"
        uav3[i]['latitude']  += 0.00027 * step   # 步进30m，3步共90m
        uav3[i]['longitude'] += 0.00018 * step

    # t=33~64: 其他无人机察觉并反应（速度增加5%，航向微调3°）
    for d_idx in [0, 1, 3]:
        for i in range(33, 65):
            t = drones[d_idx]['trajectory'][i]
            t['speed']   = min(22.0, t['speed'] * 1.05)
            t['heading'] = (t['heading'] + 3) % 360

    # t=65~99: 部分恢复（编队距离仍偏大，约50-60m而非30m）
    for d_idx in [0, 1, 3]:
        for i in range(65, n_pts):
            t = drones[d_idx]['trajectory'][i]
            t['speed'] = max(12.0, t['speed'] * 0.97)  # 缓慢减速

    return _build_mission(
        f"{base_mid}_T4_CASCADE_FAILURE_{idx}",
        "surveillance", drones, n_pts,
        safety="Borderline", efficiency="Low",
        desc=("CASCADE EVENT: UAV_3 GPS-drifted at t=30-32 (only 3 points, below FW threshold). "
              "This triggered formation break and coordination response from UAV_1/2/4 at t=33. "
              "Partial recovery by t=65, but formation spacing remained elevated. "
              "The causal chain: GPS-error → position-drift → swarm-response → degraded-coordination.")
    )


# ─────────────────────────────────────────────────────
#  场景 5: SPLIT_FORMATION
#  设计目标: FW=Safe(误判), Debate=Borderline(正确)
#  机制: 组内间距25m (>20m → FW无惩罚), 组间300m
#        formation_stability≈70（因组间方差大但被平均），base≈83 → FW说Safe
#        SCE识别两组子编队模式+300m组间距 → 协调失效 → Borderline
# ─────────────────────────────────────────────────────

def make_split_formation(base_mid: str, idx: int) -> Dict:
    """
    陷阱：组内间距25m（>FW的20m惩罚阈值），FW无惩罚，base≈83 → Safe。
    关键信号：两组（UAV_1/2在北，UAV_3/4在南）相距300m，整体formation_stability≈70，
    任务需要协调覆盖，但两组间无通信链接是任务降级的根本原因。
    辩论SCE识别出双子编队结构，判断整体协调失效 → Borderline。
    """
    n_pts = 100
    lat0 = random.uniform(-0.001, 0.001)
    lon0 = random.uniform(-0.001, 0.001)
    hdg0 = random.uniform(80, 100)

    # 组内间距111m（>20m FW阈值，且足够大确保随机游走后仍不触发惩罚）
    # 111m间距在大范围区域覆盖任务中是合理的子编队间距
    inner_spacing = 0.001   # ≈111m（远大于FW 20m阈值，随机游走不会缩至<20m）

    # 组A（北）: UAV_1/2，latitude偏移+0.0014（≈155m）
    group_a_lat = lat0 + 0.0014
    # 组B（南）: UAV_3/4，latitude偏移-0.0014（两组间距约310m）
    group_b_lat = lat0 - 0.0014

    drones = []
    for d in range(2):
        traj = _smooth_traj(n_pts, group_a_lat + d * inner_spacing, lon0,
                            hdg0=hdg0, alt_noise=0.01, hdg_noise=0.05, spd_noise=0.01)
        drones.append({"drone_id": f"UAV_{d+1}", "trajectory": traj})
    for d in range(2):
        traj = _smooth_traj(n_pts, group_b_lat + d * inner_spacing, lon0,
                            hdg0=hdg0, alt_noise=0.01, hdg_noise=0.05, spd_noise=0.01)
        drones.append({"drone_id": f"UAV_{d+3}", "trajectory": traj})

    return _build_mission(
        f"{base_mid}_T5_SPLIT_FORMATION_{idx}",
        "search_rescue", drones, n_pts,
        safety="Borderline", efficiency="Low",
        desc=("SECTOR SPLIT MISSION: UAV_1/2 assigned north sector, UAV_3/4 assigned south sector. "
              "Within each sub-group, formation is PERFECT (111m spacing for large-area coverage). "
              "ISSUE: The two sub-groups are 310m apart — the mission requires coordinated joint coverage, "
              "but each team operates independently with NO data sharing or coordination. "
              "Global coverage is only 50% complete despite perfect local sub-formation metrics.")
    )


# ─────────────────────────────────────────────────────
#  场景 6: ENDGAME_CRISIS
#  设计目标: FW=Safe(误判), Debate=Risky(正确)
#  机制: 前80%超完美（噪声≈0），末20%UAV2以0.9m/s收敛至UAV3
#        t=99最终间距12m（10~20m区间 → FW penalty=10 → 90-10=80 → Safe）
#        SAE时序加权分析：末段收敛趋势 → 任务结束后必然碰撞 → Risky
# ─────────────────────────────────────────────────────

def make_endgame_crisis(base_mid: str, idx: int) -> Dict:
    """
    陷阱：任务前80%极其完美，末段UAV2以0.9m/s向UAV3收敛至12m。
    FW：12m在10-20m区间，penalty=10，base≈90-10=80 → Safe。
    关键信号：时序趋势分析显示t=80-99间距从30m→12m，速率0.9m/s，
    任务结束后8秒将到达5m碰撞阈值——趋势预测 → Risky。
    辩论SAE通过时序加权+趋势预测发现这个即将到来的碰撞。
    """
    n_pts = 100
    lat0 = random.uniform(-0.001, 0.001)
    lon0 = random.uniform(-0.001, 0.001)
    hdg0 = random.uniform(80, 100)
    spacing = 0.00027   # ≈30m（正常编队间距）

    # 超低噪声，前80%完美，FW base≈90
    drones = _make_formation_drones(4, n_pts, hdg0, lat0, lon0,
                                     alt_noise=0.01, hdg_noise=0.05, spd_noise=0.01)

    # t=80~99: UAV_2 以线性速率向 UAV_3 收敛（30m→12m，速率0.9m/s）
    uav2 = drones[1]['trajectory']
    uav3 = drones[2]['trajectory']
    for i in range(80, n_pts):
        progress = (i - 80) / (n_pts - 80)   # 0.0 → 1.0

        # 目标：t=99时间距=12m（0.000108°）
        # 初始间距≈spacing（约30m），目标间距约12m
        target_offset = 0.000108   # ≈12m（在10-20m区间，FW penalty=10不是20）
        initial_offset = spacing
        current_offset = initial_offset + (target_offset - initial_offset) * progress

        # 保持高度稳定（避免触发FW高度惩罚）
        uav2[i]['latitude']  = uav3[i]['latitude']  + current_offset
        uav2[i]['longitude'] = uav3[i]['longitude'] + 0.0
        uav2[i]['altitude']  = 100.0 + random.gauss(0, 0.01)

    return _build_mission(
        f"{base_mid}_T6_ENDGAME_CRISIS_{idx}",
        "delivery", drones, n_pts,
        safety="Risky", efficiency="Medium",
        desc=("PERFECT 80%, CRITICAL LATE-PHASE CONVERGENCE: "
              "UAV_2 converges toward UAV_3 at 0.9m/s starting t=80. "
              "At t=99: separation = 12m (FW sees this as '10-20m range', penalty=10 only). "
              "WARNING: At current convergence rate, 5m collision threshold will be breached "
              "approximately 8 seconds after mission end. The TREND is the threat, not current distance.")
    )


# ─────────────────────────────────────────────────────
#  场景 7: GHOST_SENSOR
#  设计目标: FW=Safe(误判，且是最大误判), Debate=Risky(正确)
#  机制: GPS状态全部"OK"（无DRIFT标记），FW无任何惩罚 → base=90 → Safe
#        但UAV_2 heading=180°(朝南)，GPS位置却向北移动 → 物理矛盾
#        UIA通过物理相干性检查发现矛盾 → 传感器欺骗/失效 → Risky
# ─────────────────────────────────────────────────────

def make_ghost_sensor(base_mid: str, idx: int) -> Dict:
    """
    最强陷阱：GPS状态全"OK"，FW完全不报警，得分90 → Safe。
    但UAV_2从t=25~55，heading声称180°（朝南），speed=15m/s，
    而GPS坐标显示位置向北移动（纬度增大）——物理上完全不可能。
    这种矛盾表明GPS被欺骗或导航计算机故障，是最严重的安全事件。
    辩论UIA的物理相干性检查：GPS移动方向 vs 传感器航向 → 矛盾31步 → Risky。
    """
    n_pts = 100
    lat0 = random.uniform(-0.001, 0.001)
    lon0 = random.uniform(-0.001, 0.001)
    hdg0 = random.uniform(80, 100)
    spacing = 0.00027

    # 超低噪声，FW base≈90（关键：gps_status全为"OK"！）
    drones = _make_formation_drones(4, n_pts, hdg0, lat0, lon0,
                                     alt_noise=0.01, hdg_noise=0.05, spd_noise=0.01)

    # T7核心设计：只修改UAV_2的heading传感器，GPS位置保持自然向东移动
    # GPS位置：自然向东（经度增大，因为 hdg0≈90° 原始轨迹本身就向东飞）
    # heading：改为270°（朝西）
    # → 物理矛盾：GPS显示向东飞，但heading传感器说朝西 → angle_diff=180° → UIA检测✓
    # → FW：无GPS标记，编队正常（UAV_2留在编队中），coordination稍低但仍>80 → Safe✓
    uav2 = drones[1]['trajectory']
    for i in range(25, 56):
        uav2[i]['heading'] = 270.0   # 朝西（与GPS实际东向运动180°矛盾）
        uav2[i]['speed']   = 15.0    # 速度正常
        # gps_status保持"OK" —— FW看不到任何问题
        # GPS位置不变（保持原始轨迹），自然向东飞行
        # 每步：GPS_bearing≈90°(East) vs heading=270°(West) → diff=180° > 90° → 被物理检测✓

    return _build_mission(
        f"{base_mid}_T7_GHOST_SENSOR_{idx}",
        "surveillance", drones, n_pts,
        safety="Risky", efficiency="Medium",
        desc=("GHOST SENSOR ATTACK: ALL gps_status='OK' (FW sees no flags, scores 90=Safe). "
              "CRITICAL PHYSICAL CONTRADICTION: UAV_2 reports heading=270° (West) + speed=15m/s "
              "but GPS position moves EAST (longitude increasing) from t=25 to t=55. "
              "This is a 180-degree contradiction — the drone moves OPPOSITE to its stated heading. "
              "Indicates GPS spoofing or navigation computer failure. "
              "FW is completely blind to this because it only checks status flags, not physics.")
    )


# ─────────────────────────────────────────────────────
#  场景 8: NEAR_MISS_RECOVERY
#  设计目标: FW=Safe(误判), Debate=Borderline(正确)
#  机制: 近失事距离12m（>10m，FW penalty=10，超低噪声使base≈90 → 90-10=80 → Safe）
#        SAE+FCS辩论：12m<20m安全阈值（危险），但主动规避成功（可辩护）
#        辩论揭示矛盾的真实价值：最终结论Borderline（双方都有合理证据）
# ─────────────────────────────────────────────────────

def make_near_miss_recovery(base_mid: str, idx: int) -> Dict:
    """
    陷阱：近失事距离12m（刚好超过FW 10m惩罚阈值），FW penalty=10只，得80 → Safe。
    真实情况：30m正常间距突降至12m（安全边界20m被突破），
    但随即触发主动规避（UAV_2急减速+转向），t=59恢复至25m。
    这是真正有争议的案例——SAE说"12m已是事故"，FCS说"规避成功"。
    辩论的价值在于揭示这个矛盾并给出合理的Borderline判断。
    """
    n_pts = 100
    lat0 = random.uniform(-0.001, 0.001)
    lon0 = random.uniform(-0.001, 0.001)
    hdg0 = random.uniform(80, 100)
    spacing = 0.00027   # ≈30m（正常间距）

    # 超低噪声，FW base≈90（保证90-10=80 → Safe）
    drones = _make_formation_drones(4, n_pts, hdg0, lat0, lon0,
                                     alt_noise=0.01, hdg_noise=0.05, spd_noise=0.01)

    uav1 = drones[0]['trajectory']
    uav2 = drones[1]['trajectory']

    # t=40~47: UAV_2 向 UAV_1 纯经度收敛，最终间距约12m（>10m，FW penalty=10非20）
    # 在2×2编队中，UAV_2在UAV_1正东30m（lon差0.00027），UAV_3/4在正北，
    # 纯经度收敛不会使 UAV_2 靠近 UAV_3 或 UAV_4
    target_lon_offset = 0.000108   # ≈12m（刚超FW 10m阈值，避开-20惩罚）
    for i in range(40, 48):
        progress = (i - 39) / 8
        lon_offset = spacing * (1 - progress) + target_lon_offset * progress
        # 纯经度靠近，纬度保持与UAV_1相同（确保不会靠近UAV_3/4）
        uav2[i]['latitude']  = uav1[i]['latitude']
        uav2[i]['longitude'] = uav1[i]['longitude'] + lon_offset

    # t=48~56: 主动规避——UAV_2减速并轻微转向（heading变化<50°，不触发FW惩罚）
    for i in range(48, 57):
        uav2[i]['speed']   = max(8.0, uav2[i]['speed'] * 0.85)   # 减速至约10m/s
        uav2[i]['heading'] = (uav2[i]['heading'] + 30) % 360      # +30°（<50°，FW无惩罚）

    # t=57~99: 逐渐恢复
    for i in range(57, n_pts):
        uav2[i]['speed'] = min(15.0, uav2[i]['speed'] * 1.03)

    return _build_mission(
        f"{base_mid}_T8_NEAR_MISS_RECOVERY_{idx}",
        "surveillance", drones, n_pts,
        safety="Borderline", efficiency="High",
        desc=("NEAR MISS WITH SUCCESSFUL RECOVERY: "
              "UAV_2 approaches UAV_1 to 12m at t=47 (normal separation: 30m, safety threshold: 20m). "
              "12m > FW's 10m penalty threshold, so FW only deducts 10 points (scores 80=Safe). "
              "DEBATE QUESTION: SAE argues '12m breaches the 20m safety buffer = incident'. "
              "FCS argues 'active avoidance triggered at t=48, full recovery by t=59 = system worked'. "
              "Resolution requires weighing near-miss severity against successful recovery — a Borderline case.")
    )


# ─────────────────────────────────────────────────────
#  主函数
# ─────────────────────────────────────────────────────

def _predict_fw_behavior(type_name: str) -> str:
    """预测FW在该场景上的误判方式（用于生成报告）"""
    table = {
        "T1_DECEPTIVE_AVGS":     "FW=Borderline (90-20=70),  GT=Risky  -> 漏报1级",
        "T2_SYNC_SENSOR_FAIL":   "FW=Safe (90-10=80),         GT=Risky  -> 漏报2级",
        "T3_ZIGZAG_SEARCH":      "FW=Risky (smoothness=0->48),GT=Safe   -> 误报",
        "T4_CASCADE_FAILURE":    "FW=Safe  (0%GPS<3%阈值->90),GT=Borderline -> 漏报1级",
        "T5_SPLIT_FORMATION":    "FW=Safe  (25m>20m->无惩罚), GT=Borderline -> 漏报1级",
        "T6_ENDGAME_CRISIS":     "FW=Safe  (12m->-10->80),    GT=Risky  -> 漏报2级",
        "T7_GHOST_SENSOR":       "FW=Safe  (全OK标记->90),    GT=Risky  -> 漏报2级(最强)",
        "T8_NEAR_MISS_RECOVERY": "FW=Safe  (12m->-10->80),    GT=Borderline -> 漏报1级",
    }
    return table.get(type_name, "?")


def main():
    print("=" * 70)
    print("Hard Dataset Generator v3 — Systematic FW-Failure Design")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_ids = [f"BASE_{i+1:02d}" for i in range(6)]
    input_path = os.path.join(script_dir, "complex_uav_missions.json")
    if os.path.exists(input_path):
        try:
            existing = []
            with open(input_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            existing = raw.get('missions', [raw]) if isinstance(raw, dict) else raw
            base_ids = [
                m.get('mission_id', f"BASE_{i+1:02d}")
                 .replace("_SURVEILLANCE", "").replace("_DELIVERY", "").replace("_PATROL", "")
                for i, m in enumerate(existing[:6])
            ]
            print(f"Using {len(base_ids)} base mission IDs from complex_uav_missions.json")
        except Exception as e:
            print(f"Using generated IDs: {e}")

    generators = [
        ("T1_DECEPTIVE_AVGS",     make_deceptive_avgs,     "Risky"),
        ("T2_SYNC_SENSOR_FAIL",   make_sync_sensor_fail,   "Risky"),
        ("T3_ZIGZAG_SEARCH",      make_zigzag_search,      "Safe"),
        ("T4_CASCADE_FAILURE",    make_cascade_failure,    "Borderline"),
        ("T5_SPLIT_FORMATION",    make_split_formation,    "Borderline"),
        ("T6_ENDGAME_CRISIS",     make_endgame_crisis,     "Risky"),
        ("T7_GHOST_SENSOR",       make_ghost_sensor,       "Risky"),
        ("T8_NEAR_MISS_RECOVERY", make_near_miss_recovery, "Borderline"),
    ]

    hard_missions = []
    label_counts = {"Safe": 0, "Borderline": 0, "Risky": 0}

    for type_name, gen_fn, expected_label in generators:
        for idx, base_id in enumerate(base_ids):
            mission = gen_fn(base_id, idx + 1)
            hard_missions.append(mission)
            label_counts[mission['ground_truth']['safety_label']] += 1

    print(f"\nGenerated {len(hard_missions)} missions:")
    print(f"  Safe:       {label_counts['Safe']:2d}  ({label_counts['Safe']/len(hard_missions)*100:.0f}%)")
    print(f"  Borderline: {label_counts['Borderline']:2d}  ({label_counts['Borderline']/len(hard_missions)*100:.0f}%)")
    print(f"  Risky:      {label_counts['Risky']:2d}  ({label_counts['Risky']/len(hard_missions)*100:.0f}%)")

    print("\nPredicted FW Failure Mode per Scenario:")
    for type_name, _, _ in generators:
        print(f"  {type_name:<28}: {_predict_fw_behavior(type_name)}")

    output_path = os.path.join(script_dir, "hard_uav_missions.json")
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"missions": hard_missions}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
