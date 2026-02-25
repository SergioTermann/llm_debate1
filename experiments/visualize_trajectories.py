"""
UAV Trajectory Visualizer  v2
==============================
为 hard_uav_missions.json 中的 8 种对抗场景绘制轨迹图。
改进：
  - 自适应坐标范围（不强制等比例），让轨迹充满画框
  - 相对编队坐标（以编队质心为原点）突出无人机间相对运动
  - 时间渐变颜色（早→蓝, 晚→红）
  - 场景关键事件高亮标注
"""

import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

DRONE_COLORS = ["#5dade2", "#e74c3c", "#2ecc71", "#f39c12"]   # 蓝/红/绿/橙

SCENARIO_META = {
    "T1_DECEPTIVE_AVGS":     {"title": "T1 · Deceptive Averages",     "gt": "Risky",      "fw": "Borderline", "fw_ok": False},
    "T2_SYNC_SENSOR":        {"title": "T2 · Sync GPS Failure",        "gt": "Risky",      "fw": "Safe",       "fw_ok": False},
    "T3_ZIGZAG_SEARCH":      {"title": "T3 · Zigzag Lawnmower",        "gt": "Safe",       "fw": "Risky",      "fw_ok": False},
    "T4_CASCADE_FAILURE":    {"title": "T4 · Cascade Failure",         "gt": "Borderline", "fw": "Safe",       "fw_ok": False},
    "T5_SPLIT_FORMATION":    {"title": "T5 · Split Formation",          "gt": "Borderline", "fw": "Safe",       "fw_ok": False},
    "T6_ENDGAME_CRISIS":     {"title": "T6 · Endgame Crisis",          "gt": "Risky",      "fw": "Safe",       "fw_ok": False},
    "T7_GHOST_SENSOR":       {"title": "T7 · Ghost Sensor Attack",     "gt": "Risky",      "fw": "Safe",       "fw_ok": False},
    "T8_NEAR_MISS":          {"title": "T8 · Near-Miss Recovery",      "gt": "Borderline", "fw": "Safe",       "fw_ok": False},
}

GT_COLOR = {"Risky": "#e74c3c", "Borderline": "#f39c12", "Safe": "#2ecc71"}


def latlon_to_xy(traj, lat0=None, lon0=None):
    """经纬度 → 米，以指定原点（默认首点）"""
    if lat0 is None: lat0 = traj[0]['latitude']
    if lon0 is None: lon0 = traj[0]['longitude']
    xs = [(p['longitude'] - lon0) * 111320 * math.cos(math.radians(lat0)) for p in traj]
    ys = [(p['latitude']  - lat0) * 110540 for p in traj]
    return np.array(xs), np.array(ys)


def relative_to_centroid(drones):
    """
    将所有无人机轨迹转换为相对于编队质心的坐标。
    返回 [(xs_rel, ys_rel, traj), ...] 列表。
    """
    # 先把所有无人机的轨迹转为米坐标（统一参考原点）
    lat0 = drones[0]['trajectory'][0]['latitude']
    lon0 = drones[0]['trajectory'][0]['longitude']
    drone_xy = []
    for drone in drones:
        xs, ys = latlon_to_xy(drone['trajectory'], lat0, lon0)
        drone_xy.append((xs, ys, drone['trajectory']))

    n_pts = len(drone_xy[0][0])
    # 每帧质心
    cx = np.mean([d[0] for d in drone_xy], axis=0)   # shape (n_pts,)
    cy = np.mean([d[1] for d in drone_xy], axis=0)

    rel = []
    for (xs, ys, traj) in drone_xy:
        rel.append((xs - cx, ys - cy, traj))
    return rel, cx, cy


def set_ax_range(ax, all_x, all_y, pad=0.15):
    """根据数据范围自适应设置坐标轴（不强制等比例）"""
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    xspan = max(xmax - xmin, 1)
    yspan = max(ymax - ymin, 1)
    ax.set_xlim(xmin - xspan * pad, xmax + xspan * pad)
    ax.set_ylim(ymin - yspan * pad, ymax + yspan * pad)


def draw_gradient_track(ax, xs, ys, color, lw=1.5, alpha_start=0.25, alpha_end=0.95):
    """绘制时间渐变轨迹线（早段透明→晚段实）"""
    n = len(xs)
    for i in range(n - 1):
        t = i / max(n - 2, 1)
        alpha = alpha_start + (alpha_end - alpha_start) * t
        ax.plot(xs[i:i+2], ys[i:i+2], color=color, alpha=alpha, linewidth=lw, solid_capstyle='round')


def annotate_event(ax, x, y, text, color='yellow', offset=(0.05, 0.08), xspan=1, yspan=1):
    """在事件点画带标注的箭头"""
    ax.scatter(x, y, marker='*', s=180, color=color, zorder=10,
               edgecolors='black', linewidths=0.5,
               path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + offset[0] * xspan, y + offset[1] * yspan),
        fontsize=6.5, color=color, zorder=11,
        arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#111', alpha=0.8, edgecolor=color, lw=0.8),
    )


def draw_scenario(ax, mission, meta):
    drones  = mission['drones']
    n_pts   = len(drones[0]['trajectory'])
    mtype   = '_'.join(mission['mission_id'].split('_')[2:5])

    # ── 坐标模式：T5(编队分裂)用绝对坐标，其余用相对质心坐标 ──
    if mtype == 'T5_SPLIT_FORMATION':
        lat0 = drones[0]['trajectory'][0]['latitude']
        lon0 = drones[0]['trajectory'][0]['longitude']
        drone_rel = [(latlon_to_xy(d['trajectory'], lat0, lon0)[0],
                      latlon_to_xy(d['trajectory'], lat0, lon0)[1],
                      d['trajectory']) for d in drones]
        ax.set_xlabel('East (m)', fontsize=7, color='gray')
        ax.set_ylabel('North (m)', fontsize=7, color='gray')
    else:
        drone_rel, cx, cy = relative_to_centroid(drones)
        ax.set_xlabel('Rel East (m)', fontsize=7, color='gray')
        ax.set_ylabel('Rel North (m)', fontsize=7, color='gray')

    # 收集全部坐标求范围
    all_x = np.concatenate([d[0] for d in drone_rel])
    all_y = np.concatenate([d[1] for d in drone_rel])
    set_ax_range(ax, all_x, all_y)
    xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
    yspan = ax.get_ylim()[1] - ax.get_ylim()[0]

    # 绘制各机轨迹
    for d_idx, (xs, ys, traj) in enumerate(drone_rel):
        dc = DRONE_COLORS[d_idx % len(DRONE_COLORS)]
        draw_gradient_track(ax, xs, ys, dc)
        ax.scatter(xs[0],  ys[0],  marker='^', s=50, color=dc, zorder=8, edgecolors='white', linewidths=0.4)
        ax.scatter(xs[-1], ys[-1], marker='o', s=50, color=dc, zorder=8, edgecolors='white', linewidths=0.4)
        # 无人机编号标在起点旁
        ax.text(xs[0] + xspan * 0.01, ys[0] + yspan * 0.04,
                f'D{d_idx+1}', fontsize=6, color=dc, zorder=9,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    # ── 场景专项标注 ──
    xs0, ys0, _ = drone_rel[0]
    xs1, ys1, _ = drone_rel[1]

    if mtype == 'T1_DECEPTIVE_AVGS':
        # t=75 近碰点（相对质心后两机距离很小）
        annotate_event(ax, xs1[75], ys1[75],
                       '0.1m collision\nt=75', color='gold',
                       offset=(0.12, 0.15), xspan=xspan, yspan=yspan)

    elif mtype == 'T2_SYNC_SENSOR':
        # t=50~53 高亮所有机的漂移段
        for d_idx, (xs, ys, traj) in enumerate(drone_rel):
            ax.plot(xs[50:54], ys[50:54], color='yellow', linewidth=3, zorder=7, alpha=0.85)
        annotate_event(ax, drone_rel[0][0][51], drone_rel[0][1][51],
                       'SYNC DRIFT\nt=50-53\n4/4 UAVs', color='yellow',
                       offset=(0.12, 0.15), xspan=xspan, yspan=yspan)

    elif mtype == 'T3_ZIGZAG_SEARCH':
        # 只标注速度/高度锁定的注释（不画额外点）
        ax.text(0.5, 0.05,
                'Speed locked at 15.0 m/s\nAlt locked at 100.0 m\n→ Controlled maneuver',
                transform=ax.transAxes, fontsize=6.5, color='#2ecc71',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', alpha=0.85, edgecolor='#2ecc71', lw=0.8))

    elif mtype == 'T4_CASCADE_FAILURE':
        xs2, ys2, _ = drone_rel[2]
        annotate_event(ax, xs2[31], ys2[31],
                       'UAV_3 GPS drift\nt=30-32 (3pts)', color='yellow',
                       offset=(0.1, 0.15), xspan=xspan, yspan=yspan)
        # t=33~64 其他机反应段
        for d_idx in [0, 1, 3]:
            xs_d, ys_d, _ = drone_rel[d_idx]
            ax.plot(xs_d[33:65], ys_d[33:65], linewidth=2.5,
                    color=DRONE_COLORS[d_idx], alpha=1.0, linestyle='--', zorder=7)

    elif mtype == 'T5_SPLIT_FORMATION':
        # 用双向箭头标注两组间距
        mid_t = n_pts // 2
        xs2, ys2, _ = drone_rel[2]
        ax.annotate('', xy=(xs2[mid_t], ys2[mid_t]),
                    xytext=(xs0[mid_t], ys0[mid_t]),
                    arrowprops=dict(arrowstyle='<->', color='white', lw=1.5))
        mid_x = (xs0[mid_t] + xs2[mid_t]) / 2
        mid_y = (ys0[mid_t] + ys2[mid_t]) / 2
        ax.text(mid_x, mid_y, '~310m\ngap', fontsize=7, color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', alpha=0.85))

    elif mtype == 'T6_ENDGAME':
        # t=80 标注收敛起始
        ax.axvline(xs1[80], color='orange', linestyle=':', linewidth=1.2, alpha=0.7)
        ax.text(xs1[80] + xspan * 0.02, ys1[80] + yspan * 0.1,
                't=80\nconvergence\nstarts', fontsize=6, color='orange',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#111', alpha=0.75))
        annotate_event(ax, xs1[99], ys1[99], '12m\nt=99', color='orange',
                       offset=(-0.18, 0.12), xspan=xspan, yspan=yspan)

    elif mtype == 'T7_GHOST':
        # t=25~55 heading矛盾段（紫色高亮）
        ax.plot(xs1[25:56], ys1[25:56], color='magenta', linewidth=3, zorder=7, alpha=0.9)
        # 标注朝西箭头（与实际向东运动矛盾）
        arrow_t = 38
        ax.annotate('',
            xy=(xs1[arrow_t] - xspan * 0.09, ys1[arrow_t]),
            xytext=(xs1[arrow_t], ys1[arrow_t]),
            arrowprops=dict(arrowstyle='->', color='magenta', lw=2))
        ax.text(xs1[arrow_t] - xspan * 0.12, ys1[arrow_t] - yspan * 0.12,
                'heading=270°W\nGPS moves East\n180° contradiction',
                fontsize=6.5, color='magenta', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#111', alpha=0.85, edgecolor='magenta', lw=0.8))

    elif mtype == 'T8_NEAR_MISS':
        # t=47 最近点连线
        ax.plot([xs0[47], xs1[47]], [ys0[47], ys1[47]],
                'r-', linewidth=2.5, zorder=7, alpha=0.9)
        mid_x = (xs0[47] + xs1[47]) / 2
        mid_y = (ys0[47] + ys1[47]) / 2
        ax.text(mid_x + xspan * 0.04, mid_y + yspan * 0.1,
                '12m  t=47', fontsize=7, color='red',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#111', alpha=0.85, edgecolor='red', lw=0.8))
        # t=48~56 规避段
        ax.plot(xs1[48:57], ys1[48:57], linewidth=2.5, color=DRONE_COLORS[1],
                linestyle='--', zorder=7, alpha=1.0)
        ax.text(xs1[52] + xspan * 0.04, ys1[52] - yspan * 0.12,
                'avoidance\nt=48-56', fontsize=6.5, color=DRONE_COLORS[1],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#111', alpha=0.75))

    # ── 标题栏 ──
    gt   = meta['gt']
    fw   = meta['fw']
    gtc  = GT_COLOR.get(gt, 'white')
    fwc  = '#e74c3c' if not meta['fw_ok'] else '#2ecc71'   # 误判=红

    ax.set_title(meta['title'], fontsize=9, fontweight='bold', color='white', pad=4)
    ax.text(0.01, 0.99, f'GT: {gt}', transform=ax.transAxes,
            fontsize=7.5, va='top', color=gtc, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#111', alpha=0.85, edgecolor=gtc, lw=0.8))
    ax.text(0.99, 0.99, f'FW: {fw} ✗', transform=ax.transAxes,
            fontsize=7.5, va='top', ha='right', color=fwc,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#111', alpha=0.85, edgecolor=fwc, lw=0.8))

    ax.tick_params(labelsize=6, colors='gray')
    ax.set_facecolor('#0b0c1a')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.grid(True, linewidth=0.35, alpha=0.35, color='#334')


# ─────────────────────────────────────────────────────
#  Figure 1: 轨迹俯视图（2×4）
# ─────────────────────────────────────────────────────
def plot_overview(missions, output_path):
    type_first = {}
    for m in missions:
        mtype = '_'.join(m['mission_id'].split('_')[2:5])
        if mtype not in type_first:
            type_first[mtype] = m

    ordered = [
        "T1_DECEPTIVE_AVGS", "T2_SYNC_SENSOR",      "T3_ZIGZAG_SEARCH",  "T4_CASCADE_FAILURE",
        "T5_SPLIT_FORMATION","T6_ENDGAME_CRISIS",    "T7_GHOST_SENSOR",   "T8_NEAR_MISS",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor('#090912')
    fig.suptitle(
        'UAV Swarm Trajectory Overview  ·  Adversarial Scenarios (Relative Formation Frame)\n'
        'FW Accuracy: 2.1%   vs   Multi-Agent Debate Accuracy: ~70%+',
        fontsize=13, fontweight='bold', color='white', y=1.01
    )

    for idx, mtype in enumerate(ordered):
        ax   = axes[idx // 4][idx % 4]
        meta = SCENARIO_META.get(mtype, {"title": mtype, "gt": "?", "fw": "?", "fw_ok": False})
        m    = type_first.get(mtype)
        if m is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', color='gray', transform=ax.transAxes)
            continue
        draw_scenario(ax, m, meta)

    # 图例
    drone_patches = [mpatches.Patch(color=DRONE_COLORS[i], label=f'UAV_{i+1}') for i in range(4)]
    extra = [
        mpatches.Patch(color='#2471a3', alpha=0.5, label='Early (t≈0)'),
        mpatches.Patch(color='#c0392b', alpha=0.9, label='Late  (t≈end)'),
        mpatches.Patch(color='black',   alpha=0,   label='▲=start  ●=end'),
    ]
    fig.legend(
        handles=drone_patches + extra,
        loc='lower center', ncol=7, fontsize=8.5,
        facecolor='#1a1a2e', edgecolor='#555', labelcolor='white',
        framealpha=0.95, bbox_to_anchor=(0.5, -0.04)
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────
#  Figure 2: 高度 & 速度时序图（4×4）
# ─────────────────────────────────────────────────────
def plot_timeseries(missions, output_path):
    type_first = {}
    for m in missions:
        mtype = '_'.join(m['mission_id'].split('_')[2:5])
        if mtype not in type_first:
            type_first[mtype] = m

    ordered = [
        "T1_DECEPTIVE_AVGS", "T2_SYNC_SENSOR",      "T3_ZIGZAG_SEARCH",  "T4_CASCADE_FAILURE",
        "T5_SPLIT_FORMATION","T6_ENDGAME_CRISIS",    "T7_GHOST_SENSOR",   "T8_NEAR_MISS",
    ]

    fig, axes = plt.subplots(4, 4, figsize=(20, 14))
    fig.patch.set_facecolor('#090912')
    fig.suptitle('Altitude & Speed Time-Series  ·  8 Adversarial Scenarios',
                 fontsize=13, fontweight='bold', color='white')

    for col, mtype in enumerate(ordered):
        ax_alt = axes[(col // 4) * 2    ][col % 4]
        ax_spd = axes[(col // 4) * 2 + 1][col % 4]
        m = type_first.get(mtype)
        if not m:
            continue
        meta = SCENARIO_META.get(mtype, {})
        drones = m['drones']
        n_pts  = len(drones[0]['trajectory'])

        for d_idx, drone in enumerate(drones):
            traj = drone['trajectory']
            ts   = [p['time'] for p in traj]
            alts = [p['altitude'] for p in traj]
            spds = [p['speed'] for p in traj]
            dc   = DRONE_COLORS[d_idx % len(DRONE_COLORS)]

            # GPS漂移时间段用虚线
            gps_drift = [i for i, p in enumerate(traj) if p.get('gps_status') != 'OK']
            ax_alt.plot(ts, alts, color=dc, linewidth=1.0, alpha=0.85, label=f'D{d_idx+1}')
            ax_spd.plot(ts, spds, color=dc, linewidth=1.0, alpha=0.85)

            # 高亮GPS漂移段
            if gps_drift:
                drift_ts  = [ts[i] for i in gps_drift]
                drift_alt = [alts[i] for i in gps_drift]
                drift_spd = [spds[i] for i in gps_drift]
                ax_alt.scatter(drift_ts, drift_alt, c='yellow', s=12, zorder=6, alpha=0.8)
                ax_spd.scatter(drift_ts, drift_spd, c='yellow', s=12, zorder=6, alpha=0.8)

        # 场景特定标注
        mtype_short = '_'.join(mtype.split('_')[:2])
        if mtype == 'T6_ENDGAME_CRISIS':
            ax_alt.axvline(80, color='orange', linestyle='--', linewidth=1.2, alpha=0.8)
            ax_alt.text(81, ax_alt.get_ylim()[0], 't=80\nconvergence', fontsize=5.5, color='orange')
        elif mtype == 'T1_DECEPTIVE_AVGS':
            ax_spd.axvline(75, color='gold', linestyle='--', linewidth=1.2, alpha=0.8)
            ax_spd.text(76, ax_spd.get_ylim()[0], 't=75\ncollision', fontsize=5.5, color='gold')
        elif mtype == 'T7_GHOST_SENSOR':
            ax_alt.axvspan(25, 55, alpha=0.15, color='magenta')
            ax_spd.axvspan(25, 55, alpha=0.15, color='magenta')
            ax_spd.text(28, ax_spd.get_ylim()[0], 'heading\ncontradiction', fontsize=5.5, color='magenta')

        # 标题和轴标签
        gt  = meta.get('gt', '?')
        gtc = GT_COLOR.get(gt, 'white')
        ax_alt.set_title(f"{mtype.replace('_',' ')}\nGT={gt} | FW={meta.get('fw','?')} ✗",
                         fontsize=7, color=gtc, pad=2)
        ax_alt.set_ylabel('Alt (m)', fontsize=6.5, color='lightgray')
        ax_spd.set_ylabel('Spd (m/s)', fontsize=6.5, color='lightgray')
        ax_spd.set_xlabel('Time (s)', fontsize=6.5, color='lightgray')

        for ax in [ax_alt, ax_spd]:
            ax.set_facecolor('#0b0c1a')
            ax.tick_params(labelsize=5.5, colors='gray')
            ax.grid(True, linewidth=0.35, alpha=0.4, color='#334')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')

        if col % 4 == 0:
            ax_alt.legend(fontsize=5.5, labelcolor='white',
                          facecolor='#1a1a2e', edgecolor='#555', loc='upper right', framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────
#  Figure 3: 编队间距时序图（关键指标）
# ─────────────────────────────────────────────────────
def plot_formation_distance(missions, output_path):
    """绘制8个场景的逐帧编队最小间距，最直观展示FW盲区"""
    import itertools

    type_first = {}
    for m in missions:
        mtype = '_'.join(m['mission_id'].split('_')[2:5])
        if mtype not in type_first:
            type_first[mtype] = m

    ordered = [
        "T1_DECEPTIVE_AVGS", "T2_SYNC_SENSOR",      "T3_ZIGZAG_SEARCH",  "T4_CASCADE_FAILURE",
        "T5_SPLIT_FORMATION","T6_ENDGAME_CRISIS",    "T7_GHOST_SENSOR",   "T8_NEAR_MISS",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.patch.set_facecolor('#090912')
    fig.suptitle('Per-Frame Min Formation Distance  ·  Why FW Fails vs. Why Debate Wins',
                 fontsize=13, fontweight='bold', color='white')

    def min_dist_per_frame(drones):
        """计算每帧的编队最小间距（米）"""
        lat0 = drones[0]['trajectory'][0]['latitude']
        lon0 = drones[0]['trajectory'][0]['longitude']
        drone_xy = []
        for d in drones:
            xs, ys = latlon_to_xy(d['trajectory'], lat0, lon0)
            drone_xy.append(np.column_stack([xs, ys]))

        n = len(drone_xy[0])
        n_d = len(drone_xy)
        mins = []
        for t in range(n):
            positions = [drone_xy[i][t] for i in range(n_d)]
            d_min = float('inf')
            for i, j in itertools.combinations(range(n_d), 2):
                dist = np.linalg.norm(positions[i] - positions[j])
                d_min = min(d_min, dist)
            mins.append(d_min)
        return np.array(mins)

    for idx, mtype in enumerate(ordered):
        ax   = axes[idx // 4][idx % 4]
        m    = type_first.get(mtype)
        meta = SCENARIO_META.get(mtype, {})
        if not m:
            continue

        dists = min_dist_per_frame(m['drones'])
        ts    = np.arange(len(dists))
        gt    = meta.get('gt', '?')
        gtc   = GT_COLOR.get(gt, 'white')

        # 阈值线
        ax.axhline(20,  color='#f39c12', linestyle='--', linewidth=1.0, alpha=0.7, label='20m safety buffer')
        ax.axhline(0.5, color='#e74c3c', linestyle=':', linewidth=1.2, alpha=0.8, label='0.5m collision')

        # 距离曲线（颜色随危险程度变化）
        colors = []
        for d in dists:
            if d < 0.5:
                colors.append('#e74c3c')
            elif d < 20:
                colors.append('#f39c12')
            else:
                colors.append('#3498db')

        for i in range(len(ts) - 1):
            ax.plot(ts[i:i+2], dists[i:i+2], color=colors[i], linewidth=1.5, alpha=0.9)

        # FW看到的"均值"水平线（模拟FW的感知）
        fw_sees = np.mean(dists)
        ax.axhline(fw_sees, color='gray', linestyle='-', linewidth=1.5, alpha=0.6, label=f'FW sees avg={fw_sees:.0f}m')
        ax.fill_between(ts, dists, fw_sees, where=(dists < fw_sees),
                        alpha=0.15, color='red', label='FW underestimates risk')
        ax.fill_between(ts, dists, fw_sees, where=(dists > fw_sees),
                        alpha=0.08, color='green')

        ax.set_title(f"{mtype.replace('_',' ')}\nGT={gt}  |  FW={meta.get('fw','?')} ✗",
                     fontsize=7.5, color=gtc, pad=3)
        ax.set_xlabel('Time (s)', fontsize=6.5, color='gray')
        ax.set_ylabel('Min Dist (m)', fontsize=6.5, color='gray')
        ax.set_ylim(bottom=0)

        if idx == 0:
            ax.legend(fontsize=5.5, labelcolor='white',
                      facecolor='#1a1a2e', edgecolor='#555',
                      loc='upper right', framealpha=0.85, ncol=2)

        ax.set_facecolor('#0b0c1a')
        ax.tick_params(labelsize=5.5, colors='gray')
        ax.grid(True, linewidth=0.35, alpha=0.4, color='#334')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────
#  主函数
# ─────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(script_dir, 'hard_uav_missions.json')

    with open(data_path, encoding='utf-8') as f:
        missions = json.load(f)['missions']

    plot_overview(missions, os.path.join(script_dir, 'trajectory_overview.png'))
    plot_timeseries(missions, os.path.join(script_dir, 'trajectory_timeseries.png'))
    plot_formation_distance(missions, os.path.join(script_dir, 'trajectory_formation_dist.png'))


if __name__ == '__main__':
    main()
