"""
Publication-Quality Figures for UAV Multi-Agent Debate Paper
=============================================================
生成三张可直接投稿的论文图：

  Fig.1  trajectory_paper.pdf/.png
         8 种对抗场景的编队相对坐标俯视图（2×4 子图，带子图标号 (a)-(h)）

  Fig.2  formation_dist_paper.pdf/.png
         逐帧最小编队间距时序图
         灰虚线=FW均值感知，红区=FW低估风险区，蓝区=FW高估风险区

  Fig.3  timeseries_paper.pdf/.png
         精选3个最有代表性场景的高度+速度时序对比（T3/T7/T6）

排版规范：
  - 白色背景，Times New Roman 字体
  - 色盲友好调色板（IBM accessible）
  - 300 DPI，同时输出 PDF 矢量版
  - 图例简洁，轴标签完整，子图标号 (a)(b)...
  - 关键事件标注采用学术风格（无装饰框，细箭头）
"""

import json, math, os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
from matplotlib import rcParams

# ─────────────────────────────────────────────────────
#  全局字体与样式设置
# ─────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "Times New Roman",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        300,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "lines.linewidth":   1.2,
    "pdf.fonttype":      42,    # TrueType in PDF（可编辑）
    "ps.fonttype":       42,
})

# IBM 色盲友好调色板（UAV_1~4）
DRONE_COLORS = ["#648FFF", "#FE6100", "#785EF0", "#DC267F"]
# GT 标签颜色（深色系，适合白底）
GT_COLOR = {"Risky": "#d62728", "Borderline": "#ff7f0e", "Safe": "#2ca02c"}
# 连续色谱（用于时序）
CMAP_TIME = plt.cm.viridis

SCENARIO_LABELS = {
    "T1_DECEPTIVE_AVGS":  ("T1", "Deceptive\nAverages",    "Risky",      "Borderline"),
    "T2_SYNC_SENSOR":     ("T2", "Sync GPS\nFailure",       "Risky",      "Safe"),
    "T3_ZIGZAG_SEARCH":   ("T3", "Zigzag\nLawnmower",       "Safe",       "Risky"),
    "T4_CASCADE_FAILURE": ("T4", "Cascade\nFailure",        "Borderline", "Safe"),
    "T5_SPLIT_FORMATION": ("T5", "Split\nFormation",        "Borderline", "Safe"),
    "T6_ENDGAME_CRISIS":  ("T6", "Endgame\nCrisis",         "Risky",      "Safe"),
    "T7_GHOST_SENSOR":    ("T7", "Ghost Sensor\nAttack",    "Risky",      "Safe"),
    "T8_NEAR_MISS":       ("T8", "Near-Miss\nRecovery",     "Borderline", "Safe"),
}

SUBFIG_LABELS = list("abcdefgh")


# ─────────────────────────────────────────────────────
#  坐标工具
# ─────────────────────────────────────────────────────
def latlon_to_xy(traj, lat0=None, lon0=None):
    if lat0 is None: lat0 = traj[0]["latitude"]
    if lon0 is None: lon0 = traj[0]["longitude"]
    xs = [(p["longitude"] - lon0) * 111320 * math.cos(math.radians(lat0)) for p in traj]
    ys = [(p["latitude"]  - lat0) * 110540 for p in traj]
    return np.array(xs), np.array(ys)


def to_relative(drones):
    """所有无人机转为相对编队质心坐标（米）"""
    lat0 = drones[0]["trajectory"][0]["latitude"]
    lon0 = drones[0]["trajectory"][0]["longitude"]
    raw = [latlon_to_xy(d["trajectory"], lat0, lon0) for d in drones]
    cx  = np.mean([r[0] for r in raw], axis=0)
    cy  = np.mean([r[1] for r in raw], axis=0)
    return [(r[0]-cx, r[1]-cy, d["trajectory"]) for r, d in zip(raw, drones)], cx, cy


def abs_xy(drones):
    """绝对坐标（用于 T5 分裂编队）"""
    lat0 = drones[0]["trajectory"][0]["latitude"]
    lon0 = drones[0]["trajectory"][0]["longitude"]
    return [(latlon_to_xy(d["trajectory"], lat0, lon0)[0],
             latlon_to_xy(d["trajectory"], lat0, lon0)[1],
             d["trajectory"]) for d in drones]


def min_dist_series(drones):
    """逐帧最小编队间距（米）"""
    lat0 = drones[0]["trajectory"][0]["latitude"]
    lon0 = drones[0]["trajectory"][0]["longitude"]
    pts  = [np.column_stack(latlon_to_xy(d["trajectory"], lat0, lon0)) for d in drones]
    n    = len(pts[0])
    mins = []
    for t in range(n):
        pos   = [pts[i][t] for i in range(len(pts))]
        d_min = min(np.linalg.norm(pos[i]-pos[j])
                    for i, j in itertools.combinations(range(len(pos)), 2))
        mins.append(d_min)
    return np.array(mins)


# ─────────────────────────────────────────────────────
#  Figure 1: Trajectory Overview
# ─────────────────────────────────────────────────────
def plot_trajectory_overview(missions, out_stem):
    ordered = [
        "T1_DECEPTIVE_AVGS", "T2_SYNC_SENSOR",      "T3_ZIGZAG_SEARCH",  "T4_CASCADE_FAILURE",
        "T5_SPLIT_FORMATION","T6_ENDGAME_CRISIS",    "T7_GHOST_SENSOR",   "T8_NEAR_MISS",
    ]
    type_first = {('_'.join(m['mission_id'].split('_')[2:5])): m for m in reversed(missions)}

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8))  # 双栏宽度：7.2英寸
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for idx, mtype in enumerate(ordered):
        ax  = axes[idx // 4][idx % 4]
        m   = type_first.get(mtype)
        lbl = SCENARIO_LABELS.get(mtype, (f"T{idx+1}", mtype, "?", "?"))
        tag_id, scenario_name, gt_label, fw_label = lbl
        gtc = GT_COLOR.get(gt_label, "black")

        if m is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue

        drones = m["drones"]
        n_pts  = len(drones[0]["trajectory"])

        if mtype == "T5_SPLIT_FORMATION":
            drone_data = abs_xy(drones)
            ax.set_xlabel("East (m)", fontsize=7.5)
        else:
            drone_data, _, _ = to_relative(drones)
            ax.set_xlabel("Rel. East (m)", fontsize=7.5)
        ax.set_ylabel("Rel. North (m)", fontsize=7.5)

        # 收集坐标范围
        all_x = np.concatenate([d[0] for d in drone_data])
        all_y = np.concatenate([d[1] for d in drone_data])
        xpad = (all_x.max()-all_x.min()) * 0.18 + 1
        ypad = (all_y.max()-all_y.min()) * 0.18 + 1
        ax.set_xlim(all_x.min()-xpad, all_x.max()+xpad)
        ax.set_ylim(all_y.min()-ypad, all_y.max()+ypad)
        xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0]

        # 轨迹（时间渐变不透明度）
        for d_idx, (xs, ys, traj) in enumerate(drone_data):
            dc = DRONE_COLORS[d_idx % 4]
            # 绘制整条轨迹（由浅入深）
            for i in range(n_pts-1):
                alpha = 0.2 + 0.8 * (i/(n_pts-2))
                ax.plot(xs[i:i+2], ys[i:i+2], color=dc, alpha=alpha, lw=0.9, solid_capstyle="round")
            # 起点(三角)和终点(圆)
            ax.plot(xs[0],  ys[0],  "^", color=dc, ms=4, markeredgewidth=0.4,
                    markeredgecolor="white", zorder=6)
            ax.plot(xs[-1], ys[-1], "o", color=dc, ms=4, markeredgewidth=0.4,
                    markeredgecolor="white", zorder=6)

        xs0, ys0, _ = drone_data[0]
        xs1, ys1, _ = drone_data[1]

        # ── 场景关键标注 ──
        ann_kw = dict(fontsize=6.5, xycoords="data", textcoords="data",
                      arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.7, mutation_scale=6),
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#aaa",
                                alpha=0.9, linewidth=0.6))

        if mtype == "T1_DECEPTIVE_AVGS":
            ax.plot(xs1[75], ys1[75], "*", color="#d62728", ms=8, zorder=10, markeredgewidth=0.3,
                    markeredgecolor="black", label="collision t=75")
            ax.annotate("0.1 m\ncollision\nt=75",
                        xy=(xs1[75], ys1[75]),
                        xytext=(xs1[75]+xspan*0.2, ys1[75]+yspan*0.2), **ann_kw)

        elif mtype == "T2_SYNC_SENSOR":
            for d_idx, (xs, ys, _) in enumerate(drone_data):
                ax.plot(xs[50:54], ys[50:54], color="#d62728", lw=2.0, zorder=7, alpha=0.9)
            ax.annotate("SYNC drift\nt=50–53\n4/4 UAVs",
                        xy=(drone_data[0][0][51], drone_data[0][1][51]),
                        xytext=(drone_data[0][0][51]+xspan*0.18, drone_data[0][1][51]+yspan*0.2), **ann_kw)

        elif mtype == "T3_ZIGZAG_SEARCH":
            ax.text(0.5, 0.05,
                    r"$v \equiv 15.0\ \mathrm{m/s}$, $h \equiv 100.0\ \mathrm{m}$""\n(controlled maneuver)",
                    transform=ax.transAxes, fontsize=6.5, ha="center", va="bottom",
                    color="#2ca02c",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#2ca02c", alpha=0.9, linewidth=0.7))

        elif mtype == "T4_CASCADE_FAILURE":
            xs2, ys2, _ = drone_data[2]
            ax.plot(xs2[30:33], ys2[30:33], color="#d62728", lw=2.0, zorder=7)
            ax.annotate("GPS drift\nt=30–32\n(3 pts)",
                        xy=(xs2[31], ys2[31]),
                        xytext=(xs2[31]-xspan*0.22, ys2[31]+yspan*0.15), **ann_kw)

        elif mtype == "T5_SPLIT_FORMATION":
            t_mid = n_pts // 2
            ax.annotate("", xy=(drone_data[2][0][t_mid], drone_data[2][1][t_mid]),
                        xytext=(xs0[t_mid], ys0[t_mid]),
                        arrowprops=dict(arrowstyle="<->", color="#555", lw=1.0))
            mx = (xs0[t_mid]+drone_data[2][0][t_mid])/2
            my = (ys0[t_mid]+drone_data[2][1][t_mid])/2
            ax.text(mx+xspan*0.05, my, "~310 m", fontsize=7, color="#555")

        elif mtype == "T6_ENDGAME_CRISIS":
            ax.plot(xs1[99], ys1[99], "X", color="#d62728", ms=7, zorder=10,
                    markeredgewidth=0.4, markeredgecolor="black")
            ax.annotate("12 m gap\nt=99",
                        xy=(xs1[99], ys1[99]),
                        xytext=(xs1[99]-xspan*0.22, ys1[99]+yspan*0.22), **ann_kw)

        elif mtype == "T7_GHOST_SENSOR":
            ax.plot(xs1[25:56], ys1[25:56], color="#d62728", lw=2.0, zorder=7, alpha=0.85)
            ax.annotate("heading=270°W\nGPS→East\n(t=25–55)",
                        xy=(xs1[40], ys1[40]),
                        xytext=(xs1[40]+xspan*0.18, ys1[40]-yspan*0.25), **ann_kw)

        elif mtype == "T8_NEAR_MISS":
            ax.plot([xs0[47], xs1[47]], [ys0[47], ys1[47]], "-", color="#d62728", lw=1.8, zorder=7)
            mx = (xs0[47]+xs1[47])/2
            my = (ys0[47]+ys1[47])/2
            ax.text(mx+xspan*0.03, my+yspan*0.08, "12 m\nt=47", fontsize=6.5,
                    color="#d62728", ha="left")
            ax.plot(xs1[48:57], ys1[48:57], "--", color=DRONE_COLORS[1],
                    lw=1.5, zorder=7, alpha=0.9)

        # 子图标号 (a)(b)...
        ax.text(-0.14, 1.05, f"({SUBFIG_LABELS[idx]})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

        # 场景名称（居中顶部）
        ax.set_title(scenario_name, fontsize=8, pad=3, linespacing=1.3)

        # GT / FW 标签（右上角两行）
        ax.text(0.99, 0.99, f"GT: {gt_label}",
                transform=ax.transAxes, fontsize=7, va="top", ha="right",
                color=gtc, fontweight="bold")
        ax.text(0.99, 0.87, f"FW: {fw_label} [X]",
                transform=ax.transAxes, fontsize=7, va="top", ha="right",
                color="#888")
        ax.grid(True, linewidth=0.35, linestyle=":", color="#ccc", alpha=0.8)

    # 全局图例（无人机编号）
    patches = [mlines.Line2D([], [], color=DRONE_COLORS[i], lw=1.5, label=f"UAV {i+1}")
               for i in range(4)]
    patches += [
        mlines.Line2D([], [], marker="^", color="gray", ms=5, lw=0, label="Start"),
        mlines.Line2D([], [], marker="o", color="gray", ms=5, lw=0, label="End"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               fontsize=8, frameon=True, framealpha=0.9, edgecolor="#ccc",
               bbox_to_anchor=(0.5, -0.02))

    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_stem}.pdf",           bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_stem}.png / .pdf")


# ─────────────────────────────────────────────────────
#  Figure 2: Per-Frame Min Formation Distance
# ─────────────────────────────────────────────────────
def plot_formation_distance(missions, out_stem):
    ordered = [
        "T1_DECEPTIVE_AVGS", "T2_SYNC_SENSOR",      "T3_ZIGZAG_SEARCH",  "T4_CASCADE_FAILURE",
        "T5_SPLIT_FORMATION","T6_ENDGAME_CRISIS",    "T7_GHOST_SENSOR",   "T8_NEAR_MISS",
    ]
    type_first = {('_'.join(m['mission_id'].split('_')[2:5])): m for m in reversed(missions)}

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6))
    fig.subplots_adjust(hspace=0.55, wspace=0.38)

    for idx, mtype in enumerate(ordered):
        ax  = axes[idx // 4][idx % 4]
        m   = type_first.get(mtype)
        lbl = SCENARIO_LABELS.get(mtype, (f"T{idx+1}", mtype, "?", "?"))
        tag_id, scenario_name, gt_label, fw_label = lbl
        gtc = GT_COLOR.get(gt_label, "black")

        if m is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue

        dists = min_dist_series(m["drones"])
        ts    = np.arange(len(dists))
        fw_avg = dists.mean()

        # 绘制曲线（颜色按危险程度分段）
        seg_colors = ["#d62728" if d < 0.5 else "#ff7f0e" if d < 20 else "#1f77b4"
                      for d in dists]
        for i in range(len(ts)-1):
            ax.plot(ts[i:i+2], dists[i:i+2], color=seg_colors[i], lw=1.0, alpha=0.9)

        # FW 均值感知（虚线）
        ax.axhline(fw_avg, color="#555", linestyle="--", lw=0.9, alpha=0.85,
                   label=f"FW avg = {fw_avg:.0f} m")

        # 填色：红=FW低估风险，蓝=FW高估
        ax.fill_between(ts, dists, fw_avg,
                        where=(dists < fw_avg), alpha=0.18, color="#d62728",
                        label="FW underestimates risk")
        ax.fill_between(ts, dists, fw_avg,
                        where=(dists > fw_avg), alpha=0.10, color="#1f77b4",
                        label="FW overestimates risk")

        # 安全阈值线
        ax.axhline(20,  color="#ff7f0e", linestyle=":", lw=1.0, alpha=0.8)
        ax.axhline(0.5, color="#d62728", linestyle=":", lw=1.0, alpha=0.8)

        # 子图标号
        ax.text(-0.16, 1.07, f"({SUBFIG_LABELS[idx]})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

        ax.set_title(f"{tag_id}: {scenario_name.replace(chr(10),' ')}",
                     fontsize=7.5, pad=3)
        ax.text(0.99, 0.98, f"GT: {gt_label}",
                transform=ax.transAxes, fontsize=7, va="top", ha="right",
                color=gtc, fontweight="bold")
        ax.text(0.99, 0.86, f"FW: {fw_label} [X]",
                transform=ax.transAxes, fontsize=7, va="top", ha="right", color="#888")

        ax.set_xlabel("Time step", fontsize=7.5)
        ax.set_ylabel("Min. dist. (m)", fontsize=7.5)
        ax.set_xlim(0, len(ts)-1)
        ax.set_ylim(bottom=0)
        ax.grid(True, linewidth=0.35, linestyle=":", color="#ccc", alpha=0.8)

    # 公共图例
    leg_elements = [
        mlines.Line2D([], [], color="#1f77b4", lw=1.2, label="Separation > 20 m (safe)"),
        mlines.Line2D([], [], color="#ff7f0e", lw=1.2, label="Separation < 20 m (warning)"),
        mlines.Line2D([], [], color="#d62728", lw=1.2, label="Separation < 0.5 m (collision)"),
        mlines.Line2D([], [], color="#555",    lw=1.0, linestyle="--", label="FW avg. estimate"),
        mpatches.Patch(color="#d62728", alpha=0.25, label="FW underestimates risk"),
        mlines.Line2D([], [], color="#ff7f0e", lw=1.0, linestyle=":", label="20 m safety threshold"),
    ]
    fig.legend(handles=leg_elements, loc="lower center", ncol=3,
               fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="#ccc",
               bbox_to_anchor=(0.5, -0.07))

    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_stem}.pdf",           bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_stem}.png / .pdf")


# ─────────────────────────────────────────────────────
#  Figure 3: Representative Time-Series (T3 / T6 / T7)
# ─────────────────────────────────────────────────────
def plot_representative_timeseries(missions, out_stem):
    """
    精选三个最有代表性的场景各画一列（高度+速度），共6个子图（3列×2行）：
      T3 ZIGZAG    - 速度/高度完全锁定（FW 误判 Safe→Risky 的根因）
      T6 ENDGAME   - 末段收敛趋势（FW 用均值无法感知）
      T7 GHOST     - GPS 与 Heading 物理矛盾
    """
    target = {
        "T3_ZIGZAG_SEARCH":  ("T3: Zigzag Lawnmower Search",  "Safe",       "Risky"),
        "T6_ENDGAME_CRISIS": ("T6: Endgame Crisis",            "Risky",      "Safe"),
        "T7_GHOST_SENSOR":   ("T7: Ghost Sensor Attack",       "Risky",      "Safe"),
    }
    type_first = {}
    for m in missions:
        mtype = '_'.join(m['mission_id'].split('_')[2:5])
        if mtype not in type_first:
            type_first[mtype] = m

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.6), sharex=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.38)

    col_subfig = ["a", "b", "c"]

    for col_idx, (mtype, (title, gt, fw)) in enumerate(target.items()):
        ax_alt = axes[0][col_idx]
        ax_spd = axes[1][col_idx]
        m = type_first.get(mtype)
        gtc = GT_COLOR.get(gt, "black")

        if m is None:
            continue

        drones = m["drones"]
        n_pts  = len(drones[0]["trajectory"])

        for d_idx, drone in enumerate(drones):
            traj = drone["trajectory"]
            ts   = [p["time"] for p in traj]
            alts = [p["altitude"] for p in traj]
            spds = [p["speed"] for p in traj]
            dc   = DRONE_COLORS[d_idx % 4]
            lw_v = 1.4 if mtype == "T7_GHOST_SENSOR" and d_idx == 1 else 0.9
            ax_alt.plot(ts, alts, color=dc, lw=lw_v, alpha=0.85, label=f"UAV {d_idx+1}")
            ax_spd.plot(ts, spds, color=dc, lw=lw_v, alpha=0.85)

            # GPS漂移点标记
            drift = [i for i, p in enumerate(traj) if p.get("gps_status") != "OK"]
            if drift:
                ax_alt.scatter([ts[i] for i in drift], [alts[i] for i in drift],
                               c="#d62728", s=8, zorder=6, alpha=0.85)
                ax_spd.scatter([ts[i] for i in drift], [spds[i] for i in drift],
                               c="#d62728", s=8, zorder=6, alpha=0.85)

        # 专项标注
        if mtype == "T3_ZIGZAG_SEARCH":
            ax_spd.text(0.5, 0.5,
                        r"$v \equiv 15.0\ \mathrm{m/s}$" + "\n(zero variance)",
                        transform=ax_spd.transAxes, fontsize=8, ha="center", va="center",
                        color="#2ca02c",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="#2ca02c", alpha=0.9, lw=0.8))
            ax_alt.text(0.5, 0.5,
                        r"$h \equiv 100.0\ \mathrm{m}$" + "\n(zero variance)",
                        transform=ax_alt.transAxes, fontsize=8, ha="center", va="center",
                        color="#2ca02c",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="#2ca02c", alpha=0.9, lw=0.8))

        elif mtype == "T6_ENDGAME_CRISIS":
            for ax_ in [ax_alt, ax_spd]:
                ax_.axvspan(80, n_pts-1, alpha=0.08, color="#d62728")
                ax_.axvline(80, color="#d62728", lw=0.8, linestyle="--", alpha=0.7)
            ax_spd.text(82, ax_spd.get_ylim()[0] * 1.01 if ax_spd.get_ylim()[0] > 0
                        else ax_spd.get_ylim()[1] * 0.02,
                        "converge\nphase", fontsize=6.5, color="#d62728")

        elif mtype == "T7_GHOST_SENSOR":
            for ax_ in [ax_alt, ax_spd]:
                ax_.axvspan(25, 55, alpha=0.10, color="#9467bd")
            ax_spd.text(26, ax_spd.get_ylim()[0],
                        "heading vs. GPS\n(t=25–55)", fontsize=6.5, color="#9467bd",
                        va="bottom")

        # 子图标号、标题、轴标
        ax_alt.text(-0.16, 1.1, f"({col_subfig[col_idx]})", transform=ax_alt.transAxes,
                    fontsize=10, fontweight="bold", va="top")
        ax_alt.set_title(title, fontsize=8, pad=3)
        ax_alt.text(0.99, 0.99, f"GT: {gt}", transform=ax_alt.transAxes,
                    fontsize=7, va="top", ha="right", color=gtc, fontweight="bold")
        ax_alt.text(0.99, 0.87, f"FW: {fw} [X]", transform=ax_alt.transAxes,
                    fontsize=7, va="top", ha="right", color="#888")

        ax_alt.set_ylabel("Altitude (m)", fontsize=8)
        ax_spd.set_ylabel("Speed (m/s)", fontsize=8)
        ax_spd.set_xlabel("Time (s)", fontsize=8)

        for ax_ in [ax_alt, ax_spd]:
            ax_.grid(True, linewidth=0.35, linestyle=":", color="#ccc", alpha=0.8)
            # 禁用偏移记法，显示完整数值（避免 +1e2 这种写法）
            ax_.ticklabel_format(useOffset=False, axis="y")

        # 第一列显示图例
        if col_idx == 0:
            ax_alt.legend(fontsize=7, ncol=2, frameon=True, framealpha=0.9,
                          edgecolor="#ccc", loc="upper right")

    # 公共注释：红点=GPS fault
    fig.text(0.5, -0.02,
             r"$\bullet$" + "  Red markers: GPS fault points",
             ha="center", fontsize=7.5, color="#d62728")

    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_stem}.pdf",           bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_stem}.png / .pdf")


# ─────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(script_dir, "hard_uav_missions.json")

    with open(data_path, encoding="utf-8") as f:
        missions = json.load(f)["missions"]

    stems = {
        "trajectory":   os.path.join(script_dir, "fig1_trajectory"),
        "formation":    os.path.join(script_dir, "fig2_formation_dist"),
        "timeseries":   os.path.join(script_dir, "fig3_timeseries"),
    }

    plot_trajectory_overview(missions,          stems["trajectory"])
    plot_formation_distance(missions,           stems["formation"])
    plot_representative_timeseries(missions,    stems["timeseries"])

    print("\nAll figures saved (PNG + PDF vector):")
    for k, v in stems.items():
        print(f"  {v}.png")
        print(f"  {v}.pdf")


if __name__ == "__main__":
    main()
