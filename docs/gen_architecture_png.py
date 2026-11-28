"""用 matplotlib 画 SceneSense 分层架构图（中文标注 + 数据流箭头）。

mermaid-cli 在本机装不上时用这个兜底。输出 docs/architecture.png。

用法: python gen_arch_png.py docs/architecture.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/architecture.png")

# (标题, [条目], 填充色, 描边色, 向下箭头标签)；层高按条目数自动算，层间留固定间隙
LAYERS = [
    ("输入", ["摄像头 / 视频文件 / 图片序列"], "#eceff1", "#546e7a", "逐帧 BGR"),
    ("感知层  perception/  （10–30 fps）",
     ["YOLOv8 检测 → bbox / class / conf",
      "YOLOv8-pose → 17 个 COCO 关键点（可选）",
      "Tracker：GMC 稀疏光流 + 8 状态卡尔曼 + OSNet ReID"],
     "#e1f5fe", "#0288d1", "tracklets"),
    ("实体层  perception/entity_profile.py",
     ["tracklet 按 位置 0.35 + 时间 0.30 + 外观 0.35 打分合并成 Entity",
      "≥0.72 自动合并，0.30–0.72 记 tentative；外观余弦 >0.90 时放宽 gap 上限"],
     "#e0f2f1", "#00695c", "entities"),
    ("事件层  perception/event_builder.py",
     ["每 30 帧（--event-interval）切一个 SpatioTemporalEvent",
      "tracklet 去重 + 空间近邻 → spatial_relations"],
     "#fce4ec", "#c62828", "event_batch"),
    ("Agent 层  agents/  （星型编排，事件触发）",
     ["Orchestrator：按 route_order 串行路由，中心持有全部 responses",
      "TrajectoryAgent → InteractionAgent → [AnomalyAgent] → SummaryAgent",
      "Agent 只依赖中心传入的 context；findings 带 kg_operations"],
     "#fff3e0", "#f57c00", "kg_operations"),
    ("KG 层  kg/",
     ["GraphStore：NetworkX DiGraph + GraphML 读写 · Event 节点带 expires_at",
      "节点 Entity / Event / Behavior / Scene",
      "边 belongs_to / interacts_with / matches / next",
      "continuity 写入 · rule_engine 规则检查"],
     "#e8f5e9", "#388e3c", "report"),
    ("输出", ["轨迹 JSON · 中文场景报告 · 推理链 · GraphML 场景图谱"], "#f3e5f5", "#7b1fa2", ""),
]

TOP_GAP = 0.46      # 标题与第一层之间
LAYER_GAP = 0.34    # 层间间隙（放箭头与标签）
ITEM_STEP = 0.27
LAYER_PAD = 0.50


def layout():
    """从顶部往下排布，返回 [(title, items, fc, ec, y, h, label), ...]。"""
    rows = []
    y = None
    for title, items, fc, ec, label in LAYERS:
        h = LAYER_PAD + ITEM_STEP * len(items)
        rows.append([title, items, fc, ec, y, h, label])
        y_tmp = h
        # 先记高度，位置后面统一回填
        rows[-1][4] = y_tmp
        y = y_tmp
    total = sum(r[5] for r in rows) + LAYER_GAP * (len(rows) - 1)
    cursor = total
    for r in rows:
        r[4] = cursor - r[5]
        cursor = r[4] - LAYER_GAP
    return rows, total


def main() -> None:
    rows, total = layout()
    height = total + TOP_GAP + 0.35
    fig, ax = plt.subplots(figsize=(13.5, height * 1.02), dpi=130)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, height)
    ax.axis("off")

    for title, items, fc, ec, y, h, _ in rows:
        ax.add_patch(FancyBboxPatch((0.35, y), 9.0, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    linewidth=2.0, facecolor=fc, edgecolor=ec))
        ax.text(0.62, y + h - 0.13, title, fontsize=13, fontweight="bold", color=ec,
                va="top", ha="left")
        for i, item in enumerate(items):
            ax.text(0.85, y + h - 0.44 - i * ITEM_STEP, "· " + item, fontsize=10,
                    color="#263238", va="top", ha="left")

    # 层间数据流箭头
    for i in range(len(rows) - 1):
        y_bottom = rows[i][4]
        y_next_top = rows[i + 1][4] + rows[i + 1][5]
        label = rows[i][6]
        ax.add_patch(FancyArrowPatch((4.85, y_bottom - 0.01), (4.85, y_next_top + 0.01),
                                     arrowstyle="-|>", mutation_scale=20,
                                     linewidth=2.0, color="#37474f"))
        if label:
            ax.text(5.0, (y_bottom + y_next_top) / 2, label, fontsize=9, color="#37474f",
                    ha="left", va="center")

    # Agent 层 ←→ KG 层 读写旁路
    agent_y = rows[4][4]
    kg_y, kg_h = rows[5][4], rows[5][5]
    ax.add_patch(FancyArrowPatch((9.55, agent_y + 0.05), (9.55, kg_y + kg_h - 0.05),
                                 arrowstyle="<|-|>", mutation_scale=18,
                                 linewidth=1.8, color="#2e7d32",
                                 connectionstyle="arc3,rad=0.0"))
    ax.text(9.74, (agent_y + kg_y + kg_h) / 2, "read / write", fontsize=9, color="#2e7d32",
            rotation=90, va="center", ha="center")

    ax.text(5.0, total + 0.30, "SceneSense 分层架构：感知 → 实体 → 事件 → Agent → KG",
            fontsize=16.5, fontweight="bold", ha="center", va="center", color="#102027")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"written {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
