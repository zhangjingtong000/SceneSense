"""
KG 子图可视化。

用法:
    python scripts/visualize_kg.py --kg kg_snapshot.graphml [--output kg_subgraph.png]
    或从 pipeline 中调用: visualize(graphstore.to_dict(), "output.png")
"""

import argparse

import matplotlib.pyplot as plt
import networkx as nx

# 节点颜色（按类型）
NODE_COLORS = {
    "Entity": "#42A5F5",   # 蓝
    "Event":  "#FFEE58",    # 黄
    "Behavior": "#FF9800",  # 橙
    "Scene":  "#66BB6A",    # 绿
    "unknown": "#BDBDBD",   # 灰
}

# 边样式（按类型）
EDGE_STYLES = {
    "interacts_with": ("solid", "#78909C", 1.0),
    "belongs_to":     ("dotted", "#90A4AE", 0.8),
    "matches":        ("dashed", "#FF7043", 1.5),
    "next":           ("dotted", "#B0BEC5", 0.6),
    "violates":       ("dashed", "#E53935", 2.0),  # 红色粗虚线
    "unknown":        ("solid", "#BDBDBD", 0.5),
}


def visualize(
    data: dict,
    output_path: str = "kg_subgraph.png",
    title: str = "Knowledge Graph",
    figsize: tuple = (16, 12),
    node_size: int = 600,
    font_size: int = 7,
    show_labels: bool = True,
):
    """画 KG 子图。

    Args:
        data: GraphStore.to_dict() 的输出 {"nodes": [...], "edges": [...]}
        或者直接传 NetworkX Graph 对象
        或者 GraphML 文件路径
    """
    if isinstance(data, str):
        data = _load_graphml(data)

    G = _build_nx(data)

    if G.number_of_nodes() == 0:
        print("KG is empty, nothing to visualize.")
        return

    plt.figure(figsize=figsize)

    # 按节点类型分组，分别着色
    node_type_groups: dict[str, list[str]] = {}
    for nid, ndata in G.nodes(data=True):
        nt = ndata.get("node_type", "unknown")
        node_type_groups.setdefault(nt, []).append(nid)

    # 使用 spring layout
    pos = nx.spring_layout(G, k=2.0, iterations=30, seed=42)

    # 画节点
    for ntype, nids in node_type_groups.items():
        color = NODE_COLORS.get(ntype, NODE_COLORS["unknown"])
        nx.draw_networkx_nodes(
            G, pos, nodelist=nids, node_color=color,
            node_size=node_size, alpha=0.9,
            label=f"{ntype} ({len(nids)})"
        )

    # 画边（分类别画不同线型）
    edge_type_groups: dict[str, list[tuple]] = {}
    for src, tgt, edata in G.edges(data=True):
        et = edata.get("edge_type", "unknown")
        edge_type_groups.setdefault(et, []).append((src, tgt))

    for etype, elist in edge_type_groups.items():
        style, color, width = EDGE_STYLES.get(etype, EDGE_STYLES["unknown"])
        nx.draw_networkx_edges(
            G, pos, edgelist=elist,
            style=style, edge_color=color, width=width,
            alpha=0.5, arrows=True if G.is_directed() else False,
        )

    # 标签
    if show_labels:
        labels = {}
        for nid, ndata in G.nodes(data=True):
            nt = ndata.get("node_type", "?")
            short_id = str(nid)[:20]
            if nt == "Entity":
                lbl = ndata.get("motion_label", short_id)
            else:
                lbl = short_id
            labels[nid] = lbl
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size)

    # Legend
    node_legend = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=c, markersize=10, label=t)
                   for t, c in NODE_COLORS.items() if t in node_type_groups]
    edge_legend = [plt.Line2D([0], [0], color=color, linestyle=style, linewidth=width, label=et)
                   for et, (style, color, width) in EDGE_STYLES.items() if et in edge_type_groups]

    legend1 = plt.legend(handles=node_legend, title="Nodes", loc="upper left",
                         bbox_to_anchor=(1.02, 1))
    legend2 = plt.legend(handles=edge_legend, title="Edges", loc="upper left",
                         bbox_to_anchor=(1.02, 0.5))
    plt.gca().add_artist(legend1)

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"KG visualization saved to {output_path}")
    print(f"  nodes={G.number_of_nodes()} edges={G.number_of_edges()}")


def _build_nx(data: dict | nx.Graph) -> nx.Graph:
    """从 to_dict() 输出或 nx.Graph 构建有向图。"""
    if isinstance(data, nx.Graph):
        return data

    G = nx.DiGraph()
    nodes = data.get("nodes", []) if isinstance(data, dict) else []

    for n in nodes:
        nid = n.pop("node_id", str(id(n)))
        props = {k: v for k, v in n.items()}
        G.add_node(nid, **props)

    edges = data.get("edges", []) if isinstance(data, dict) else []
    for e in edges:
        src = e.pop("src", "")
        tgt = e.pop("tgt", "")
        G.add_edge(src, tgt, **e)

    return G


def _load_graphml(path: str) -> nx.Graph:
    return nx.read_graphml(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize KG subgraph")
    parser.add_argument("--kg", required=True, help="GraphML file or JSON dict path")
    parser.add_argument("--output", default="kg_subgraph.png", help="Output PNG path")
    parser.add_argument("--title", default="Knowledge Graph", help="Plot title")
    args = parser.parse_args()

    if args.kg.endswith(".graphml"):
        G = nx.read_graphml(args.kg)
        visualize(G, args.output, args.title)
    elif args.kg.endswith(".json"):
        import json
        with open(args.kg) as f:
            data = json.load(f)
        visualize(data, args.output, args.title)
    else:
        print("Unsupported format. Use .graphml or .json")
