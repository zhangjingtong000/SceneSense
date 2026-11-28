"""
自然语言问答 REPL / KGQA 实现。

用法:
    python scripts/qa_repl.py --kg mot17_pose.graphml [--api-key sk-xxx]

查询逻辑:
    1. 规则匹配（问题模板 → KG 查询）→ 模板拼回复
    2. 匹配不上 → 兜底 LLM 自由回答

实体编号按轨迹号提问（如「3号在干什么」）。KG 里的实体节点当前命名为
`Person-3`，旧版本是 `Entity-3`；两种前缀都能解析，所以 `web/backend/main.py`
的 `/chat` 和命令行 REPL 共用同一套查询逻辑。
"""

import argparse
import os
import re

import networkx as nx


def _edge_type(e: dict) -> str:
    return e.get("edge_type", e.get("type", "?"))


class KGQA:
    def __init__(self, graph: nx.Graph, llm_client=None):
        self.G = graph
        self.llm = llm_client

    def answer(self, question: str) -> str:
        q = question.strip()
        if not q:
            return "请输入问题。"

        # 前端「确认合并」按钮发来的机器指令：合并在前端本地状态里完成，
        # 后端只回一句说明，不为此消耗一次 LLM 调用。
        if q.startswith("__merge__"):
            return "合并确认由前端本地状态维护，KG 文件不落库。"

        # 1. X号在干什么/状态/情况
        m = re.match(r".*?(\d+)\s*号?\s*(在干什么|在做什么|状态|情况|信息).*", q)
        if m:
            return self._entity_status(int(m.group(1)))

        # 2. X号和Y号什么关系
        m = re.match(r".*?(\d+)\s*号?\s*(和|与|跟)\s*(\d+)\s*号?\s*(什么关系|的关系|关系).*", q)
        if m:
            return self._entity_relation(int(m.group(1)), int(m.group(3)))

        # 3. 场景里有多少人/目标
        if re.search(r"(多少|几个|总共|统计).*(人|目标|对象|实体)", q) or re.search(r"场景.*(人|目标|对象|实体)", q):
            return self._scene_stats()

        # 4. 谁在徘徊/跑动/站立/走动
        m = re.search(r"谁在?(徘徊|跑动|奔跑|站立|走动|行走|静止)", q)
        if m:
            return self._who_is(m.group(1))

        # 5. X号附近有谁
        m = re.match(r".*?(\d+)\s*号?\s*(附近|周围|旁边|邻近).*", q)
        if m:
            return self._nearby(int(m.group(1)))

        # 兜底: LLM
        if self.llm:
            return self._llm_fallback(q)
        return "抱歉，我不理解这个问题。试试: '3号在干什么' / '1号和5号什么关系' / '场景里有多少人' / '谁在跑动' / '4号附近有谁'"

    # ── 节点解析 ──

    def _resolve_entity(self, eid: int) -> str | None:
        """把题面里的编号解析成实际节点 ID。

        先试 `Entity-{n}` / `Person-{n}`，再按 `-{n}` 后缀扫一遍 Entity 节点，
        这样以后改命名（如 `Track-3`）也不至于整条问答链失效。
        """
        for candidate in (f"Entity-{eid}", f"Person-{eid}"):
            if candidate in self.G.nodes:
                return candidate
        suffix = f"-{eid}"
        for nid, data in self.G.nodes(data=True):
            if data.get("node_type") == "Entity" and str(nid).endswith(suffix):
                return nid
        return None

    def _entity_edges(self, nid: str):
        """实体的全部关联边。DiGraph 下 `G.edges(nid)` 只给out边，入边要单独取。"""
        seen = set()
        for src, tgt, data in self.G.edges(nid, data=True):
            key = (src, tgt)
            if key not in seen:
                seen.add(key)
                yield src, tgt, data
        if self.G.is_directed():
            for src, tgt, data in self.G.in_edges(nid, data=True):
                key = (src, tgt)
                if key not in seen:
                    seen.add(key)
                    yield src, tgt, data

    # ── 规则查询 ──

    def _entity_status(self, eid: int) -> str:
        nid = self._resolve_entity(eid)
        if nid is None:
            return f"未找到 {eid} 号目标。"

        ndata = self.G.nodes[nid]
        motion = ndata.get("motion_description", ndata.get("motion_profile", "未知"))
        velocity = ndata.get("velocity", ndata.get("velocity_ms", 0))
        cls = ndata.get("class", "person")

        interactions = []
        for src, tgt, edata in self._entity_edges(nid):
            if _edge_type(edata) == "interacts_with":
                other = tgt if src == nid else src
                itype = edata.get("relation_type", "共现")
                dist = edata.get("distance", "?")
                interactions.append(f"与 {other} {itype}（距离 {dist}px）")

        parts = [f"{nid}（{cls}）当前状态: {motion}"]
        if velocity:
            # 无相机标定，速度是像素/秒，不是米/秒
            parts.append(f"速度约 {velocity}px/s（归一化值）")
        if interactions:
            parts.append("交互: " + "; ".join(sorted(set(interactions))))
        else:
            parts.append("无直接交互对象。")
        return "\n".join(parts)

    def _edge_between(self, n1: str, n2: str):
        if self.G.has_edge(n1, n2):
            return self.G.edges[n1, n2]
        if self.G.has_edge(n2, n1):
            return self.G.edges[n2, n1]
        return None

    def _entity_relation(self, eid1: int, eid2: int) -> str:
        n1, n2 = self._resolve_entity(eid1), self._resolve_entity(eid2)
        missing = [str(e) for e, n in ((eid1, n1), (eid2, n2)) if n is None]
        if missing:
            return f"未找到 {'、'.join(missing)} 号目标。"

        edge = self._edge_between(n1, n2)
        if edge is None:
            return f"{n1} 与 {n2} 无直接交互关系。"
        etype = _edge_type(edge)
        dist = edge.get("distance", "?")
        score = edge.get("interaction_score", edge.get("confidence", "?"))
        dur = edge.get("duration", "?")
        return f"{n1} 与 {n2}: {etype}（距离 {dist}px, 共现 {dur}s, 交互评分 {score}）"

    def _scene_stats(self) -> str:
        entities = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "Entity"]
        if not entities:
            return "场景中没有目标。"

        profiles: dict[str, int] = {}
        for nid in entities:
            ndata = self.G.nodes[nid]
            lbl = ndata.get("motion_label", ndata.get("motion_profile", "unknown"))
            profiles[lbl] = profiles.get(lbl, 0) + 1

        parts = [f"场景中共有 {len(entities)} 个目标。"]
        if profiles:
            parts.append("运动模式分布: " + ", ".join(f"{k}:{v}" for k, v in sorted(profiles.items())))
        return "\n".join(parts)

    _MOTION_ALIAS = {
        "徘徊": "loitering", "跑动": "running", "奔跑": "running",
        "站立": "stationary", "静止": "stationary", "走动": "walking",
        "行走": "walking", "跑步": "running",
    }

    def _who_is(self, action: str) -> str:
        target = self._MOTION_ALIAS.get(action, action)
        matches = []
        for nid, ndata in self.G.nodes(data=True):
            if ndata.get("node_type") != "Entity":
                continue
            lbl = str(ndata.get("motion_label", ndata.get("motion_profile", "")))
            desc = str(ndata.get("motion_description", ""))
            if target in lbl.lower() or target in desc.lower():
                matches.append(nid)
        if matches:
            return f"以下目标在{action}: " + ", ".join(matches)
        return f"没有目标在{action}。"

    def _nearby(self, eid: int) -> str:
        nid = self._resolve_entity(eid)
        if nid is None:
            return f"未找到 {eid} 号目标。"

        neighbors: list[str] = []
        for src, tgt, edata in self._entity_edges(nid):
            if _edge_type(edata) == "interacts_with":
                other = tgt if src == nid else src
                dist = edata.get("distance", "?")
                neighbors.append(f"{other}（距离 {dist}px）")

        if neighbors:
            return f"{nid} 附近有: " + ", ".join(sorted(set(neighbors)))
        return f"{nid} 附近没有其他目标。"

    # ── LLM 兜底 ──

    def _llm_fallback(self, question: str) -> str:
        entities = []
        for nid, ndata in self.G.nodes(data=True):
            if ndata.get("node_type") != "Entity":
                continue
            motion = ndata.get("motion_description", ndata.get("motion_profile", "?"))
            entities.append(f"  {nid}: {motion}")

        prompt = f"""你是场景分析助手。以下是从视频分析得到的实体信息：

场景中的目标:
{chr(10).join(entities) if entities else '（无数据）'}

请基于以上信息回答用户问题。如果信息不足，如实说明。
问题: {question}
回答:"""

        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}])
            return resp.strip()
        except Exception as e:
            return f"LLM 调用失败: {e}"


def _maybe_load_llm(api_key: str):
    if not api_key:
        return None
    try:
        from agents.llm_client import LLMClient
        return LLMClient(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="KG 自然语言问答 REPL")
    parser.add_argument("--kg", required=True, help="GraphML 文件路径")
    parser.add_argument("--api-key", default="", help="DeepSeek API Key（兜底 LLM 用）")
    args = parser.parse_args()

    G = nx.read_graphml(args.kg)
    print(f"Loaded KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    llm = _maybe_load_llm(args.api_key or os.getenv("DEEPSEEK_API_KEY", ""))
    qa = KGQA(G, llm)

    print("输入问题（exit 退出）")
    print("-" * 50)

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue

        ans = qa.answer(q)
        print(ans)
        print()


if __name__ == "__main__":
    main()
