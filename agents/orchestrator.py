"""
协调 Agent（Orchestrator）。

星型拓扑的中心节点。
路由: trajectory_agent → interaction_agent → summary_agent → FINAL_ANSWER
输出: scene_report
"""

import uuid
from datetime import datetime
from typing import Any

from .base_agent import BaseAgent
from .llm_client import LLMClient


class MockKGAPI:
    """KG API Mock。等 kg/ 层完成后替换。"""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []

    def query_nodes(self, node_type: str, filters: dict) -> list[dict]:
        results = []
        for nid, n in self.nodes.items():
            if n.get("node_type") != node_type:
                continue
            if all(n.get("properties", {}).get(k) == v for k, v in filters.items()):
                results.append({"node_id": nid, **n})
        return results

    def query_edges(self, edge_type: str, filters: dict) -> list[dict]:
        return [e for e in self.edges if e.get("type") == edge_type]

    def get_subgraph(self, node_ids: list[str], depth: int = 1) -> dict:
        return {"nodes": {}, "edges": []}

    def match_pattern(self, pattern_graph) -> list[dict]:
        return []

    def upsert_node(self, node_id: str, node_type: str, properties: dict):
        if node_id in self.nodes:
            self.nodes[node_id]["properties"].update(properties)
        else:
            self.nodes[node_id] = {"node_type": node_type, "properties": dict(properties)}

    def upsert_edge(self, src: str, tgt: str, edge_type: str, properties: dict):
        self.edges.append({"from": src, "to": tgt, "type": edge_type, "properties": dict(properties)})

    def evict_before(self, timestamp: float) -> int:
        return 0

    def set_ttl(self, node_id: str, ttl_seconds: float):
        pass


class Orchestrator:
    """协调 Agent。星型拓扑中心节点。

    固定路由: trajectory → interaction → summary → report
    """

    ROUTE_ORDER = ["trajectory_agent", "interaction_agent", "summary_agent"]

    def __init__(self, llm_client: LLMClient, kg_api=None):
        self.llm = llm_client
        self.kg = kg_api if kg_api is not None else MockKGAPI()
        self.agents: dict[str, BaseAgent] = {}
        # 实际路由顺序。默认与 ROUTE_ORDER 一致；注册可选 Agent 时可按位置插入
        # （例如 anomaly_agent 插在 summary_agent 之前，让总结能看到异常评分）。
        self.route_order: list[str] = list(self.ROUTE_ORDER)

        self.event: dict = {}
        self.responses: dict[str, dict] = {}
        self.route_history: list[str] = []
        self._trail: list[dict] = []

    def register_agent(self, agent: BaseAgent, position: int | None = None):
        """注册 Agent。

        Args:
            agent: 分析 Agent。
            position: 插入 route_order 的位置；None 表示追加到末尾。
                可选 Agent（如 anomaly_agent）用 position=2 插到 summary_agent 之前。
        """
        self.agents[agent.name] = agent
        if agent.name in self.route_order:
            return
        if position is None:
            self.route_order.append(agent.name)
        else:
            self.route_order.insert(position, agent.name)

    def process_event(self, event: dict) -> dict:
        """处理一个 SpatioTemporalEvent，返回 scene_report。"""
        self.event = event
        self.responses = {}
        self.route_history = []
        self._trail = []

        scene_context = self._query_scene_context(event)

        for agent_name in self.route_order:
            if agent_name not in self.agents:
                continue
            context = self._build_context(agent_name, scene_context)
            agent = self.agents[agent_name]
            agent.observe(context)

            # 记录输入摘要 (在实际调用前)
            input_summary = self._summarize_input(agent_name, context)

            response = agent.act()
            self.responses[agent_name] = response
            self.route_history.append(agent_name)

            # 记录输出摘要
            output_summary = self._summarize_output(agent_name, response)

            self._trail.append({
                "agent": agent_name,
                "input_summary": input_summary,
                "output_summary": output_summary,
            })

        report = self._build_scene_report()
        self._execute_kg_ops(report)
        return report

    def _summarize_input(self, agent_name: str, context: dict) -> str:
        n_tracklets = len(context.get("tracklets", []))
        n_relations = len(context.get("spatial_relations", []))
        n_previous = len(context.get("previous_findings", []))
        scene = context.get("scene_type", "?")
        parts = [f"收到 {n_tracklets} 个目标轨迹, {n_relations} 条空间关系, 场景={scene}"]
        if n_previous:
            sources = [p.get("source", "?") for p in context.get("previous_findings", [])]
            parts.append(f"前序Agent发现来自: {', '.join(sources)}")
        return " | ".join(parts)

    def _summarize_output(self, agent_name: str, response: dict) -> str:
        findings = response.get("findings", [])
        conf = response.get("agent_confidence", 0)
        warnings = response.get("warnings", [])
        parts = [f"产出 {len(findings)} 条发现, 全局置信度={conf:.2f}"]
        if warnings:
            parts.append(f"警告: {'; '.join(warnings[:3])}")
        # 每条发现的首句
        for i, f in enumerate(findings[:5]):
            val = f.get("value", "")
            if isinstance(val, dict):
                lbl = val.get("label", val.get("description", str(val)))
                short = f"{lbl} spd={val.get('speed_px','?')} pct={val.get('speed_pct','?')}"
            else:
                short = str(val)[:100].replace("\n", " ")
            parts.append(f"  [{f.get('finding_type', '?')}] {short}")
        return "\n".join(parts)

    def _build_scene_report(self) -> dict:
        """从三个 Agent 的响应构建场景报告。"""
        traj = self.responses.get("trajectory_agent", {})
        inter = self.responses.get("interaction_agent", {})
        summary = self.responses.get("summary_agent", {})

        # 场景摘要
        scene_text = ""
        for f in summary.get("findings", []):
            if f.get("finding_type") == "scene_summary":
                scene_text = f.get("value", "")
                break

        # 运动描述
        motion_descriptions = [
            {"entity_id": f.get("entity_id", ""), "description": f.get("value", ""),
             "confidence": f.get("confidence", 0)}
            for f in traj.get("findings", [])
            if f.get("finding_type") == "motion_description"
        ]

        # 交互描述
        interaction_descriptions = [
            {"entities": [f.get("entity_id", "")] + f.get("related_entities", []),
             "description": f.get("value", ""), "confidence": f.get("confidence", 0)}
            for f in inter.get("findings", [])
            if "interaction" in f.get("finding_type", "") or "group" in f.get("finding_type", "")
        ]

        # 收集所有 kg_operations
        kg_updates: list[dict] = []
        for resp in self.responses.values():
            for finding in resp.get("findings", []):
                kg_updates.extend(finding.get("kg_operations", []))

        return {
            "output_id": f"out-{uuid.uuid4().hex[:8]}",
            "reasoning_trail": self._trail,
            "timestamp": datetime.now().timestamp(),
            "event_id": self.event.get("event_id", ""),
            "duration": self.event.get("duration", []),
            "frame_range": [self.event.get("frame_start", 0), self.event.get("frame_end", 0)],
            "scene_report": {
                "summary": scene_text,
                "entity_count": len(motion_descriptions),
                "motion_descriptions": motion_descriptions,
                "interaction_descriptions": interaction_descriptions,
            },
            "agent_summary": {
                "trajectory_agent_confidence": traj.get("agent_confidence", 0),
                "trajectory_agent_warnings": traj.get("warnings", []),
                "interaction_agent_confidence": inter.get("agent_confidence", 0),
                "interaction_agent_warnings": inter.get("warnings", []),
                "summary_agent_confidence": summary.get("agent_confidence", 0),
                "anomaly_agent_confidence": self.responses.get("anomaly_agent", {}).get(
                    "agent_confidence", 0
                ),
            },
            "kg_updates": kg_updates,
        }

    def _execute_kg_ops(self, report: dict):
        for op in report.get("kg_updates", []):
            try:
                if op.get("op") == "upsert_node":
                    self.kg.upsert_node(
                        node_id=op.get("node_id", ""),
                        node_type=op.get("node_type", ""),
                        properties=op.get("properties", {}),
                    )
                elif op.get("op") == "upsert_edge":
                    self.kg.upsert_edge(
                        src=op.get("from", op.get("src", "")),
                        tgt=op.get("to", op.get("tgt", "")),
                        edge_type=op.get("edge_type", op.get("type", "")),
                        properties=op.get("properties", {}),
                    )
            except Exception:
                pass

    def _query_scene_context(self, event: dict) -> dict:
        return {
            "scene": {},
            "behaviors": [],
            "zone_id": event.get("zone_id", ""),
            "zone_type": event.get("zone_type", "unknown"),
        }

    def _build_context(self, agent_name: str, scene_context: dict) -> dict:
        context: dict[str, Any] = {
            "event_ids": [self.event.get("event_id", "")],
            "tracklets": self.event.get("tracklets", []),
            "spatial_relations": self.event.get("spatial_relations", []),
            "scene_type": scene_context.get("zone_type", "unknown"),
            "zone_id": self.event.get("zone_id", ""),
            "previous_findings": [],
        }

        for name, resp in self.responses.items():
            context["previous_findings"].append({"source": name, "finding": resp})

        # 完整的结构化响应，供 anomaly_agent 直接读取轨迹/交互的 findings
        context["all_agent_responses"] = dict(self.responses)

        return context
