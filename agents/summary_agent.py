"""
场景总结 Agent。

接收轨迹分析 Agent 和交互分析 Agent 的发现，
生成自然语言场景整体描述。
"""

import time
from typing import Any

from .base_agent import BaseAgent
from .llm_client import LLMClient


class SummaryAgent(BaseAgent):
    """生成场景级别的自然语言摘要。

    输入：前两个 Agent 的 findings（motion_description + interaction_description）
    输出：场景整体描述
    """

    def __init__(self, llm_client: LLMClient) -> None:
        prompt = self._load_prompt("summary")
        super().__init__("summary_agent", llm_client, prompt)

    def think(self) -> dict:
        """综合前两个 Agent 的发现，生成场景摘要。

        Returns:
            {
                "agent_confidence": float,
                "warnings": list[str],
                "findings": list[{
                    "entity_id": "",
                    "finding_type": "scene_summary",
                    "value": str,  # 自然语言场景摘要
                    "confidence": float,
                    "evidence": list[str],
                    "kg_operations": list[dict]
                }]
            }
        """
        previous = self.context.get("previous_findings", [])
        tracklets = self.context.get("tracklets", [])
        scene_type = self.context.get("scene_type", "unknown")

        # 收集结构化运动数据 + 交互描述
        motion_labels: dict[str, int] = {}
        pose_labels: dict[str, int] = {}
        motion_details: list[dict] = []
        interaction_descs: list[str] = []
        all_evidence: list[str] = []

        for pf in previous:
            for f in pf.get("finding", {}).get("findings", []):
                if not isinstance(f, dict):
                    continue
                ftype = f.get("finding_type", "")
                val = f.get("value", "")
                if "motion" in ftype and isinstance(val, dict):
                    lbl = val.get("label", "未知")
                    motion_labels[lbl] = motion_labels.get(lbl, 0) + 1
                    pl = val.get("pose", "")
                    if pl:
                        pose_labels[pl] = pose_labels.get(pl, 0) + 1
                    motion_details.append(val)
                    all_evidence.extend(f.get("evidence", []))
                elif "interaction" in ftype or "group" in ftype:
                    desc = val if isinstance(val, str) else val.get("description", str(val))
                    interaction_descs.append(desc)
                    all_evidence.extend(f.get("evidence", []))

        entity_count = len(tracklets)
        time_span = ""
        if tracklets:
            first = tracklets[0].get("trajectory", [[0,0,0]])
            last = tracklets[-1].get("trajectory", [[0,0,0]])
            if len(first[0]) >= 3 and len(last[0]) >= 3:
                t0, t1 = first[0][2], last[0][2]
                time_span = f"，时间跨度 {max(t1-t0, 0):.1f} 秒"

        parts: list[str] = []
        parts.append(f"场景类型 {scene_type}，{entity_count} 个目标{time_span}。")
        if motion_labels:
            dist = "、".join(f"{k} {v}" for k, v in sorted(motion_labels.items(), key=lambda x:-x[1]))
            parts.append(f"运动分布：{dist}。")
        if pose_labels:
            dist = "、".join(f"{k} {v}" for k, v in sorted(pose_labels.items(), key=lambda x:-x[1]))
            parts.append(f"姿态分布：{dist}。")
        if interaction_descs:
            parts.append(f"交互关系 {len(interaction_descs)} 条。")

        summary = " ".join(parts)
        coverage = min(1.0, (len(motion_details) + len(interaction_descs)) / max(entity_count * 1.5, 1))
        confidence = 0.6 + coverage * 0.3

        return {
            "agent_confidence": round(confidence, 3),
            "warnings": [],
            "findings": [{
                "entity_id": "",
                "finding_type": "scene_summary",
                "value": summary,
                "confidence": round(confidence, 3),
                "evidence": all_evidence[:20],  # 限制证据数量
                "kg_operations": [{
                    "op": "upsert_node",
                    "node_type": "Scene",
                    "node_id": f"Scene-{int(time.time())}",
                    "properties": {
                        "scene_type": scene_type,
                        "global_description": summary,
                        "entity_count": entity_count,
                    },
                }],
            }],
        }
