"""
异常检测 Agent。

检测异常行为。基于KG中的行为模板做子图匹配。
输出 raw_score（原始子图匹配分），不是最终告警严重度。
"""

from typing import Any

from .base_agent import BaseAgent
from .llm_client import LLMClient


class AnomalyAgent(BaseAgent):
    """检测异常行为。

    基于轨迹发现 + 交互发现 + KG 行为模板做子图匹配。
    输出 raw_score，最终 fused_severity 由协调Agent加权融合。
    """

    # 行为模板关键词定义（模拟从 KG 加载）
    BEHAVIOR_TEMPLATES = [
        {
            "behavior_id": "Behavior-loitering-001",
            "pattern_type": "loitering",
            "severity": 0.7,
            "keywords": [
                "loitering",
                "徘徊",
                "low_speed",
                "direction_change",
                "prolonged_stay",
            ],
            "description": "目标在特定区域长时间徘徊，速度低且方向频繁变化",
        },
        {
            "behavior_id": "Behavior-following-001",
            "pattern_type": "following",
            "severity": 0.8,
            "keywords": [
                "following",
                "跟随",
                "tail",
                "same_path",
                "proximity",
                "sustained",
            ],
            "description": "目标长时间保持与另一目标相近的距离和相同路径",
        },
        {
            "behavior_id": "Behavior-running-001",
            "pattern_type": "running",
            "severity": 0.3,
            "keywords": [
                "running",
                "奔跑",
                "high_speed",
                "urgent",
            ],
            "description": "目标以异常高速移动",
        },
        {
            "behavior_id": "Behavior-erratic-001",
            "pattern_type": "erratic",
            "severity": 0.6,
            "keywords": [
                "erratic",
                " erratic",
                "unstable",
                "direction_change",
                "unpredictable",
            ],
            "description": "目标运动方向频繁无规律变化",
        },
        {
            "behavior_id": "Behavior-isolated-001",
            "pattern_type": "isolated",
            "severity": 0.4,
            "keywords": [
                "isolated",
                "孤独",
                "alone",
                "no_interaction",
                "distant",
            ],
            "description": "目标与其他所有目标距离较远，没有社交互动",
        },
    ]

    def __init__(self, llm_client: LLMClient) -> None:
        # 关键词模板打分，不调模型：没有 system prompt（原 prompts/anomaly.md 已删）
        super().__init__("anomaly_agent", llm_client, "")

    def think(self) -> dict:
        """检测异常行为。

        1. 从场景获取相关行为模板（Behavior节点）
        2. 构建当前实体子图
        3. 与每个模板做图匹配
        4. 输出 raw_score（匹配度），不是最终 severity

        Returns:
            {
                "agent_confidence": float,
                "warnings": list[str],
                "findings": list[dict]
            }
        """
        tracklets = self.context.get("tracklets", [])

        # 轨迹/交互的 findings 直接从结构化响应里取（previous_findings 只是摘要）
        all_responses = self.context.get("all_agent_responses", {})
        traj_response = all_responses.get("trajectory_agent", {})
        inter_response = all_responses.get("interaction_agent", {})

        traj_findings = traj_response.get("findings", [])
        inter_findings = inter_response.get("findings", [])

        # 构建实体子图（按 entity_id 聚合）
        entity_subgraph = self._build_entity_subgraph(
            traj_findings, inter_findings, tracklets
        )

        # 加载行为模板
        templates = self.BEHAVIOR_TEMPLATES

        # 子图匹配
        matched_results = self._match_behavior_templates(
            entity_subgraph, templates
        )

        # 构建 findings
        findings: list[dict] = []
        confidences: list[float] = []
        warnings: list[str] = []

        for result in matched_results:
            entity_id = result["entity_id"]
            matched_template = result["matched_template"]
            raw_score = result["raw_score"]
            evidence = result["evidence"]

            # 只有分数高于阈值才输出
            if raw_score < 0.3:
                continue

            finding = {
                "entity_id": entity_id,
                "finding_type": "anomaly",
                "value": matched_template.get(
                    "pattern_type", "unknown"
                ),
                "confidence": round(raw_score, 3),  # raw_score
                "evidence": evidence,
                "matched_template": matched_template.get(
                    "behavior_id", "unknown"
                ),
                "kg_operations": [
                    {
                        "op": "upsert_edge",
                        "edge_type": "matches",
                        "from": entity_id,
                        "to": matched_template.get(
                            "behavior_id", "unknown"
                        ),
                        "properties": {
                            "confidence": round(raw_score, 3),
                            "match_timestamp": __import__(
                                "time"
                            ).time(),
                        },
                    }
                ],
            }
            findings.append(finding)
            confidences.append(raw_score)

        # Agent 全局置信度
        agent_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.5
        )

        return {
            "agent_confidence": round(agent_confidence, 3),
            "warnings": warnings,
            "findings": findings,
        }

    def _build_entity_subgraph(
        self,
        traj_findings: list[dict],
        inter_findings: list[dict],
        tracklets: list[dict],
    ) -> dict[str, Any]:
        """构建当前实体子图。

        Returns:
            {
                "entities": {entity_id: {"motion": ..., "interactions": [...], ...}},
                "edges": [{"from": ..., "to": ..., "type": ...}]
            }
        """
        entities: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        # 从 tracklets 构建实体基本信息
        for t in tracklets:
            eid = f"Entity-{t.get('track_id', 0)}"
            entities[eid] = {
                "entity_id": eid,
                "class": t.get("class", "unknown"),
                "velocity_ms": t.get("velocity_ms", 0),
                "direction_deg": t.get("direction_deg", 0),
                "centroid": t.get("centroid", [0, 0]),
                "motion": None,
                "interactions": [],
            }

        # 注入轨迹发现
        for f in traj_findings:
            eid = f.get("entity_id", "")
            if eid in entities:
                entities[eid]["motion"] = {
                    "pattern": f.get("value"),
                    "confidence": f.get("confidence"),
                }

        # 注入交互发现
        for f in inter_findings:
            eid = f.get("entity_id", "")
            related = f.get("related_entities", [])
            interaction = {
                "type": f.get("value"),
                "confidence": f.get("confidence"),
                "with": related,
            }
            if eid in entities:
                entities[eid]["interactions"].append(interaction)
            for reid in related:
                if reid not in entities:
                    entities[reid] = {
                        "entity_id": reid,
                        "class": "unknown",
                        "motion": None,
                        "interactions": [],
                    }

        # 从 kg_operations 构建边
        for f in traj_findings + inter_findings:
            kg_ops = f.get("kg_operations", [])
            for op in kg_ops:
                if op.get("op") == "upsert_edge":
                    edges.append(
                        {
                            "from": op.get("from", ""),
                            "to": op.get("to", ""),
                            "type": op.get("edge_type", ""),
                            "properties": op.get("properties", {}),
                        }
                    )

        return {"entities": entities, "edges": edges}

    def _match_behavior_templates(
        self,
        entity_subgraph: dict[str, Any],
        templates: list[dict],
    ) -> list[dict]:
        """对每个行为模板计算匹配分数。

        简单实现：比较模板关键词与当前实体状态的匹配度。
        完整实现：调用 KG API 的 match_pattern() 做子图同构。

        Args:
            entity_subgraph: 当前实体子图
            templates: 行为模板列表

        Returns:
            [{"entity_id": str, "matched_template": dict,
              "raw_score": float, "evidence": [str]}]
        """
        results: list[dict] = []
        entities = entity_subgraph.get("entities", {})

        for eid, entity_data in entities.items():
            entity_state_str = self._entity_to_text(entity_data)

            for tmpl in templates:
                score, matched_keywords = self._compute_template_match(
                    entity_data, tmpl
                )

                if score >= 0.3:  # 只保留有意义的匹配
                    evidence = [
                        f"子图匹配 {tmpl['behavior_id']} "
                        f"({tmpl['pattern_type']}), "
                        f"match_score={score:.3f}",
                    ]
                    if matched_keywords:
                        evidence.append(
                            f"匹配关键词: {', '.join(matched_keywords)}"
                        )
                    evidence.append(
                        f"实体状态: {entity_state_str[:100]}"
                    )

                    results.append(
                        {
                            "entity_id": eid,
                            "matched_template": tmpl,
                            "raw_score": score,
                            "evidence": evidence,
                        }
                    )

        # 去重：每个 entity-template 对只保留最高分
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for r in sorted(
            results, key=lambda x: x["raw_score"], reverse=True
        ):
            key = (r["entity_id"], r["matched_template"]["behavior_id"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return deduped

    def _compute_template_match(
        self,
        entity_data: dict,
        template: dict,
    ) -> tuple[float, list[str]]:
        """计算实体与单个模板的匹配分数。

        基于关键词匹配的简单余弦相似度。

        Args:
            entity_data: 实体数据
            template: 行为模板

        Returns:
            (score, matched_keywords)
        """
        motion = entity_data.get("motion") or {}
        interactions = entity_data.get("interactions", [])
        velocity = entity_data.get("velocity_ms", 0)
        keywords = template["keywords"]

        # 轨迹/交互 Agent 的 finding.value 可能是结构化 dict（如 {"label": "徘徊"}）
        # 或中文描述串；这里统一取成文本，否则下面按关键词求交集会崩。
        def _as_text(value: Any) -> str:
            if isinstance(value, dict):
                for key in ("label", "description", "value"):
                    text = value.get(key)
                    if isinstance(text, str) and text:
                        return text
                return ""
            return value if isinstance(value, str) else ""

        # 将实体状态转为关键词
        entity_keywords: list[str] = []

        # 运动模式关键词
        pattern = _as_text(motion.get("pattern"))
        if pattern:
            entity_keywords.append(pattern)
            if pattern == "loitering":
                entity_keywords.extend(
                    ["low_speed", "direction_change"]
                )
            elif pattern == "running":
                entity_keywords.append("high_speed")
            elif pattern == "erratic":
                entity_keywords.extend(
                    ["direction_change", "unpredictable"]
                )
            elif pattern == "stationary":
                entity_keywords.append("prolonged_stay")
            elif pattern == "walking":
                entity_keywords.append("normal_speed")

        # 交互关键词
        for inter in interactions:
            inter_type = _as_text(inter.get("type"))
            if inter_type == "following":
                entity_keywords.extend(
                    ["following", "same_path", "proximity"]
                )
            elif inter_type == "isolated":
                entity_keywords.extend(
                    ["isolated", "alone", "no_interaction"]
                )
            elif inter_type == "interacting":
                entity_keywords.append("proximity")

        # 速度关键词
        if velocity > 5.0:
            entity_keywords.append("high_speed")
        elif velocity < 0.3:
            entity_keywords.append("low_speed")

        # 计算匹配（交集大小 / 关键词集合大小）
        entity_set = set(k.lower() for k in entity_keywords)
        template_set = set(k.lower() for k in keywords)

        if not template_set:
            return (0.0, [])

        matched = entity_set & template_set
        # 加权匹配：匹配数 / 模板关键词数
        score = len(matched) / len(template_set) if template_set else 0.0

        # 约束到 0-1
        score = min(1.0, max(0.0, score))

        return (score, list(matched))

    def _entity_to_text(self, entity_data: dict) -> str:
        """将实体数据转为文本描述（用于 evidence 和 LLM 调用）。"""
        parts = [
            f"class={entity_data.get('class', 'N/A')}",
            # 无相机标定，速度是像素/秒
            f"velocity={entity_data.get('velocity_ms', 0):.2f}px/s",
        ]
        motion = entity_data.get("motion")
        if motion:
            parts.append(
                f"motion={motion.get('pattern', 'N/A')}"
                f"(conf={motion.get('confidence', 0):.2f})"
            )
        interactions = entity_data.get("interactions", [])
        if interactions:
            for inter in interactions:
                parts.append(
                    f"interaction={inter.get('type', 'N/A')}"
                    f"->{','.join(inter.get('with', []))}"
                )
        return " | ".join(parts)
