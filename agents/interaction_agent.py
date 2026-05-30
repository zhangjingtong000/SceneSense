"""
交互分析 Agent。

分析多目标之间的交互关系。支持成对关系计算、群体聚类、孤立检测。
"""

import math
from typing import Any

from .base_agent import BaseAgent
from .llm_client import LLMClient


class InteractionAgent(BaseAgent):
    """分析多目标之间的交互关系。

    输入：SpatioTemporalEvent 中的 spatial_relations + tracklets
    输出：成对交互关系、群体聚类、孤立个体
    """

    # 距离聚类阈值（米）
    GROUP_DISTANCE_THRESHOLD = 2.0
    # 交互评分阈值
    INTERACTION_LOW = 0.3
    INTERACTION_HIGH = 0.7

    def __init__(self, llm_client: LLMClient) -> None:
        prompt = self._load_prompt("interaction")
        super().__init__("interaction_agent", llm_client, prompt)

    def think(self) -> dict:
        """分析目标间的交互关系，生成描述文本。

        Returns:
            {"agent_confidence": float, "warnings": list[str], "findings": list[dict]}
        """
        tracklets = self.context.get("tracklets", [])
        spatial_relations = self.context.get("spatial_relations", [])
        tracklet_map: dict[int, dict] = {t.get("track_id", 0): t for t in tracklets}

        pair_scores = self._compute_all_pairs(tracklets, spatial_relations)
        groups = self._cluster_groups(tracklets)

        all_ids_in_groups: set[int] = set()
        for g in groups:
            all_ids_in_groups.update(g)
        all_track_ids = {t.get("track_id", 0) for t in tracklets}
        isolated_ids = all_track_ids - all_ids_in_groups

        findings: list[dict] = []
        warnings: list[str] = []
        confidences: list[float] = []
        seen_pairs: set[tuple[int, int]] = set()

        # 成对关系描述
        for (tid_a, tid_b), score in pair_scores.items():
            if score is None:
                continue
            pair_key = (min(tid_a, tid_b), max(tid_a, tid_b))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            entity_a, entity_b = f"Entity-{tid_a}", f"Entity-{tid_b}"
            dist = self._get_distance(tid_a, tid_b, spatial_relations)
            dur = self._get_duration(tid_a, tid_b, spatial_relations)

            dist_qualifier = (
                "非常接近" if dist < 50 else "接近" if dist < 100 else
                "中等距离" if dist < 200 else "较远"
            )
            score_qualifier = "强交互" if score > 0.7 else "中等交互" if score > 0.4 else "弱交互"

            # 相对运动方向差异
            ta = tracklet_map.get(tid_a, {})
            tb = tracklet_map.get(tid_b, {})
            dir_diff = abs(ta.get("direction_deg", 0) - tb.get("direction_deg", 0))
            dir_diff = min(dir_diff, 360 - dir_diff)
            dir_text = "同向移动" if dir_diff < 45 else "交叉移动" if dir_diff < 135 else "反向移动"

            description = (
                f"{entity_a} 与 {entity_b} 间距 {dist:.0f}px（{dist_qualifier}），"
                f"共现 {dur:.1f}s，{score_qualifier}（评分 {score:.2f}），{dir_text}"
            )
            evidence = [
                f"distance={dist:.0f}px duration={dur:.1f}s score={score:.2f} dir_diff={dir_diff:.0f}°"
            ]
            confidence = 0.85 if score > 0.7 or score < 0.2 else 0.70 if score > 0.4 else 0.55

            findings.append({
                "entity_id": entity_a,
                "finding_type": "interaction_description",
                "value": description,
                "confidence": round(confidence, 3),
                "evidence": evidence,
                "related_entities": [entity_b],
                "kg_operations": [{
                    "op": "upsert_edge", "edge_type": "interacts_with",
                    "from": entity_a, "to": entity_b,
                    "properties": {"interaction_score": round(score, 3), "distance": dist, "duration": dur},
                }],
            })
            confidences.append(confidence)

        # 孤立个体描述
        for tid in isolated_ids:
            entity_id = f"Entity-{tid}"
            nearest = self._find_nearest(tid, tracklets)
            nearest_eid = f"Entity-{nearest}" if nearest is not None else "无"
            dist = self._get_distance(tid, nearest, spatial_relations) if nearest else float("inf")
            description = (
                f"{entity_id} 独立于所有群体之外"
                + (f"，最近目标为 {nearest_eid}（距离 {dist:.0f}px）" if nearest else "，周围无其他目标")
            )
            findings.append({
                "entity_id": entity_id,
                "finding_type": "interaction_description",
                "value": description,
                "confidence": 0.80,
                "evidence": [f"not in any group, nearest={nearest_eid} dist={dist:.0f}px"],
                "related_entities": [nearest_eid] if nearest else [],
                "kg_operations": [],
            })
            confidences.append(0.80)

        # 群体描述
        for group in groups:
            if len(group) >= 2:
                gnames = [f"Entity-{tid}" for tid in group]
                desc = f"群体 [{', '.join(gnames)}]（共 {len(group)} 人）聚集在同一区域"
                findings.append({
                    "entity_id": gnames[0],
                    "finding_type": "group_description",
                    "value": desc,
                    "confidence": 0.85,
                    "evidence": [f"cluster: {group}"],
                    "related_entities": gnames[1:],
                    "kg_operations": [],
                })
                confidences.append(0.85)

        agent_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        return {"agent_confidence": round(agent_confidence, 3), "warnings": warnings, "findings": findings}

    def _compute_all_pairs(
        self,
        tracklets: list[dict],
        spatial_relations: list[dict],
    ) -> dict[tuple[int, int], float | None]:
        """计算所有目标对的交互评分。

        Returns:
            {(tid_a, tid_b): score or None}
        """
        pairs: dict[tuple[int, int], float | None] = {}
        all_ids = [t.get("track_id", 0) for t in tracklets]

        # 先处理已有空间关系数据的成对
        for rel in spatial_relations:
            t1 = rel.get("track_1", 0)
            t2 = rel.get("track_2", 0)
            score = self._compute_interaction_score_from_rel(rel)
            pairs[(t1, t2)] = score

        # 补充未在 spatial_relations 中的对
        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                tid_a = all_ids[i]
                tid_b = all_ids[j]
                if (tid_a, tid_b) not in pairs and (
                    tid_b,
                    tid_a,
                ) not in pairs:
                    pairs[(tid_a, tid_b)] = None

        return pairs

    def _compute_interaction_score_from_rel(
        self, rel: dict
    ) -> float:
        """从空间关系数据计算交互评分 0-1。"""
        distance_m = rel.get("distance_m", 10.0)
        duration_s = rel.get("duration_s", 0.0)

        # 距离越近分数越高（阈值 5m）
        dist_score = max(0, 1 - distance_m / 5.0)

        # 共现时长越长分数越高（阈值 10s）
        dur_score = min(1.0, duration_s / 10.0)

        # 加权融合
        score = 0.6 * dist_score + 0.4 * dur_score
        return min(1.0, max(0.0, score))

    def _compute_interaction_score(
        self, track_a: dict, track_b: dict
    ) -> float:
        """基于距离、相对速度、共现时长计算交互评分 0-1。

        Args:
            track_a: 目标 A 的 tracklet
            track_b: 目标 B 的 tracklet

        Returns:
            交互评分 0-1
        """
        # 使用 centroid 的欧氏距离估算
        ca = track_a.get("centroid", [0, 0])
        cb = track_b.get("centroid", [0, 0])
        if len(ca) >= 2 and len(cb) >= 2:
            dist = math.sqrt(
                (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2
            )
        else:
            dist = 10.0

        # 速度差异
        va = track_a.get("velocity_ms", 0)
        vb = track_b.get("velocity_ms", 0)
        vel_diff = abs(va - vb)
        vel_score = max(0, 1 - vel_diff / 5.0)

        # 距离评分
        dist_score = max(0, 1 - dist / 5.0)

        score = 0.7 * dist_score + 0.3 * vel_score
        return min(1.0, max(0.0, score))

    def _classify_relation(
        self,
        tid_a: int,
        tid_b: int,
        score: float | None,
        spatial_relations: list[dict],
    ) -> tuple[str, float, list[str]]:
        """基于评分归类成对关系。

        Returns:
            (relation_type, confidence, evidence_list)
        """
        evidence: list[str] = []

        if score is None:
            return ("no_interaction", 0.5, ["无空间关系数据"])

        # 找对应 spatial_relation 条目
        rel = None
        for r in spatial_relations:
            if (
                r.get("track_1") == tid_a and r.get("track_2") == tid_b
            ) or (
                r.get("track_1") == tid_b and r.get("track_2") == tid_a
            ):
                rel = r
                break

        relation_type = rel.get("relation_type", "") if rel else ""
        distance = (
            rel.get("distance_m", float("inf")) if rel else float("inf")
        )

        evidence.append(
            f"interaction_score={score:.3f}, "
            f"distance={distance:.1f}m"
        )

        # 高交互分
        if score >= 0.8:
            if relation_type == "following":
                return ("following", 0.85, evidence)
            return ("interacting", 0.80, evidence)

        if score >= 0.6:
            if distance < 1.0:
                return ("approaching", 0.75, evidence)
            return ("interacting", 0.70, evidence)

        if score >= 0.3:
            return ("near", 0.60, evidence)

        # 低交互分
        if distance > 5.0:
            return ("departing", 0.70, evidence)

        return ("no_interaction", 0.65, evidence)

    def _cluster_groups(
        self, tracklets: list[dict]
    ) -> list[list[int]]:
        """简单的距离聚类（< 2m 算一组）。使用贪心算法。

        Args:
            tracklets: 轨迹列表

        Returns:
            群体列表，每个群体是 track_id 列表
        """
        if not tracklets:
            return []

        centroids: dict[int, tuple[float, float]] = {}
        for t in tracklets:
            tid = t.get("track_id", 0)
            c = t.get("centroid", [0, 0])
            centroids[tid] = (float(c[0]), float(c[1]))

        ids = list(centroids.keys())
        groups: list[list[int]] = []
        assigned: set[int] = set()

        for tid in ids:
            if tid in assigned:
                continue
            group = [tid]
            assigned.add(tid)
            cx, cy = centroids[tid]

            for other in ids:
                if other in assigned:
                    continue
                ox, oy = centroids[other]
                dist = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
                if dist < self.GROUP_DISTANCE_THRESHOLD:
                    group.append(other)
                    assigned.add(other)

            if len(group) > 0:
                groups.append(group)

        return groups

    def _find_nearest(
        self, tid: int, tracklets: list[dict]
    ) -> int | None:
        """找最近目标的 track_id。"""
        target = None
        for t in tracklets:
            if t.get("track_id") == tid:
                target = t
                break
        if target is None:
            return None

        tc = target.get("centroid", [0, 0])
        if len(tc) < 2:
            return None

        nearest_id: int | None = None
        min_dist = float("inf")

        for t in tracklets:
            other_id = t.get("track_id", 0)
            if other_id == tid:
                continue
            oc = t.get("centroid", [0, 0])
            if len(oc) < 2:
                continue
            dist = math.sqrt(
                (tc[0] - oc[0]) ** 2 + (tc[1] - oc[1]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                nearest_id = other_id

        return nearest_id

    def _get_duration(
        self, tid_a: int, tid_b: int, spatial_relations: list[dict]
    ) -> float:
        for rel in spatial_relations:
            if (rel.get("track_1") == tid_a and rel.get("track_2") == tid_b) or \
               (rel.get("track_1") == tid_b and rel.get("track_2") == tid_a):
                return rel.get("duration_s", 0.0)
        return 0.0

    def _get_distance(
        self, tid_a: int, tid_b: int, spatial_relations: list[dict]
    ) -> float:
        """从空间关系中取距离，没有则返回一个大数。"""
        for rel in spatial_relations:
            if (
                rel.get("track_1") == tid_a
                and rel.get("track_2") == tid_b
            ) or (
                rel.get("track_1") == tid_b
                and rel.get("track_2") == tid_a
            ):
                return rel.get("distance_m", float("inf"))
        return float("inf")

    def _llm_refine(
        self,
        tid_a: int,
        tid_b: int,
        score: float | None,
        spatial_relations: list[dict],
        tracklet_map: dict[int, dict],
    ) -> dict | None:
        """边界情况调用 LLM 做语义判断。"""
        user_prompt = (
            f"Entity-{tid_a} <-> Entity-{tid_b}: "
            f"interaction_score={score:.3f}\n"
        )
        if (tid_a, tid_b) in [
            (r.get("track_1"), r.get("track_2"))
            for r in spatial_relations
        ] or (tid_b, tid_a) in [
            (r.get("track_1"), r.get("track_2"))
            for r in spatial_relations
        ]:
            rel = next(
                r
                for r in spatial_relations
                if (
                    r.get("track_1") == tid_a
                    and r.get("track_2") == tid_b
                )
                or (
                    r.get("track_1") == tid_b
                    and r.get("track_2") == tid_a
                )
            )
            user_prompt += (
                f"relation_type={rel.get('relation_type', 'N/A')}, "
                f"distance={rel.get('distance_m', 'N/A')}m, "
                f"duration={rel.get('duration_s', 'N/A')}s\n"
            )

        user_prompt += (
            "\nClassify interaction type: "
            "interacting, following, approaching, departing, "
            "surrounding, no_interaction.\n"
            'Output JSON: {"relation_type": str, '
            '"confidence": float, "evidence": [str]}'
        )

        result = self.llm.chat_with_json_output(
            system_prompt=self.system_prompt, user_prompt=user_prompt
        )
        if "error" not in result:
            return result
        return None
