"""
交互分析 Agent。

分析多目标之间的交互关系。支持成对关系计算、群体聚类、孤立检测。

v2: 基于 benchmark 验证的特征提取 + 跳变检测 → 分类为 结伴同行/面对面交谈/短暂接触/无交互。
"""

import math
from typing import Any

import numpy as np

from .base_agent import BaseAgent
from .llm_client import LLMClient


# ── 交互类型常量 ─────────────────────────────────────────────────────
INTERACTION_TYPES = ("结伴同行", "面对面交谈", "短暂接触", "无交互")


def _detect_jumps(trajectory: list[list[float]], threshold: float = 100.0) -> list[list[list[float]]]:
    """将轨迹按跳变点（身份切换）分割为连续段。"""
    if len(trajectory) < 2:
        return [trajectory] if trajectory else []
    segments = []
    seg = [trajectory[0]]
    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1]
        curr = trajectory[i]
        jump = math.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
        if jump > threshold:
            if len(seg) >= 3:
                segments.append(seg)
            seg = []
        seg.append(curr)
    if len(seg) >= 3:
        segments.append(seg)
    return segments if segments else [trajectory]


def _compute_pair_features(
    traj_a: list[list[float]], traj_b: list[list[float]],
) -> dict:
    """从两条轨迹计算交互几何特征（含跳变检测和最佳段对齐）。

    对应 benchmark 验证的特征集：distance_mean, distance_std,
    trajectory_cosine, co_present_seconds, distance_trend。
    """
    segs_a = _detect_jumps(traj_a)
    segs_b = _detect_jumps(traj_b)

    best_dist = float("inf")
    best_features: dict | None = None

    for sa in segs_a:
        for sb in segs_b:
            # 找两条轨迹的时间重叠帧
            times_a = {pt[2] for pt in sa}
            sync_pairs = []
            for pt_b in sb:
                t = pt_b[2]
                for pt_a in sa:
                    if abs(pt_a[2] - t) < 0.1:
                        sync_pairs.append((pt_a, pt_b))
                        break
            if len(sync_pairs) < 2:
                continue

            dists = [math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) for a, b in sync_pairs]
            mean_d = float(np.mean(dists))
            if mean_d >= best_dist:
                continue

            # 计算特征
            dur = sync_pairs[-1][0][2] - sync_pairs[0][0][2]
            vec_a = np.array([sa[-1][0] - sa[0][0], sa[-1][1] - sa[0][1]])
            vec_b = np.array([sb[-1][0] - sb[0][0], sb[-1][1] - sb[0][1]])
            na, nb = float(np.linalg.norm(vec_a)), float(np.linalg.norm(vec_b))
            cos_val = round(float(np.dot(vec_a, vec_b) / max(na * nb, 1e-6)), 3) if na > 5 and nb > 5 else None

            features = {
                "distance_mean": round(mean_d, 1),
                "distance_std": round(float(np.std(dists)), 1) if len(dists) > 1 else 0.0,
                "trajectory_cosine": cos_val,
                "co_present_seconds": round(dur, 1),
            }
            if len(dists) >= 3:
                frames_arr = np.array([i for i in range(len(dists))])
                dist_arr = np.array(dists)
                features["distance_trend"] = round(float(np.polyfit(frames_arr, dist_arr, 1)[0]), 2)
            else:
                features["distance_trend"] = None

            best_dist = mean_d
            best_features = features

    return best_features or {"distance_mean": None}


def _heuristic_classify(features: dict) -> tuple[str, float, str]:
    """基于几何特征的规则分类——benchmark 验证 89% 准确率。

    Returns: (interaction_type, confidence, reasoning)
    """
    def _val(key, default):
        v = features.get(key)
        return default if v is None else v

    dist = _val("distance_mean", 999)
    dist_std = _val("distance_std", 999)
    traj_cos = features.get("trajectory_cosine")
    duration = _val("co_present_seconds", 0)
    dist_trend = features.get("distance_trend")

    # 结伴同行: 同向、近距离、稳定、长时间
    if traj_cos is not None and traj_cos > 0.7 and dist < 200 and dist_std < 60 and duration > 2:
        return ("结伴同行", 0.80 if (traj_cos > 0.9 and dist < 100) else 0.65,
                f"同向(cos={traj_cos:.2f})、距离{dist:.0f}px、共存{duration:.1f}s")

    # 面对面交谈: 相向、近距离、持续
    if traj_cos is not None and traj_cos < -0.5 and dist < 120 and duration > 3:
        return ("面对面交谈", 0.70, f"相向(cos={traj_cos:.2f})、近距离({dist:.0f}px)、持续{duration:.1f}s")

    # 短暂接触: 明显距离变化（靠近或远离）
    if dist_trend is not None and abs(dist_trend) > 1.5 and dist < 200 and duration < 15:
        action = "接近" if dist_trend < 0 else "远离"
        return ("短暂接触", 0.60, f"距离变化({action}, slope={dist_trend:.1f}px/frame)、均值{dist:.0f}px")

    # 相向擦肩
    if traj_cos is not None and traj_cos < -0.5 and dist < 150:
        return ("短暂接触", 0.70, f"轨迹反向(cos={traj_cos:.2f})、近距离({dist:.0f}px)")

    # 无交互
    if dist > 300:
        return ("无交互", 0.85, f"距离远({dist:.0f}px)")
    if duration < 0.5:
        return ("无交互", 0.75, f"共现极短({duration:.1f}s)")

    return ("无交互", 0.55, f"未达交互阈值(dist={dist:.0f}px, dur={duration:.1f}s)")


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
        """分析目标间的交互关系，v2: 基于几何特征 + 跳变检测的分类。

        Returns:
            {"agent_confidence": float, "warnings": list[str], "findings": list[dict]}
        """
        tracklets = self.context.get("tracklets", [])
        spatial_relations = self.context.get("spatial_relations", [])
        tracklet_map: dict[int, dict] = {t.get("track_id", 0): t for t in tracklets}

        entity_map: dict[int, str] = {}
        for t in tracklets:
            tid = t.get("track_id", 0)
            eid = t.get("entity_id", "") or f"Entity-{tid}"
            entity_map[tid] = eid

        groups = self._cluster_groups(tracklets)
        all_ids_in_groups: set[int] = set()
        for g in groups:
            all_ids_in_groups.update(g)
        all_track_ids = {t.get("track_id", 0) for t in tracklets}
        isolated_ids = all_track_ids - all_ids_in_groups

        findings: list[dict] = []
        warnings: list[str] = []
        confidences: list[float] = []
        seen_entities: set[tuple[str, str]] = set()

        # ── 成对关系：基于几何特征分类 ──
        for i in range(len(tracklets)):
            for j in range(i + 1, len(tracklets)):
                ta = tracklets[i]
                tb = tracklets[j]
                tid_a = ta.get("track_id", 0)
                tid_b = tb.get("track_id", 0)
                eid_a = entity_map.get(tid_a, f"Entity-{tid_a}")
                eid_b = entity_map.get(tid_b, f"Entity-{tid_b}")
                pair_key = (min(eid_a, eid_b), max(eid_a, eid_b))
                if pair_key in seen_entities:
                    continue
                seen_entities.add(pair_key)

                # 计算几何特征
                traj_a = ta.get("trajectory", []) or []
                traj_b = tb.get("trajectory", []) or []
                features = _compute_pair_features(traj_a, traj_b)

                if features.get("distance_mean") is None:
                    # 没有有效同步帧，用 spatial_relations 回退
                    dist = self._get_distance(tid_a, tid_b, spatial_relations)
                    dur = self._get_duration(tid_a, tid_b, spatial_relations)
                    if dist > 300:
                        i_type, confidence, reasoning = "无交互", 0.80, f"距离{dist:.0f}px,无同步帧"
                    else:
                        i_type, confidence, reasoning = "无交互", 0.50, f"距离{dist:.0f}px,无法提取特征"
                    features["distance_mean"] = dist
                else:
                    i_type, confidence, reasoning = _heuristic_classify(features)

                # 低置信度 → LLM 精判
                llm_result = None
                if confidence < 0.60 and self.llm:
                    llm_result = self._llm_classify_pair(eid_a, eid_b, features)
                    if llm_result and llm_result.get("confidence", 0) > confidence:
                        i_type = llm_result["interaction_type"]
                        confidence = llm_result["confidence"]
                        reasoning = llm_result.get("reasoning", reasoning)

                dist = features.get("distance_mean") or 0
                dur = features.get("co_present_seconds") or 0
                cos_val = features.get("trajectory_cosine")

                evidence = [
                    f"dist={dist:.0f}px dur={dur:.1f}s"
                    + (f" cos={cos_val:.2f}" if cos_val is not None else "")
                ]
                if features.get("distance_trend") is not None:
                    evidence.append(f"trend={features['distance_trend']:.2f}")

                description = f"{eid_a} × {eid_b}: {i_type} ({reasoning[:100]})"

                findings.append({
                    "entity_id": eid_a,
                    "finding_type": "interaction_classification",
                    "value": description,
                    "interaction_type": i_type,
                    "confidence": round(confidence, 3),
                    "evidence": evidence,
                    "related_entities": [eid_b],
                    "kg_operations": [{
                        "op": "upsert_edge", "edge_type": "interacts_with",
                        "from": eid_a, "to": eid_b,
                        "properties": {
                            "interaction_type": i_type,
                            "confidence": round(confidence, 3),
                            "distance_px": dist,
                            "duration_s": dur,
                        },
                    }],
                })
                confidences.append(confidence)

        # 孤立个体描述
        for tid in isolated_ids:
            eid = entity_map.get(tid, f"Entity-{tid}")
            nearest = self._find_nearest(tid, tracklets)
            nearest_eid = entity_map.get(nearest, f"Entity-{nearest}") if nearest else "无"
            dist = self._get_distance(tid, nearest, spatial_relations) if nearest else float("inf")
            findings.append({
                "entity_id": eid,
                "finding_type": "interaction_description",
                "value": f"{eid} 独立于群体之外"
                         + (f"，最近目标为 {nearest_eid}（距离 {dist:.0f}px）" if nearest else ""),
                "confidence": 0.80,
                "evidence": [f"not in any group, nearest={nearest_eid} dist={dist:.0f}px"],
                "related_entities": [nearest_eid] if nearest else [],
                "kg_operations": [],
            })
            confidences.append(0.80)

        # 群体描述
        for group in groups:
            if len(group) >= 2:
                gnames = [entity_map.get(tid, f"Entity-{tid}") for tid in group]
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

    def _llm_classify_pair(
        self, eid_a: str, eid_b: str, features: dict
    ) -> dict | None:
        """LLM 精判：对低置信度的 Heuristic 结果做语义分析。"""
        def _fmt(v, unit=""):
            if v is None: return "未知"
            if isinstance(v, float): return f"{v:.1f}{unit}"
            return f"{v}{unit}"

        ft = features
        feature_text = (
            f"距离均值: {_fmt(ft.get('distance_mean'), 'px')}, "
            f"距离标准差: {_fmt(ft.get('distance_std'), 'px')}, "
            f"轨迹方向余弦: {_fmt(ft.get('trajectory_cosine'))}, "
            f"共存时长: {_fmt(ft.get('co_present_seconds'), 's')}, "
            f"距离趋势: {_fmt(ft.get('distance_trend'))} (负=靠近 正=远离)"
        )
        prompt = (
            f"分析 {eid_a} 与 {eid_b} 的交互类型。\n"
            f"特征：{feature_text}\n\n"
            "余弦≈1=同向，≈-1=相向，≈0=垂直。距离趋势负=靠近，正=远离。\n"
            "类型：结伴同行 / 面对面交谈 / 短暂接触 / 无交互\n"
            '输出 JSON: {"interaction_type": "短暂接触", "confidence": 0.xx, "reasoning": "..."}'
        )
        try:
            result = self.llm.chat_with_json_output(
                system_prompt="你是视频监控交互分析助手。仅输出 JSON。",
                user_prompt=prompt,
            )
            if "error" not in result:
                return result
        except Exception:
            pass
        return None
