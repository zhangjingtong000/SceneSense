"""Gate 指标计算（Gate 1 检测、Gate 2 连续性）。

口径：

- 帧级指标只在核心目标上计入硬门槛，挑战目标单独统计。
- 人物发现率：某 GT 人物的核心帧中，至少一次正确匹配即视为已发现。
- 连续漏检：只累计该人物满足核心条件、但没有匹配预测的连续帧。
- 错误合并：同一帧内两个不同 GT 人物匹配到同一 Entity 的次数。
- 碎片数：每个 GT 人物对应过的不同 Entity 数（平均）。

恢复率（Gate 2 的 95% / 80%）由 GT 的 ``person_id`` 在可见帧断口处自动计算
（``compute_recovery_rates``），不需要人工标注；只有调用 ``gate2_continuity`` 时
没传 ``gt_by_frame`` / ``pred_by_frame``，或某一档没有可判定的「消失—重现」事件，
对应档位才返回 ``None``。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .matcher import DEFAULT_IOU_THRESHOLD, Match, match_frame
from .mot17_gt import GTBox

# Gate 1 硬门槛
TARGET_DISCOVERY_RECALL = 1.0
TARGET_FRAME_RECALL = 0.90
TARGET_FRAME_PRECISION = 0.90
MAX_CONSECUTIVE_MISS = 10


@dataclass
class FrameResult:
    frame: int
    core_gt: int
    core_matched: int
    pred_count: int
    pred_matched_any: int
    matches: list[Match] = field(default_factory=list)


@dataclass
class PersonResult:
    person_id: int
    core_frames: list[int] = field(default_factory=list)
    matched_frames: set[int] = field(default_factory=set)
    max_consecutive_miss: int = 0
    entities: set[str] = field(default_factory=set)


def _consecutive_runs(frames: Sequence[int], missing: set[int]) -> int:
    """返回 frames（升序）中最长的连续漏检长度。

    只有帧号真正连续（相差 1）时才累计；中间缺帧视为断点。
    """
    longest = 0
    run = 0
    prev = None
    for frame in frames:
        contiguous = prev is not None and frame == prev + 1
        if frame in missing:
            run = run + 1 if contiguous else 1
            longest = max(longest, run)
        else:
            run = 0
        prev = frame
    return longest


def evaluate_frames(
    gt_by_frame: Mapping[int, Sequence[GTBox]],
    pred_by_frame: Mapping[int, Sequence[Mapping[str, object]]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> tuple[list[FrameResult], dict[int, PersonResult], list[dict], dict[str, set[int]]]:
    """逐帧匹配并汇总人物与实体映射。

    Returns:
        (frame_results, person_results, false_merge_events, entity_persons)
    """
    person_results: dict[int, PersonResult] = {}
    frame_results: list[FrameResult] = []
    false_merges: list[dict] = []
    entity_persons: dict[str, set[int]] = defaultdict(set)

    for frame in sorted(gt_by_frame):
        boxes = list(gt_by_frame[frame])
        core_boxes = [b for b in boxes if b.is_core]
        challenge_boxes = [b for b in boxes if not b.is_core]
        preds = list(pred_by_frame.get(frame, ()))

        for box in core_boxes:
            person_results.setdefault(box.person_id, PersonResult(box.person_id))

        matches_core = match_frame(frame, core_boxes, preds, iou_threshold)
        used_pred = {m.pred_index for m in matches_core}
        remaining = [p for i, p in enumerate(preds) if i not in used_pred]
        matches_challenge = match_frame(frame, challenge_boxes, remaining, iou_threshold)

        # 人物级统计
        matched_person_ids = {m.person_id for m in matches_core}
        for box in core_boxes:
            pr = person_results[box.person_id]
            pr.core_frames.append(frame)
            if box.person_id in matched_person_ids:
                pr.matched_frames.add(frame)

        # 实体映射与错误合并
        per_entity_matches: dict[str, list[Match]] = defaultdict(list)
        for m in matches_core:
            pr = person_results[m.person_id]
            pr.entities.add(m.entity_id)
            entity_persons[m.entity_id].add(m.person_id)
            per_entity_matches[m.entity_id].append(m)

        for entity_id, items in per_entity_matches.items():
            persons = {m.person_id for m in items}
            if len(persons) > 1:
                false_merges.append(
                    {
                        "frame": frame,
                        "entity_id": entity_id,
                        "person_ids": sorted(persons),
                        "track_ids": sorted({m.track_id for m in items}),
                    }
                )

        frame_results.append(
            FrameResult(
                frame=frame,
                core_gt=len(core_boxes),
                core_matched=len(matches_core),
                pred_count=len(preds),
                pred_matched_any=len(matches_core) + len(matches_challenge),
                matches=matches_core,
            )
        )

    for pr in person_results.values():
        pr.core_frames.sort()
        missing = set(pr.core_frames) - pr.matched_frames
        pr.max_consecutive_miss = _consecutive_runs(pr.core_frames, missing)

    return frame_results, person_results, false_merges, entity_persons


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def gate1_detection(
    frame_results: Sequence[FrameResult],
    person_results: Mapping[int, PersonResult],
    target_discovery: float = TARGET_DISCOVERY_RECALL,
    target_recall: float = TARGET_FRAME_RECALL,
    target_precision: float = TARGET_FRAME_PRECISION,
    max_miss: int = MAX_CONSECUTIVE_MISS,
) -> dict:
    """Gate 1：人物发现率、帧级召回/精确率、最长连续漏检。"""
    total_core = sum(fr.core_gt for fr in frame_results)
    matched_core = sum(fr.core_matched for fr in frame_results)
    total_pred = sum(fr.pred_count for fr in frame_results)
    matched_pred_inside = sum(fr.pred_matched_any for fr in frame_results)

    persons = list(person_results.values())
    discovered = [p for p in persons if p.matched_frames]
    undiscovered = [p.person_id for p in persons if not p.matched_frames]

    discovery_recall = _ratio(len(discovered), len(persons))
    frame_recall = _ratio(matched_core, total_core)
    # 主口径：匹配到任一 GT（核心或挑战）的预测占比
    precision_any = _ratio(matched_pred_inside, total_pred)
    # 严格口径：只算匹配到核心 GT 的预测
    precision_core = _ratio(matched_core, total_pred)

    worst_miss = max((p.max_consecutive_miss for p in persons), default=0)
    over_miss = [
        {"person_id": p.person_id, "max_consecutive_miss": p.max_consecutive_miss}
        for p in persons
        if p.max_consecutive_miss > max_miss
    ]
    fragments = [
        {"person_id": p.person_id, "entity_count": len(p.entities)}
        for p in persons
        if len(p.entities) > 1
    ]

    checks = {
        "person_discovery_recall": {
            "value": discovery_recall,
            "target": target_discovery,
            "passed": discovery_recall >= target_discovery,
        },
        "detection_recall": {
            "value": frame_recall,
            "target": target_recall,
            "passed": frame_recall >= target_recall,
        },
        "detection_precision": {
            "value": precision_any,
            "target": target_precision,
            "passed": precision_any >= target_precision,
        },
        "max_consecutive_miss": {
            "value": worst_miss,
            "target": f"<={max_miss}",
            "passed": worst_miss <= max_miss,
        },
    }

    return {
        "gate": 1,
        "name": "人物检测",
        "metrics": {
            "person_count": len(persons),
            "discovered_person_count": len(discovered),
            "core_gt_boxes": total_core,
            "matched_core_boxes": matched_core,
            "predictions": total_pred,
            "precision_any_gt": precision_any,
            "precision_core_only": precision_core,
            "undiscovered_persons": undiscovered,
            "persons_over_miss_limit": over_miss,
            "persons_with_multiple_entities": fragments,
        },
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
    }


def _person_entities_in_frames(
    gt_by_frame: Mapping[int, Sequence[GTBox]],
    pred_by_frame: Mapping[int, Sequence[Mapping[str, object]]],
    person_id: int,
    frames: Sequence[int],
    iou_threshold: float,
) -> list[str]:
    """在给定帧上，找出该 GT 人物匹配到的 entity 列表。"""
    entities: list[str] = []
    for frame in frames:
        boxes = [b for b in gt_by_frame.get(frame, []) if b.person_id == person_id]
        if not boxes:
            continue
        for match in match_frame(frame, boxes, pred_by_frame.get(frame, ()), iou_threshold):
            entities.append(match.entity_id)
    return entities


def compute_recovery_rates(
    gt_by_frame: Mapping[int, Sequence[GTBox]],
    pred_by_frame: Mapping[int, Sequence[Mapping[str, object]]],
    fps: float = 25.0,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    window: int = 5,
    min_gap_frames: int = 5,
) -> dict:
    """自动计算遮挡恢复率，无需人工标注。

    思路：GT 中同一 person_id 的可见帧若出现断口（帧号不连续 ≥ min_gap_frames），
    就是一个“消失—重现”事件；比较断口前后预测给出的 entity 是否一致。

    分层：0–2 秒、2–10 秒、>10 秒。
    """
    from .mot17_gt import core_frames_by_person

    core_frames = core_frames_by_person(dict(gt_by_frame))
    events: list[dict] = []

    for person_id, frames in core_frames.items():
        frames = sorted(frames)
        for index in range(1, len(frames)):
            gap = frames[index] - frames[index - 1]
            if gap < min_gap_frames:
                continue

            before_frames = frames[max(0, index - window):index]
            after_frames = frames[index:index + window]
            before_entities = _person_entities_in_frames(
                gt_by_frame, pred_by_frame, person_id, before_frames, iou_threshold
            )
            after_entities = _person_entities_in_frames(
                gt_by_frame, pred_by_frame, person_id, after_frames, iou_threshold
            )
            if not before_entities or not after_entities:
                continue  # 断口前后有一侧没匹配上，无法判定恢复，不计入

            before_entity = max(set(before_entities), key=before_entities.count)
            after_entity = max(set(after_entities), key=after_entities.count)
            events.append({
                "person_id": person_id,
                "gap_frames": gap,
                "gap_seconds": round(gap / fps, 2),
                "before_entity": before_entity,
                "after_entity": after_entity,
                "recovered": before_entity == after_entity,
            })

    def summarize(selected: list[dict]) -> dict:
        total = len(selected)
        recovered = sum(1 for e in selected if e["recovered"])
        return {
            "events": total,
            "recovered": recovered,
            "rate": round(recovered / total, 4) if total else None,
        }

    short = [e for e in events if e["gap_seconds"] <= 2.0]
    medium = [e for e in events if 2.0 < e["gap_seconds"] <= 10.0]
    long_gap = [e for e in events if e["gap_seconds"] > 10.0]

    return {
        "short_gap": summarize(short),
        "medium_gap": summarize(medium),
        "long_gap": summarize(long_gap),
        "all": summarize(events),
        "event_count": len(events),
        "events": events[:200],
    }


def gate2_continuity(
    frame_results: Sequence[FrameResult],
    person_results: Mapping[int, PersonResult],
    false_merges: Sequence[dict],
    entity_persons: Mapping[str, set[int]],
    baseline_fragmentation: float | None = None,
    gt_by_frame: Mapping[int, Sequence[GTBox]] | None = None,
    pred_by_frame: Mapping[int, Sequence[Mapping[str, object]]] | None = None,
    fps: float = 25.0,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    min_gap_frames: int = 5,
) -> dict:
    """Gate 2：错误合并、碎片数、遮挡恢复率（从 GT 自动计算）。"""
    persons = list(person_results.values())
    # 碎片数只对“至少匹配到一次”的人物统计，未发现者单独计数，避免稀释指标
    discovered = [p for p in persons if p.entities]
    fragment_counts = [len(p.entities) for p in discovered]
    avg_fragmentation = round(sum(fragment_counts) / len(fragment_counts), 3) if fragment_counts else 0.0

    multi_person_entities = {
        eid: sorted(pids) for eid, pids in entity_persons.items() if len(pids) > 1
    }

    fragmentation_target = None
    fragmentation_passed = None
    if baseline_fragmentation is not None and baseline_fragmentation > 0:
        fragmentation_target = round(baseline_fragmentation * 0.7, 3)
        fragmentation_passed = avg_fragmentation <= fragmentation_target

    checks = {
        "false_merge_count": {
            "value": len(false_merges),
            "target": 0,
            "passed": len(false_merges) == 0,
        }
    }
    if fragmentation_passed is not None:
        checks["entity_fragmentation"] = {
            "value": avg_fragmentation,
            "target": fragmentation_target,
            "passed": fragmentation_passed,
        }

    # 遮挡恢复率：直接从 GT 的 person_id 连续性计算，无需人工标注
    recovery = None
    if gt_by_frame is not None and pred_by_frame is not None:
        recovery = compute_recovery_rates(
            gt_by_frame, pred_by_frame, fps=fps,
            iou_threshold=iou_threshold,
            min_gap_frames=min_gap_frames,
        )
        for key, target in (("short_gap", 0.95), ("medium_gap", 0.80)):
            block = recovery[key]
            if block["rate"] is not None:
                checks[f"recovery_rate_{key}"] = {
                    "value": block["rate"],
                    "target": target,
                    "passed": block["rate"] >= target,
                }

    return {
        "gate": 2,
        "name": "单目标连续性",
        "metrics": {
            "false_merge_count": len(false_merges),
            "false_merge_events": list(false_merges[:50]),
            "entities_mapping_multiple_persons": multi_person_entities,
            "average_entity_fragmentation": avg_fragmentation,
            "persons_without_any_match": len(persons) - len(discovered),
            "baseline_fragmentation": baseline_fragmentation,
            "recovery_rate_short_gap": recovery["short_gap"]["rate"] if recovery else None,
            "recovery_rate_medium_gap": recovery["medium_gap"]["rate"] if recovery else None,
            "recovery_rate_long_gap": recovery["long_gap"]["rate"] if recovery else None,
            "recovery_events": recovery["event_count"] if recovery else 0,
            "recovery_detail": recovery,
        },
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
        "notes": [
            "恢复率由 GT 的 person_id 在可见帧断口处自动计算，不需要人工标注。",
            "碎片数基线缺省时只报告数值，不做相对下降判定。",
        ],
    }
