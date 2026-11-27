"""IoU 与帧内一对一匹配。

匹配规则（与验收 spec §7.0 一致）：

- 同帧一对一：一个预测框最多匹配一个真值框，反之亦然。
- 匹配条件为 IoU >= 0.5。
- 按 IoU 从高到低贪心分配，保证高重叠优先。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .mot17_gt import GTBox

DEFAULT_IOU_THRESHOLD = 0.5

PredBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Match:
    """一次成功的同帧匹配。"""

    frame: int
    gt_index: int
    pred_index: int
    iou: float
    person_id: int
    entity_id: str
    track_id: int


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """计算两个 xyxy 框的交并比。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _pred_bbox(pred: Mapping[str, object]) -> PredBox:
    bbox = pred.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"预测框格式错误: {bbox!r}")
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def match_frame(
    frame: int,
    gt_boxes: Sequence[GTBox],
    preds: Sequence[Mapping[str, object]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[Match]:
    """在同一帧内做一对一贪心匹配。"""
    candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(preds):
            score = iou(gt.bbox, _pred_bbox(pred))
            if score >= iou_threshold:
                candidates.append((score, gi, pi))

    candidates.sort(key=lambda item: item[0], reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[Match] = []
    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        gt = gt_boxes[gi]
        pred = preds[pi]
        matches.append(
            Match(
                frame=frame,
                gt_index=gi,
                pred_index=pi,
                iou=round(score, 4),
                person_id=gt.person_id,
                entity_id=str(pred.get("entity_id") or pred.get("track_id", "")),
                track_id=int(pred.get("track_id", -1)),
            )
        )
    return matches
