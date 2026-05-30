from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _count_items(items: Any) -> int:
    if items is None:
        return 0
    try:
        return len(items)
    except TypeError:
        return 0


def _round(value: float) -> float:
    return round(float(value), 4)


@dataclass
class TrackerMetrics:
    """Lightweight tracker quality counters.

    These are online proxy metrics for local tuning, not MOT benchmark scores.
    Ground-truth MOT17 evaluation can compare the same fields before and after
    parameter changes to anchor detection drop and raw ID fragmentation.
    """

    frame_count: int = 0
    detection_total: int = 0
    tracked_total: int = 0
    frames_with_detections: int = 0
    frames_with_tracks: int = 0
    frames_with_detections_no_tracks: int = 0
    _entity_track_ids: dict[str, set[int]] = field(default_factory=dict)

    def observe(
        self,
        detections: Any,
        tracked_objects: list[dict[str, Any]] | None,
        entities: list[dict[str, Any]] | None = None,
    ) -> None:
        detection_count = _count_items(detections)
        track_count = _count_items(tracked_objects)

        self.frame_count += 1
        self.detection_total += detection_count
        self.tracked_total += track_count
        if detection_count:
            self.frames_with_detections += 1
        if track_count:
            self.frames_with_tracks += 1
        if detection_count and not track_count:
            self.frames_with_detections_no_tracks += 1

        for obj in tracked_objects or []:
            entity_id = str(obj.get("entity_id") or f"Track-{obj.get('track_id')}")
            try:
                track_id = int(obj["track_id"])
            except (KeyError, TypeError, ValueError):
                continue
            self._entity_track_ids.setdefault(entity_id, set()).add(track_id)

        for entity in entities or []:
            entity_id = str(entity.get("entity_id", ""))
            if not entity_id:
                continue
            bucket = self._entity_track_ids.setdefault(entity_id, set())
            for track_id in entity.get("track_ids", []) or []:
                try:
                    bucket.add(int(track_id))
                except (TypeError, ValueError):
                    continue

    @property
    def entity_count(self) -> int:
        return len(self._entity_track_ids)

    @property
    def raw_track_count(self) -> int:
        return sum(len(track_ids) for track_ids in self._entity_track_ids.values())

    @property
    def id_switch_count(self) -> int:
        return sum(max(0, len(track_ids) - 1) for track_ids in self._entity_track_ids.values())

    def to_dict(self) -> dict[str, Any]:
        tracker_drop_count = max(0, self.detection_total - self.tracked_total)
        return {
            "frame_count": self.frame_count,
            "detection_total": self.detection_total,
            "tracked_total": self.tracked_total,
            "frames_with_detections": self.frames_with_detections,
            "frames_with_tracks": self.frames_with_tracks,
            "frames_with_detections_no_tracks": self.frames_with_detections_no_tracks,
            "tracker_drop_count": tracker_drop_count,
            "tracker_drop_rate": _round(tracker_drop_count / self.detection_total)
            if self.detection_total else 0.0,
            "detection_gap_rate": _round(self.frames_with_detections_no_tracks / self.frames_with_detections)
            if self.frames_with_detections else 0.0,
            "entity_count": self.entity_count,
            "raw_track_count": self.raw_track_count,
            "id_switch_count": self.id_switch_count,
            "id_switch_per_entity": _round(self.id_switch_count / self.entity_count)
            if self.entity_count else 0.0,
        }

    def compare_to(self, baseline: dict[str, Any]) -> dict[str, Any]:
        current = self.to_dict()
        keys = ("tracker_drop_rate", "detection_gap_rate", "id_switch_count", "id_switch_per_entity")
        return {
            key: {
                "current": current[key],
                "baseline": baseline.get(key, 0),
                "absolute_change": _round(current[key] - float(baseline.get(key, 0))),
                "relative_change": _round((current[key] - float(baseline.get(key, 0))) / float(baseline[key]))
                if baseline.get(key) else 0.0,
            }
            for key in keys
        }

    def acceptance_anchor(self, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics = self.to_dict()
        anchor = {
            "requirement": "tracker调参验收需记录tracker_drop_rate、detection_gap_rate、id_switch_count，并与基线比较绝对值或相对变化。",
            "current": metrics,
        }
        if baseline:
            anchor["comparison"] = self.compare_to(baseline)
        return anchor
