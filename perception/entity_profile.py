from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0

    n = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    norm_a = math.sqrt(sum(float(a[i]) ** 2 for i in range(n)))
    norm_b = math.sqrt(sum(float(b[i]) ** 2 for i in range(n)))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _centroid_distance(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("inf")
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


@dataclass
class TrackSegment:
    track_id: int
    frame_start: int
    frame_end: int
    confidence: float = 1.0
    quality: str = "stable"

    def to_lifeline_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "quality": self.quality,
        }


@dataclass
class EntityProfile:
    entity_id: str
    track_ids: list[int] = field(default_factory=list)
    segments: list[TrackSegment] = field(default_factory=list)
    global_trajectory: list[list[float]] = field(default_factory=list)
    occlusion_gaps: list[dict[str, Any]] = field(default_factory=list)
    merge_evidence: list[dict[str, Any]] = field(default_factory=list)
    tentative_merge_candidates: list[str] = field(default_factory=list)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    last_centroid: list[float] = field(default_factory=lambda: [0.0, 0.0])
    last_feature: list[float] = field(default_factory=list)
    tracking_quality: str = "stable"

    def trajectory_summary(self) -> dict[str, Any]:
        if not self.global_trajectory:
            return {"point_count": 0}

        xs = [float(point[0]) for point in self.global_trajectory]
        ys = [float(point[1]) for point in self.global_trajectory]
        return {
            "point_count": len(self.global_trajectory),
            "activity_bbox": [min(xs), min(ys), max(xs), max(ys)],
        }

    def to_entity_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "track_ids": list(self.track_ids),
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "tracking_quality": self.tracking_quality,
            "trajectory_summary": self.trajectory_summary(),
            "merge_evidence_summary": copy.deepcopy(self.merge_evidence[-3:]),
            "tentative_merge_candidates": list(self.tentative_merge_candidates),
        }


class EntityProfileManager:
    def __init__(
        self,
        auto_merge_threshold: float = 0.78,
        tentative_threshold: float = 0.30,
        max_merge_gap_frames: int = 150,
        max_position_gap_px: float = 140.0,
        enable_debug: bool = False,
        max_debug_records: int = 5000,
    ) -> None:
        self.auto_merge_threshold = auto_merge_threshold
        self.tentative_threshold = tentative_threshold
        self.max_merge_gap_frames = max_merge_gap_frames
        self.max_position_gap_px = max_position_gap_px
        self.enable_debug = enable_debug
        self.max_debug_records = max_debug_records
        self.debug_records: list[dict[str, Any]] = []
        self.debug_stats: dict[str, int] = {
            "auto_merge": 0,
            "tentative": 0,
            "new_entity": 0,
            "no_candidate": 0,
        }
        self._profiles: dict[str, EntityProfile] = {}
        self._track_to_entity: dict[int, str] = {}
        self._next_entity_index = 1

    def update(self, tracked: list[dict[str, Any]], frame_idx: int, timestamp: float) -> list[dict[str, Any]]:
        enriched = []
        candidate_entity_ids = set(self._profiles)
        for obj in tracked:
            entity_id, merge_evidence = self._resolve_entity(obj, frame_idx, candidate_entity_ids)
            profile = self._profiles[entity_id]
            self._apply_observation(profile, obj, frame_idx, timestamp, merge_evidence)

            out = dict(obj)
            out["entity_id"] = entity_id
            out["tracking_quality"] = profile.tracking_quality
            enriched.append(out)
        return enriched

    def get_profile(self, entity_id: str) -> EntityProfile:
        return self._profiles[entity_id]

    def profiles(self) -> list[EntityProfile]:
        return list(self._profiles.values())

    def to_entities(self) -> list[dict[str, Any]]:
        return [profile.to_entity_dict() for profile in self.profiles()]

    def to_lifelines(self) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": profile.entity_id,
                "segments": [segment.to_lifeline_dict() for segment in profile.segments],
                "occlusion_gaps": copy.deepcopy(profile.occlusion_gaps),
            }
            for profile in self.profiles()
        ]

    def _resolve_entity(
        self,
        obj: dict[str, Any],
        frame_idx: int,
        candidate_entity_ids: set[str],
    ) -> tuple[str, dict[str, Any] | None]:
        track_id = int(obj["track_id"])
        if track_id in self._track_to_entity:
            return self._track_to_entity[track_id], None

        best_entity_id = ""
        best_score = 0.0
        best_factors: dict[str, Any] = {}
        scored_candidates: list[dict[str, Any]] = []
        for entity_id in candidate_entity_ids:
            profile = self._profiles[entity_id]
            score, factors = self._score_candidate(profile, obj, frame_idx)
            scored_candidates.append({
                "entity_id": entity_id,
                "latest_track_id": profile.track_ids[-1] if profile.track_ids else None,
                "score": round(score, 3),
                "factors": factors,
            })
            if score > best_score:
                best_entity_id = entity_id
                best_score = score
                best_factors = factors

        if best_entity_id and best_score >= self.auto_merge_threshold:
            self._track_to_entity[track_id] = best_entity_id
            profile = self._profiles[best_entity_id]
            self._record_debug_decision(
                obj,
                frame_idx,
                "auto_merge",
                best_entity_id,
                best_score,
                best_factors,
                scored_candidates,
            )
            return best_entity_id, {
                "track_a": profile.track_ids[-1] if profile.track_ids else None,
                "track_b": track_id,
                "score": round(best_score, 3),
                "factors": best_factors,
            }

        entity_id = self._new_entity_id()
        self._track_to_entity[track_id] = entity_id
        self._profiles[entity_id] = EntityProfile(entity_id=entity_id, first_seen_frame=frame_idx)
        if best_entity_id and best_score >= self.tentative_threshold:
            self._profiles[entity_id].tentative_merge_candidates.append(best_entity_id)
            self._profiles[entity_id].tracking_quality = "tentative"
            self._record_debug_decision(
                obj,
                frame_idx,
                "tentative",
                best_entity_id,
                best_score,
                best_factors,
                scored_candidates,
                created_entity_id=entity_id,
            )
        else:
            self._record_debug_decision(
                obj,
                frame_idx,
                "new_entity",
                best_entity_id,
                best_score,
                best_factors,
                scored_candidates,
                created_entity_id=entity_id,
            )
        return entity_id, None

    def _score_candidate(
        self,
        profile: EntityProfile,
        obj: dict[str, Any],
        frame_idx: int,
    ) -> tuple[float, dict[str, Any]]:
        frame_gap = frame_idx - profile.last_seen_frame
        if frame_gap <= 0:
            return 0.0, {"frame_gap": frame_gap, "reason": "non_forward_frame_gap"}
        if frame_gap > self.max_merge_gap_frames:
            return 0.0, {"frame_gap": frame_gap, "reason": "gap_exceeds_limit"}

        centroid = list(obj.get("centroid", [0.0, 0.0]))
        distance = _centroid_distance(profile.last_centroid, centroid)
        position_score = max(0.0, 1.0 - distance / self.max_position_gap_px)
        time_score = max(0.0, 1.0 - frame_gap / self.max_merge_gap_frames)
        appearance_feature = list(obj.get("appearance_feature", []))
        factors = [
            (0.35, position_score),
            (0.30, time_score),
        ]
        appearance_score = None
        if profile.last_feature and appearance_feature:
            appearance_score = _cosine_similarity(profile.last_feature, appearance_feature)
            factors.append((0.35, appearance_score))

        available_weight = sum(weight for weight, _ in factors)
        score = sum(weight * value for weight, value in factors) / available_weight

        result_factors = {
            "frame_gap": frame_gap,
            "distance_px": round(distance, 3),
            "position_score": round(position_score, 3),
            "time_score": round(time_score, 3),
        }
        if appearance_score is not None:
            result_factors["appearance_score"] = round(appearance_score, 3)
        return score, result_factors

    def _apply_observation(
        self,
        profile: EntityProfile,
        obj: dict[str, Any],
        frame_idx: int,
        timestamp: float,
        merge_evidence: dict[str, Any] | None,
    ) -> None:
        track_id = int(obj["track_id"])
        if not profile.track_ids:
            profile.first_seen_frame = frame_idx

        if track_id not in profile.track_ids:
            if profile.track_ids:
                profile.occlusion_gaps.append(
                    {
                        "from_frame": profile.last_seen_frame,
                        "to_frame": frame_idx,
                        "reason": "track_reappeared",
                        "reid_score": merge_evidence["score"] if merge_evidence else 0.0,
                    }
                )
                profile.tracking_quality = "occluded"
            profile.track_ids.append(track_id)
            profile.segments.append(TrackSegment(track_id=track_id, frame_start=frame_idx, frame_end=frame_idx))
            if merge_evidence:
                profile.merge_evidence.append(merge_evidence)
        else:
            for segment in reversed(profile.segments):
                if segment.track_id == track_id:
                    segment.frame_end = frame_idx
                    break

        profile.last_seen_frame = frame_idx
        profile.last_centroid = list(obj.get("centroid", profile.last_centroid))
        profile.last_feature = list(obj.get("appearance_feature", profile.last_feature))
        self._append_trajectory(profile, obj, timestamp)

    def _append_trajectory(self, profile: EntityProfile, obj: dict[str, Any], timestamp: float) -> None:
        trajectory = obj.get("trajectory") or []
        samples = trajectory[-3:] if trajectory else [[*profile.last_centroid[:2], timestamp]]
        for point in samples:
            if len(point) >= 3:
                sample = [float(point[0]), float(point[1]), float(point[2])]
            else:
                sample = [float(point[0]), float(point[1]), float(timestamp)]
            if not profile.global_trajectory or profile.global_trajectory[-1] != sample:
                profile.global_trajectory.append(sample)

        if len(profile.global_trajectory) > 300:
            del profile.global_trajectory[:-300]

    def _new_entity_id(self) -> str:
        entity_id = f"Person-{self._next_entity_index}"
        self._next_entity_index += 1
        return entity_id

    def _record_debug_decision(
        self,
        obj: dict[str, Any],
        frame_idx: int,
        decision: str,
        best_entity_id: str,
        best_score: float,
        best_factors: dict[str, Any],
        scored_candidates: list[dict[str, Any]],
        created_entity_id: str = "",
    ) -> None:
        if not self.enable_debug:
            return

        self.debug_stats[decision] = self.debug_stats.get(decision, 0) + 1
        if not best_entity_id:
            self.debug_stats["no_candidate"] = self.debug_stats.get("no_candidate", 0) + 1

        top_candidates = sorted(
            scored_candidates,
            key=lambda item: item["score"],
            reverse=True,
        )[:10]
        blocker = self._debug_blocker(decision, best_score, best_factors)
        record = {
            "frame_idx": frame_idx,
            "track_id": int(obj.get("track_id", 0)),
            "created_entity_id": created_entity_id,
            "decision": decision,
            "best_entity_id": best_entity_id,
            "best_score": round(best_score, 3),
            "blocker": blocker,
            "thresholds": {
                "auto_merge": self.auto_merge_threshold,
                "tentative": self.tentative_threshold,
                "max_merge_gap_frames": self.max_merge_gap_frames,
                "max_position_gap_px": self.max_position_gap_px,
            },
            "best_factors": best_factors,
            "top_candidates": top_candidates,
        }
        self.debug_records.append(record)
        if len(self.debug_records) > self.max_debug_records:
            del self.debug_records[: len(self.debug_records) - self.max_debug_records]

    def _debug_blocker(self, decision: str, best_score: float, best_factors: dict[str, Any]) -> str:
        if decision == "auto_merge":
            return ""
        if not best_factors:
            return "no_candidate"
        if best_factors.get("reason"):
            return str(best_factors["reason"])
        if best_score < self.tentative_threshold:
            return "score_below_tentative_threshold"
        if best_score < self.auto_merge_threshold:
            return "score_below_auto_merge_threshold"
        return "unknown"

    def debug_summary(self) -> dict[str, Any]:
        blockers: dict[str, int] = {}
        for record in self.debug_records:
            blocker = record.get("blocker") or "auto_merge"
            blockers[blocker] = blockers.get(blocker, 0) + 1
        return {
            "enabled": self.enable_debug,
            "stats": dict(self.debug_stats),
            "blockers": blockers,
            "record_count": len(self.debug_records),
            "records": copy.deepcopy(self.debug_records),
        }
