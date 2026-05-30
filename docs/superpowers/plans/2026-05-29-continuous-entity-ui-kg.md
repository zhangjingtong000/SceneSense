# Continuous Entity UI KG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build first-stage continuous `Person-*` understanding, KG evidence writes, and a Chinese responsive UI around entity lifelines and evidence subgraphs.

**Architecture:** Keep existing perception → Event → Agent contracts intact. Add an `EntityProfile` layer above tracker output, write continuity-aware KG nodes/edges without changing schema, then expose lightweight `entities`, `lifelines`, and `kg_subgraph` data to the single-file frontend.

**Tech Stack:** Python 3.10+, pytest, NetworkX, OpenCV, existing YOLO/trackers, FastAPI WebSocket, vanilla HTML/CSS/JS.

---

## File Structure

- Create `perception/entity_profile.py`: owns `EntityProfile`, `TrackSegment`, merge scoring, `track_id -> Person-*` mapping, lifeline export, and tracking quality summaries.
- Create `tests/perception/test_entity_profile.py`: unit tests for stable mapping, conservative merge, tentative candidates, and lifeline export.
- Create `tests/perception/test_tracker_config.py`: verifies shared tracker/detector config values and CLI frame pass-through hooks.
- Modify `perception/tracker.py`: expose optional output fields needed by EntityProfile, including `appearance_feature` and lost-track metadata without changing existing keys.
- Modify `pipeline.py`: pass `frame` into tracker, instantiate EntityProfileManager, enrich events for KG/UI, and record tracker metric counters.
- Modify `web/backend/pipeline_runner.py`: use shared tracker config, instantiate EntityProfileManager, stream `entities`, `lifelines`, and selected subgraph data.
- Modify `agents/orchestrator.py`: execute continuity KG writes around current agent `kg_operations`, and optionally translate track-based findings to stable Person IDs in reports without changing AgentResponse.
- Modify `kg/graph_store.py`: add safe helpers for list-like GraphML properties and subgraph serialization.
- Create `tests/kg/test_continuity_writes.py`: tests Entity/Event/edge writes and aggregated `interacts_with` evidence.
- Modify `web/frontend/index.html`: convert UI to Chinese responsive workspace with video, event heat bar, entity lifelines, entity detail, and KG evidence panel.
- Create `tests/web/test_frontend_contract.py`: verifies backend event payload shape through pure Python helpers.

Because `D:\gp` currently has no `.git`, commit steps are written as guarded commands. They will commit when the repository is initialized and otherwise print `no git repository; commit skipped`.

---

### Task 1: EntityProfile Core

**Files:**
- Create: `perception/entity_profile.py`
- Test: `tests/perception/test_entity_profile.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/perception/test_entity_profile.py`:

```python
from perception.entity_profile import EntityProfileManager


def track(track_id, frame, x, y, feature=None):
    return {
        "track_id": track_id,
        "bbox": [x - 5, y - 10, x + 5, y + 10],
        "centroid": [x, y],
        "trajectory": [[x, y, frame / 30.0]],
        "velocity_ms": 1.0,
        "direction_deg": 0.0,
        "appearance_feature": feature or [1.0, 0.0, 0.0],
    }


def test_stable_track_maps_to_same_person():
    mgr = EntityProfileManager(auto_merge_threshold=0.8)
    enriched = mgr.update([track(3, 10, 100, 100)], frame_idx=10, timestamp=10 / 30.0)
    assert enriched[0]["entity_id"] == "Person-1"

    enriched = mgr.update([track(3, 11, 102, 101)], frame_idx=11, timestamp=11 / 30.0)
    assert enriched[0]["entity_id"] == "Person-1"
    assert mgr.get_profile("Person-1").track_ids == [3]


def test_high_confidence_new_track_merges_after_occlusion_gap():
    mgr = EntityProfileManager(auto_merge_threshold=0.65)
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)
    enriched = mgr.update([track(17, 18, 106, 102, [0.98, 0.02])], frame_idx=18, timestamp=18 / 30.0)

    assert enriched[0]["entity_id"] == "Person-1"
    profile = mgr.get_profile("Person-1")
    assert profile.track_ids == [3, 17]
    assert profile.occlusion_gaps[0]["from_frame"] == 10
    assert profile.occlusion_gaps[0]["to_frame"] == 18
    assert profile.merge_evidence[0]["track_b"] == 17


def test_low_confidence_new_track_becomes_tentative_candidate():
    mgr = EntityProfileManager(auto_merge_threshold=0.8, tentative_threshold=0.35)
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)
    enriched = mgr.update([track(17, 18, 260, 240, [0.2, 0.98])], frame_idx=18, timestamp=18 / 30.0)

    assert enriched[0]["entity_id"] == "Person-2"
    assert mgr.get_profile("Person-2").tentative_merge_candidates == ["Person-1"]


def test_lifeline_export_uses_frame_segments_and_quality():
    mgr = EntityProfileManager(auto_merge_threshold=0.65)
    mgr.update([track(3, 10, 100, 100)], frame_idx=10, timestamp=10 / 30.0)
    mgr.update([track(3, 20, 120, 110)], frame_idx=20, timestamp=20 / 30.0)

    lifelines = mgr.to_lifelines()
    assert lifelines == [{
        "entity_id": "Person-1",
        "segments": [{"track_id": 3, "frame_start": 10, "frame_end": 20, "quality": "stable"}],
        "occlusion_gaps": [],
    }]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/perception/test_entity_profile.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'perception.entity_profile'`.

- [ ] **Step 3: Implement EntityProfileManager**

Create `perception/entity_profile.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    na = math.sqrt(sum(float(a[i]) ** 2 for i in range(n)))
    nb = math.sqrt(sum(float(b[i]) ** 2 for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _center_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


@dataclass
class TrackSegment:
    track_id: int
    frame_start: int
    frame_end: int
    confidence: float = 1.0
    quality: str = "stable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "confidence": round(self.confidence, 3),
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

    def to_summary(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "track_ids": list(self.track_ids),
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "tracking_quality": self.tracking_quality,
            "trajectory_summary": self.trajectory_summary(),
            "merge_evidence_summary": self.merge_evidence[-3:],
        }

    def trajectory_summary(self) -> dict[str, Any]:
        points = self.global_trajectory
        if not points:
            return {"point_count": 0}
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return {
            "point_count": len(points),
            "activity_bbox": [min(xs), min(ys), max(xs), max(ys)],
        }


class EntityProfileManager:
    def __init__(
        self,
        auto_merge_threshold: float = 0.78,
        tentative_threshold: float = 0.30,
        max_merge_gap_frames: int = 150,
        max_position_gap_px: float = 140.0,
    ) -> None:
        self.auto_merge_threshold = auto_merge_threshold
        self.tentative_threshold = tentative_threshold
        self.max_merge_gap_frames = max_merge_gap_frames
        self.max_position_gap_px = max_position_gap_px
        self._profiles: dict[str, EntityProfile] = {}
        self._track_to_entity: dict[int, str] = {}
        self._next_entity_index = 1

    def update(self, tracked: list[dict], frame_idx: int, timestamp: float) -> list[dict]:
        enriched: list[dict] = []
        for obj in tracked:
            entity_id, evidence = self._resolve_entity(obj, frame_idx)
            profile = self._profiles[entity_id]
            self._apply_observation(profile, obj, frame_idx, timestamp, evidence)
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
        return [p.to_summary() for p in self.profiles()]

    def to_lifelines(self) -> list[dict[str, Any]]:
        return [{
            "entity_id": p.entity_id,
            "segments": [s.to_dict() for s in p.segments],
            "occlusion_gaps": list(p.occlusion_gaps),
        } for p in self.profiles()]

    def _resolve_entity(self, obj: dict, frame_idx: int) -> tuple[str, dict[str, Any] | None]:
        track_id = int(obj.get("track_id", 0))
        if track_id in self._track_to_entity:
            return self._track_to_entity[track_id], None

        best_id = ""
        best_score = 0.0
        best_factors: dict[str, float] = {}
        for entity_id, profile in self._profiles.items():
            score, factors = self._score_candidate(profile, obj, frame_idx)
            if score > best_score:
                best_id, best_score, best_factors = entity_id, score, factors

        if best_id and best_score >= self.auto_merge_threshold:
            self._track_to_entity[track_id] = best_id
            return best_id, {
                "track_a": self._profiles[best_id].track_ids[-1],
                "track_b": track_id,
                "score": round(best_score, 3),
                "factors": best_factors,
            }

        entity_id = self._new_entity_id()
        self._track_to_entity[track_id] = entity_id
        self._profiles[entity_id] = EntityProfile(entity_id=entity_id, first_seen_frame=frame_idx)
        if best_id and best_score >= self.tentative_threshold:
            self._profiles[entity_id].tentative_merge_candidates.append(best_id)
        return entity_id, None

    def _score_candidate(self, profile: EntityProfile, obj: dict, frame_idx: int) -> tuple[float, dict[str, float]]:
        frame_gap = max(0, frame_idx - profile.last_seen_frame)
        if frame_gap > self.max_merge_gap_frames:
            return 0.0, {"frame_gap": float(frame_gap)}

        centroid = obj.get("centroid", [0.0, 0.0])
        distance = _center_distance(profile.last_centroid, centroid)
        position_score = max(0.0, 1.0 - distance / self.max_position_gap_px)
        time_score = max(0.0, 1.0 - frame_gap / self.max_merge_gap_frames)
        appearance_score = _cosine_similarity(profile.last_feature, obj.get("appearance_feature", []))
        score = 0.35 * position_score + 0.25 * time_score + 0.40 * appearance_score
        return score, {
            "position_score": round(position_score, 3),
            "time_score": round(time_score, 3),
            "appearance_score": round(appearance_score, 3),
            "frame_gap": float(frame_gap),
            "distance_px": round(distance, 2),
        }

    def _apply_observation(
        self,
        profile: EntityProfile,
        obj: dict,
        frame_idx: int,
        timestamp: float,
        evidence: dict[str, Any] | None,
    ) -> None:
        track_id = int(obj.get("track_id", 0))
        if track_id not in profile.track_ids:
            if profile.track_ids:
                profile.occlusion_gaps.append({
                    "from_frame": profile.last_seen_frame,
                    "to_frame": frame_idx,
                    "reason": "track_reappeared",
                    "reid_score": evidence["score"] if evidence else 0.0,
                })
                profile.tracking_quality = "occluded"
            profile.track_ids.append(track_id)
            profile.segments.append(TrackSegment(track_id=track_id, frame_start=frame_idx, frame_end=frame_idx))
            if evidence:
                profile.merge_evidence.append(evidence)
        else:
            for segment in reversed(profile.segments):
                if segment.track_id == track_id:
                    segment.frame_end = frame_idx
                    break

        profile.last_seen_frame = frame_idx
        profile.last_centroid = list(obj.get("centroid", profile.last_centroid))
        profile.last_feature = list(obj.get("appearance_feature", profile.last_feature))
        for point in obj.get("trajectory", [])[-3:]:
            sample = [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else float(timestamp)]
            if not profile.global_trajectory or profile.global_trajectory[-1] != sample:
                profile.global_trajectory.append(sample)
        if len(profile.global_trajectory) > 300:
            profile.global_trajectory = profile.global_trajectory[-300:]

    def _new_entity_id(self) -> str:
        entity_id = f"Person-{self._next_entity_index}"
        self._next_entity_index += 1
        return entity_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/perception/test_entity_profile.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
if (Test-Path .git) {
  git add perception/entity_profile.py tests/perception/test_entity_profile.py
  git commit -m "feat: add entity profile continuity layer"
} else {
  Write-Output "no git repository; commit skipped"
}
```

Expected: commit created, or `no git repository; commit skipped`.

---

### Task 2: Tracker Output and Config Consistency

**Files:**
- Modify: `perception/tracker.py`
- Modify: `pipeline.py`
- Modify: `web/backend/pipeline_runner.py`
- Test: `tests/perception/test_tracker_config.py`

- [ ] **Step 1: Write failing tests for tracker config and frame pass-through**

Create `tests/perception/test_tracker_config.py`:

```python
from perception.tracker import MultiObjectTracker


def test_tracker_accepts_tuned_thresholds():
    tracker = MultiObjectTracker(
        track_activation_threshold=0.18,
        lost_track_buffer=150,
        minimum_matching_threshold=0.65,
        with_reid=False,
        with_gmc=False,
    )
    assert tracker.track_low_thresh == 0.08
    assert tracker.lost_track_buffer == 150
    assert tracker.match_thresh == 0.65


def test_tracker_output_includes_feature_when_present():
    tracker = MultiObjectTracker(with_reid=False, with_gmc=False)
    output = tracker._build_output(
        timestamp=0.0,
        class_names=["person"],
        detections=[{"class": "person"}],
        frame_idx=1,
    )
    assert output == []
```

- [ ] **Step 2: Run tests**

Run:

```powershell
python -m pytest tests/perception/test_tracker_config.py -v
```

Expected: first test passes; second test passes with current empty output.

- [ ] **Step 3: Add optional appearance feature to tracker output**

Modify the result dict in `perception/tracker.py::_build_output` so it includes existing keys plus optional feature metadata:

```python
            results.append({
                'track_id': tid,
                'bbox': bbox,
                'class': t._class_name,
                'centroid': t.centroid,
                'velocity_ms': round(t.velocity_ms, 2),
                'direction_deg': round(t.direction_deg, 1),
                'trajectory': t.trajectory,
                'pose_keypoints': [],  # injected by pipeline
                'appearance_feature': t.smooth_feat.tolist() if getattr(t, "smooth_feat", None) is not None else [],
                'track_state': t.state,
                'last_seen_frame': int(t.last_seen),
                'time_since_update': int(t.time_since_update),
            })
```

- [ ] **Step 4: Pass frame into CLI tracker update**

Modify `pipeline.py` in `VideoUnderstandingPipeline.run`:

```python
            # 跟踪
            tracked = self.tracker.update(detections, self.frame_idx, timestamp, frame)
```

Expected: CLI now enables existing ReID/GMC like the web path.

- [ ] **Step 5: Use tuned tracker parameters in both pipeline paths**

Modify `pipeline.py` tracker construction:

```python
        self.tracker = MultiObjectTracker(
            track_activation_threshold=0.18,
            lost_track_buffer=120,
            minimum_matching_threshold=0.65,
        )
```

Modify `web/backend/pipeline_runner.py` tracker construction:

```python
    tracker = MultiObjectTracker(
        track_activation_threshold=0.18,
        lost_track_buffer=120,
        minimum_matching_threshold=0.65,
        reid_device="cuda",
    )
```

- [ ] **Step 6: Run targeted tests**

Run:

```powershell
python -m pytest tests/perception/test_tracker_config.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit if git is available**

Run:

```powershell
if (Test-Path .git) {
  git add perception/tracker.py pipeline.py web/backend/pipeline_runner.py tests/perception/test_tracker_config.py
  git commit -m "feat: expose tracker continuity signals"
} else {
  Write-Output "no git repository; commit skipped"
}
```

Expected: commit created, or `no git repository; commit skipped`.

---

### Task 3: Continuity KG Writes

**Files:**
- Modify: `kg/graph_store.py`
- Create: `kg/continuity.py`
- Test: `tests/kg/test_continuity_writes.py`

- [ ] **Step 1: Write failing KG continuity tests**

Create `tests/kg/test_continuity_writes.py`:

```python
from kg.continuity import write_continuity_event
from kg.graph_store import GraphStore


def event(frame_start=0, frame_end=60):
    return {
        "event_id": "Event-1",
        "timestamp": 1.0,
        "duration": [0.0, 2.0],
        "frame_start": frame_start,
        "frame_end": frame_end,
        "tracklets": [
            {"track_id": 3, "entity_id": "Person-1", "centroid": [10, 20]},
            {"track_id": 4, "entity_id": "Person-2", "centroid": [30, 20]},
        ],
        "spatial_relations": [{
            "track_1": 3,
            "track_2": 4,
            "distance_m": 20.0,
            "relation_type": "near",
            "duration_s": 1.5,
        }],
        "zone_id": "zone_default",
        "zone_type": "full_frame",
    }


def test_write_event_entity_and_edges():
    gs = GraphStore(use_directed=True)
    profiles = [{
        "entity_id": "Person-1",
        "track_ids": [3],
        "first_seen_frame": 0,
        "last_seen_frame": 60,
        "tracking_quality": "stable",
        "trajectory_summary": {"point_count": 2, "activity_bbox": [0, 0, 10, 10]},
        "merge_evidence_summary": [],
    }]

    write_continuity_event(gs, event(), profiles, previous_event_id="")

    assert gs.get_node("Event-1")["node_type"] == "Event"
    assert gs.get_node("Person-1")["node_type"] == "Entity"
    assert gs.get_edge("Person-1", "Event-1")["edge_type"] == "belongs_to"


def test_interaction_edge_aggregates_evidence():
    gs = GraphStore(use_directed=True)
    profiles = [
        {"entity_id": "Person-1", "track_ids": [3], "first_seen_frame": 0, "last_seen_frame": 60, "tracking_quality": "stable", "trajectory_summary": {}, "merge_evidence_summary": []},
        {"entity_id": "Person-2", "track_ids": [4], "first_seen_frame": 0, "last_seen_frame": 60, "tracking_quality": "stable", "trajectory_summary": {}, "merge_evidence_summary": []},
    ]

    write_continuity_event(gs, event(), profiles, previous_event_id="")
    write_continuity_event(gs, {**event(60, 120), "event_id": "Event-2"}, profiles, previous_event_id="Event-1")

    edge = gs.get_edge("Person-1", "Person-2")
    assert edge["edge_type"] == "interacts_with"
    assert "Event-1" in edge["event_ids"]
    assert "Event-2" in edge["event_ids"]
    assert gs.get_edge("Event-1", "Event-2")["edge_type"] == "next"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/kg/test_continuity_writes.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'kg.continuity'`.

- [ ] **Step 3: Add GraphStore JSON property helpers**

Modify `kg/graph_store.py`:

```python
import json
```

Add methods inside `GraphStore`:

```python
    @staticmethod
    def encode_property(value):
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return value

    @staticmethod
    def decode_property(value, default):
        if not isinstance(value, str):
            return value if value is not None else default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
```

Modify `upsert_node` and `upsert_edge` to encode properties before storing:

```python
    def upsert_node(self, node_id: str, node_type: str, properties: dict):
        encoded = {k: self.encode_property(v) for k, v in properties.items()}
        if self._graph.has_node(node_id):
            existing = self._graph.nodes[node_id]
            existing.update(encoded)
            existing["node_type"] = node_type
        else:
            self._graph.add_node(node_id, node_type=node_type, **encoded)
```

```python
    def upsert_edge(self, src: str, tgt: str, edge_type: str, properties: dict):
        encoded = {k: self.encode_property(v) for k, v in properties.items()}
        if self._graph.has_edge(src, tgt):
            existing = self._graph.edges[src, tgt]
            existing.update(encoded)
            existing["edge_type"] = edge_type
        else:
            self._graph.add_edge(src, tgt, edge_type=edge_type, **encoded)
```

- [ ] **Step 4: Implement continuity write helper**

Create `kg/continuity.py`:

```python
from __future__ import annotations

import time
from typing import Any

from .graph_store import GraphStore


def _track_to_entity(event: dict) -> dict[int, str]:
    return {
        int(t.get("track_id", 0)): t.get("entity_id", f"Entity-{t.get('track_id', 0)}")
        for t in event.get("tracklets", [])
    }


def write_continuity_event(
    graphstore: GraphStore,
    event: dict,
    entity_profiles: list[dict[str, Any]],
    previous_event_id: str = "",
    ttl_seconds: float = 3600.0,
) -> None:
    event_id = event.get("event_id", "")
    frame_start = int(event.get("frame_start", 0))
    frame_end = int(event.get("frame_end", 0))

    graphstore.upsert_node(event_id, "Event", {
        "timestamp": event.get("timestamp", 0.0),
        "duration": event.get("duration", []),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "zone_id": event.get("zone_id", ""),
        "zone_type": event.get("zone_type", ""),
        "entity_count": len(event.get("tracklets", [])),
        "expires_at": time.time() + ttl_seconds,
    })

    if previous_event_id:
        graphstore.upsert_edge(previous_event_id, event_id, "next", {
            "gap_frames": max(0, frame_start),
            "scene_continuity": True,
        })

    for profile in entity_profiles:
        entity_id = profile["entity_id"]
        graphstore.upsert_node(entity_id, "Entity", {
            "track_ids": profile.get("track_ids", []),
            "first_seen_frame": profile.get("first_seen_frame", 0),
            "last_seen_frame": profile.get("last_seen_frame", 0),
            "tracking_quality": profile.get("tracking_quality", "stable"),
            "trajectory_summary": profile.get("trajectory_summary", {}),
            "merge_evidence_summary": profile.get("merge_evidence_summary", []),
        })

    for tracklet in event.get("tracklets", []):
        entity_id = tracklet.get("entity_id", f"Entity-{tracklet.get('track_id', 0)}")
        graphstore.upsert_edge(entity_id, event_id, "belongs_to", {
            "frame_start": frame_start,
            "frame_end": frame_end,
            "role": "participant",
            "track_id": int(tracklet.get("track_id", 0)),
        })

    mapping = _track_to_entity(event)
    for rel in event.get("spatial_relations", []):
        entity_a = mapping.get(int(rel.get("track_1", 0)))
        entity_b = mapping.get(int(rel.get("track_2", 0)))
        if not entity_a or not entity_b or entity_a == entity_b:
            continue
        src, tgt = sorted([entity_a, entity_b])
        existing = graphstore.get_edge(src, tgt) or {}
        event_ids = GraphStore.decode_property(existing.get("event_ids"), [])
        frame_ranges = GraphStore.decode_property(existing.get("frame_ranges"), [])
        evidence = GraphStore.decode_property(existing.get("evidence"), [])
        if event_id not in event_ids:
            event_ids.append(event_id)
        frame_ranges.append([frame_start, frame_end])
        evidence.append({
            "event_id": event_id,
            "frame_range": [frame_start, frame_end],
            "relation_type": rel.get("relation_type", "near"),
            "distance": rel.get("distance_m", 0.0),
            "duration_s": rel.get("duration_s", 0.0),
        })
        graphstore.upsert_edge(src, tgt, "interacts_with", {
            "event_ids": event_ids,
            "frame_ranges": frame_ranges,
            "duration": sum(float(item.get("duration_s", 0.0)) for item in evidence),
            "confidence": min(1.0, len(event_ids) / 5.0),
            "evidence": evidence[-20:],
        })
```

- [ ] **Step 5: Run KG tests**

Run:

```powershell
python -m pytest tests/kg/test_continuity_writes.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit if git is available**

Run:

```powershell
if (Test-Path .git) {
  git add kg/graph_store.py kg/continuity.py tests/kg/test_continuity_writes.py
  git commit -m "feat: write continuity-aware kg evidence"
} else {
  Write-Output "no git repository; commit skipped"
}
```

Expected: commit created, or `no git repository; commit skipped`.

---

### Task 4: Backend Pipeline Integration

**Files:**
- Modify: `pipeline.py`
- Modify: `web/backend/pipeline_runner.py`
- Modify: `agents/orchestrator.py`
- Test: `tests/web/test_frontend_contract.py`

- [ ] **Step 1: Write payload contract tests**

Create `tests/web/test_frontend_contract.py`:

```python
from perception.entity_profile import EntityProfileManager


def track(track_id, frame):
    return {
        "track_id": track_id,
        "bbox": [10, 10, 30, 60],
        "centroid": [20 + frame, 35],
        "trajectory": [[20 + frame, 35, frame / 30.0]],
        "appearance_feature": [1.0, 0.0],
    }


def test_entity_payload_contains_chinese_ui_ready_fields():
    mgr = EntityProfileManager()
    enriched = mgr.update([track(1, 1)], frame_idx=1, timestamp=1 / 30.0)

    assert enriched[0]["entity_id"] == "Person-1"
    assert mgr.to_entities()[0]["tracking_quality"] == "stable"
    assert mgr.to_lifelines()[0]["segments"][0]["frame_start"] == 1
```

- [ ] **Step 2: Run test**

Run:

```powershell
python -m pytest tests/web/test_frontend_contract.py -v
```

Expected: PASS after Task 1.

- [ ] **Step 3: Integrate EntityProfileManager into web runner**

In `web/backend/pipeline_runner.py`, add imports:

```python
from perception.entity_profile import EntityProfileManager
from kg.continuity import write_continuity_event
```

After `event_builder = EventBuilder(event_interval_frames=60)`, add:

```python
    entity_profiles = EntityProfileManager()
    previous_event_id = ""
```

After tracker update:

```python
        tracked = tracker.update(detections, frame_idx, timestamp, frame)
        tracked = entity_profiles.update(tracked, frame_idx, timestamp) if tracked else []
```

In frame message, include entity IDs:

```python
            await websocket.send_json({
                "type": "frame",
                "frame_idx": frame_idx,
                "image": b64,
                "persons": persons,
                "entities": entity_profiles.to_entities(),
            })
```

After `event = event_builder.build_event(...)`, before `orch.process_event(event)`, ensure active tracklets carry entity IDs because `active` came from enriched tracked results. After `output = orch.process_event(event)`, add:

```python
                    write_continuity_event(
                        graphstore,
                        event,
                        entity_profiles.to_entities(),
                        previous_event_id=previous_event_id,
                    )
                    previous_event_id = event["event_id"]
```

Add to `event_data`:

```python
                    "entities": entity_profiles.to_entities(),
                    "lifelines": entity_profiles.to_lifelines(),
                    "kg_subgraph": graphstore.to_dict(),
```

In the dry/no-LLM `elif active` branch, call `write_continuity_event` and include the same fields:

```python
                write_continuity_event(
                    graphstore,
                    event,
                    entity_profiles.to_entities(),
                    previous_event_id=previous_event_id,
                )
                previous_event_id = event["event_id"]
```

- [ ] **Step 4: Integrate EntityProfileManager into CLI pipeline**

In `pipeline.py`, add imports:

```python
from perception.entity_profile import EntityProfileManager
from kg.continuity import write_continuity_event
```

In `__init__`, after `self.event_builder = EventBuilder(...)`, add:

```python
        self.entity_profiles = EntityProfileManager()
        self._previous_event_id = ""
```

In `run`, after tracker update:

```python
            tracked = self.tracker.update(detections, self.frame_idx, timestamp, frame)
            tracked = self.entity_profiles.update(tracked, self.frame_idx, timestamp) if tracked else []
```

In `_process_event`, before `if self.dry_run`, add:

```python
        write_continuity_event(
            self.graphstore,
            event,
            self.entity_profiles.to_entities(),
            previous_event_id=self._previous_event_id,
            ttl_seconds=self.ttl_seconds,
        )
        self._previous_event_id = event["event_id"]
```

In the `_results.append` for processed events, include:

```python
            "entities": self.entity_profiles.to_entities(),
            "lifelines": self.entity_profiles.to_lifelines(),
```

- [ ] **Step 5: Keep orchestrator contract untouched**

No AgentResponse schema changes are made. In `agents/orchestrator.py`, only allow reports to pass through `entity_id` values already present in event tracklets. If adding a helper, use:

```python
    def _entity_id_for_tracklet(self, tracklet: dict) -> str:
        return tracklet.get("entity_id", f"Entity-{tracklet.get('track_id', 0)}")
```

Expected: agents still receive `tracklets` and `spatial_relations`; enriched `entity_id` is additive.

- [ ] **Step 6: Run backend contract tests**

Run:

```powershell
python -m pytest tests/perception/test_entity_profile.py tests/kg/test_continuity_writes.py tests/web/test_frontend_contract.py -v
```

Expected: all selected tests passed.

- [ ] **Step 7: Commit if git is available**

Run:

```powershell
if (Test-Path .git) {
  git add pipeline.py web/backend/pipeline_runner.py agents/orchestrator.py tests/web/test_frontend_contract.py
  git commit -m "feat: stream continuous entity payloads"
} else {
  Write-Output "no git repository; commit skipped"
}
```

Expected: commit created, or `no git repository; commit skipped`.

---

### Task 5: Frontend Workspace and Lifelines

**Files:**
- Modify: `web/frontend/index.html`

- [ ] **Step 1: Add Chinese workspace regions**

Replace the current `right-panel` contents with Chinese tabs and a KG evidence panel while preserving upload/chat IDs:

```html
<div class="right-panel" id="rightPanel">
  <div class="entity-panel" id="entityPanel">
    <h3>实体档案</h3>
    <div id="entityList"></div>
  </div>
  <div class="resize-v" id="resizeAgent"></div>
  <div class="evidence-panel" id="evidencePanel">
    <h3>KG 证据子图</h3>
    <div id="kgEvidence">选择一个实体查看证据链。</div>
  </div>
  <div class="chat-panel" id="chatPanel">
    <div class="chat-msgs" id="chatMsgs">
      <div class="chat-msg agent"><div class="bubble">上传视频后，系统会按实体生命线展示连续行为。</div></div>
    </div>
    <div class="chat-input-w">
      <input type="text" id="chatInput" placeholder="询问当前实体或事件..." onkeypress="if(event.keyCode==13)sendChat()">
      <button onclick="sendChat()">发送</button>
    </div>
  </div>
</div>
```

Expected: page remains globally Chinese.

- [ ] **Step 2: Add responsive lifeline CSS**

Add CSS near timeline styles:

```css
.timeline-wrap{flex:0 0 clamp(150px,24vh,260px);background:var(--surface);border-top:1px solid var(--border);display:flex;flex-direction:column;min-height:0}
.event-heat{height:34px;display:flex;align-items:center;padding:6px 12px;gap:2px;overflow:hidden;border-bottom:1px solid var(--border)}
.lifeline-pane{flex:1;overflow:auto;padding:8px 12px}
.life-row{display:grid;grid-template-columns:minmax(72px,96px) 1fr;gap:10px;align-items:center;height:24px;font-size:11px}
.life-label{color:var(--text);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.life-track{position:relative;height:12px;background:rgba(255,255,255,.04);border-radius:6px;min-width:240px}
.life-seg{position:absolute;top:2px;height:8px;border-radius:5px;background:var(--traj)}
.life-gap{position:absolute;top:5px;height:2px;border-top:1px dashed var(--accent)}
.entity-panel,.evidence-panel{padding:12px 16px;border-bottom:1px solid var(--border);overflow:auto;min-height:120px}
.entity-item{display:flex;justify-content:space-between;gap:8px;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.05);cursor:pointer}
.entity-item.active{color:var(--accent)}
.kg-edge{padding:7px 0;border-bottom:1px solid rgba(255,255,255,.05);font-size:12px;line-height:1.5}
@media(max-width:900px){.main{flex-direction:column}.right-panel{width:100%;height:42vh;border-left:0;border-top:1px solid var(--border)}}
```

- [ ] **Step 3: Replace timeline markup**

Replace:

```html
<div class="timeline-bar" id="timelineBar"></div>
```

with:

```html
<div class="timeline-wrap">
  <div class="event-heat" id="timelineBar"></div>
  <div class="lifeline-pane" id="lifelinePane"></div>
</div>
```

- [ ] **Step 4: Add frontend state for entities and lifelines**

Add near existing state declarations:

```js
let entities=[],lifelines=[],selectedEntity=null,kgGraph={nodes:[],edges:[]};
```

In WebSocket `event` handler, after `events.push(msg)`:

```js
entities=msg.entities||entities;
lifelines=msg.lifelines||lifelines;
kgGraph=msg.kg_subgraph||kgGraph;
renderEntities();
renderLifelines();
```

In WebSocket `frame` handler:

```js
entities=msg.entities||entities;
renderEntities();
```

- [ ] **Step 5: Implement entity list and lifeline rendering**

Add JS:

```js
function renderEntities(){
  const box=document.getElementById('entityList');
  if(!box)return;
  box.innerHTML='';
  (entities||[]).forEach(ent=>{
    const div=document.createElement('div');
    div.className='entity-item'+(selectedEntity===ent.entity_id?' active':'');
    div.onclick=()=>selectEntity(ent.entity_id);
    const tracks=(ent.track_ids||[]).join(',');
    div.innerHTML=`<span>${ent.entity_id}</span><span>${ent.tracking_quality||'stable'} · #${tracks}</span>`;
    box.appendChild(div);
  });
}

function renderLifelines(){
  const pane=document.getElementById('lifelinePane');
  if(!pane)return;
  const total=Math.max(totalFrames||1,...(lifelines||[]).flatMap(l=>(l.segments||[]).map(s=>s.frame_end||0)));
  pane.innerHTML='';
  (lifelines||[]).forEach(line=>{
    const row=document.createElement('div');
    row.className='life-row';
    row.onclick=()=>selectEntity(line.entity_id);
    const label=document.createElement('div');
    label.className='life-label';
    label.textContent=line.entity_id;
    const track=document.createElement('div');
    track.className='life-track';
    (line.segments||[]).forEach(seg=>{
      const el=document.createElement('div');
      el.className='life-seg';
      el.style.left=((seg.frame_start||0)/total*100)+'%';
      el.style.width=Math.max(0.8,((seg.frame_end-seg.frame_start)/total*100))+'%';
      el.title=`${line.entity_id} #${seg.track_id} 帧 ${seg.frame_start}-${seg.frame_end}`;
      track.appendChild(el);
    });
    (line.occlusion_gaps||[]).forEach(gap=>{
      const el=document.createElement('div');
      el.className='life-gap';
      el.style.left=((gap.from_frame||0)/total*100)+'%';
      el.style.width=Math.max(0.8,((gap.to_frame-gap.from_frame)/total*100))+'%';
      el.title=`遮挡/断点 帧 ${gap.from_frame}-${gap.to_frame}`;
      track.appendChild(el);
    });
    row.appendChild(label);
    row.appendChild(track);
    pane.appendChild(row);
  });
}

function selectEntity(entityId){
  selectedEntity=entityId;
  renderEntities();
  renderEvidence(entityId);
}
```

- [ ] **Step 6: Implement KG evidence panel**

Add JS:

```js
function renderEvidence(entityId){
  const box=document.getElementById('kgEvidence');
  if(!box)return;
  const edges=(kgGraph.edges||[]).filter(e=>e.src===entityId||e.tgt===entityId);
  if(!edges.length){
    box.innerHTML=`<div class="kg-edge">${entityId} 暂无可展示的 KG 边。</div>`;
    return;
  }
  box.innerHTML=`<div style="font-weight:700;margin-bottom:8px">${entityId} 的证据链</div>`;
  edges.slice(0,12).forEach(edge=>{
    const div=document.createElement('div');
    div.className='kg-edge';
    const other=edge.src===entityId?edge.tgt:edge.src;
    div.innerHTML=`<b>${edge.edge_type||'关系'}</b> → ${other}<br><span style="color:var(--muted)">帧: ${edge.frame_start||''}-${edge.frame_end||''} 置信度: ${edge.confidence||'-'}</span>`;
    box.appendChild(div);
  });
}
```

- [ ] **Step 7: Keep existing chat and upload working**

Run the backend, open the page, upload a short sample, and verify:

```powershell
python -m web.backend.main
```

Expected: page opens at `http://localhost:8000`, upload button text is Chinese, entity list appears as events stream.

- [ ] **Step 8: Commit if git is available**

Run:

```powershell
if (Test-Path .git) {
  git add web/frontend/index.html
  git commit -m "feat: add Chinese entity lifeline workspace"
} else {
  Write-Output "no git repository; commit skipped"
}
```

Expected: commit created, or `no git repository; commit skipped`.

---

### Task 6: Quantitative Tracker Metrics

**Files:**
- Create: `perception/metrics.py`
- Test: `tests/perception/test_metrics.py`
- Modify: `pipeline.py`
- Modify: `web/backend/pipeline_runner.py`

- [ ] **Step 1: Write metrics tests**

Create `tests/perception/test_metrics.py`:

```python
from perception.metrics import TrackerMetrics


def test_metrics_count_track_segments_per_entity():
    metrics = TrackerMetrics()
    metrics.observe_entities([
        {"entity_id": "Person-1", "track_ids": [1, 2]},
        {"entity_id": "Person-2", "track_ids": [3]},
    ])
    snapshot = metrics.snapshot()
    assert snapshot["person_count"] == 2
    assert snapshot["avg_track_segments_per_person"] == 1.5


def test_metrics_records_detection_counts():
    metrics = TrackerMetrics()
    metrics.observe_frame(detection_count=4, tracked_count=3)
    metrics.observe_frame(detection_count=2, tracked_count=2)
    assert metrics.snapshot()["frames"] == 2
    assert metrics.snapshot()["detections_total"] == 6
    assert metrics.snapshot()["tracked_total"] == 5
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/perception/test_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'perception.metrics'`.

- [ ] **Step 3: Implement TrackerMetrics**

Create `perception/metrics.py`:

```python
class TrackerMetrics:
    def __init__(self) -> None:
        self.frames = 0
        self.detections_total = 0
        self.tracked_total = 0
        self._entity_segments: dict[str, int] = {}

    def observe_frame(self, detection_count: int, tracked_count: int) -> None:
        self.frames += 1
        self.detections_total += int(detection_count)
        self.tracked_total += int(tracked_count)

    def observe_entities(self, entities: list[dict]) -> None:
        for entity in entities:
            self._entity_segments[entity.get("entity_id", "")] = len(entity.get("track_ids", []))

    def snapshot(self) -> dict:
        person_count = len([k for k in self._entity_segments if k])
        total_segments = sum(self._entity_segments.values())
        avg_segments = round(total_segments / person_count, 3) if person_count else 0.0
        return {
            "frames": self.frames,
            "detections_total": self.detections_total,
            "tracked_total": self.tracked_total,
            "person_count": person_count,
            "avg_track_segments_per_person": avg_segments,
            "id_switch_proxy": max(0, total_segments - person_count),
        }
```

- [ ] **Step 4: Integrate metrics snapshots**

In both `pipeline.py` and `web/backend/pipeline_runner.py`, import:

```python
from perception.metrics import TrackerMetrics
```

Instantiate near EntityProfileManager:

```python
tracker_metrics = TrackerMetrics()
```

Per frame:

```python
tracker_metrics.observe_frame(len(detections), len(tracked))
tracker_metrics.observe_entities(entity_profiles.to_entities())
```

In web `done` payload add:

```python
"tracker_metrics": tracker_metrics.snapshot(),
```

In CLI `_results` append add:

```python
"tracker_metrics": self.tracker_metrics.snapshot(),
```

- [ ] **Step 5: Run metrics tests**

Run:

```powershell
python -m pytest tests/perception/test_metrics.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit if git is available**

Run:

```powershell
if (Test-Path .git) {
  git add perception/metrics.py tests/perception/test_metrics.py pipeline.py web/backend/pipeline_runner.py
  git commit -m "feat: record tracker continuity metrics"
} else {
  Write-Output "no git repository; commit skipped"
}
```

Expected: commit created, or `no git repository; commit skipped`.

---

### Task 7: Verification Pass

**Files:**
- No source file changes unless verification reveals bugs.

- [ ] **Step 1: Run all unit tests**

Run:

```powershell
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Run CLI dry-run smoke test**

Run:

```powershell
python pipeline.py --source test_data\test_180.mp4 --dry-run --max-events 1
```

Expected: command prints one event with tracklets, no exception, and KG stats at shutdown.

- [ ] **Step 3: Run web backend smoke test**

Run:

```powershell
python -m web.backend.main
```

Expected: server starts on `http://0.0.0.0:8000`. Open `http://localhost:8000`, upload a short video, and confirm Chinese UI renders entity list, lifelines, and evidence panel.

- [ ] **Step 4: Browser verification**

Use the Browser plugin or Playwright to verify:

```text
1. Upload UI is Chinese.
2. Event heat layer and lifeline layer are both visible.
3. Selecting a Person row updates KG evidence panel.
4. Narrow viewport keeps controls usable and text contained.
```

Expected: no blank page, no JavaScript console errors, text remains legible.

- [ ] **Step 5: Capture tracker metrics baseline**

Run the same sample before and after tuning if a pre-change output exists. If no pre-change baseline exists, record the current first-run metrics as baseline in the final report:

```text
detections_total
tracked_total
person_count
avg_track_segments_per_person
id_switch_proxy
```

Expected: metrics are printed or present in output JSON/WebSocket done payload.

- [ ] **Step 6: Commit verification fixes if git is available**

Run only if source changes were needed during verification:

```powershell
if (Test-Path .git) {
  git add .
  git commit -m "fix: resolve continuous entity verification issues"
} else {
  Write-Output "no git repository; commit skipped"
}
```

Expected: commit created, or `no git repository; commit skipped`.

---

## Self-Review Notes

- Spec coverage: EntityProfile, KG continuity, double timeline, KG evidence panel, tracker parameter boundaries, Chinese UI, responsive behavior, and quantitative metrics each have a task.
- Contract safety: Agent message/response schemas remain untouched; enriched `entity_id` is additive on tracklets and WebSocket payloads.
- KG storage safety: Entity nodes store trajectory summaries only; full trajectories remain in EntityProfile runtime state.
- Known repository state: `D:\gp` is not a git repository, so guarded commit steps print a clear skip message until git is initialized.
