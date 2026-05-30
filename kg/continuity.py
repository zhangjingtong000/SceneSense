from __future__ import annotations

import time
from typing import Any

from kg.graph_store import GraphStore


def write_continuity_event(
    graphstore: GraphStore,
    event: dict[str, Any],
    entity_profiles: list[dict[str, Any]],
    previous_event_id: str = "",
    ttl_seconds: float = 3600.0,
) -> None:
    event_id = str(event["event_id"])
    frame_start = int(event.get("frame_start", 0))
    frame_end = int(event.get("frame_end", frame_start))
    tracklets = event.get("tracklets", [])

    graphstore.upsert_node(
        event_id,
        "Event",
        {
            "timestamp": event.get("timestamp", 0.0),
            "duration": event.get("duration", []),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "zone_id": event.get("zone_id", "zone_default"),
            "zone_type": event.get("zone_type", "full_frame"),
            "entity_count": len({ _entity_id_for_tracklet(t) for t in tracklets }),
            "expires_at": time.time() + float(ttl_seconds),
        },
    )

    if previous_event_id:
        graphstore.upsert_edge(
            previous_event_id,
            event_id,
            "next",
            {"scene_continuity": True},
        )

    for profile in entity_profiles:
        _upsert_profile_summary(graphstore, profile)

    track_to_entity = {
        int(tracklet.get("track_id", 0)): _entity_id_for_tracklet(tracklet)
        for tracklet in tracklets
    }
    for tracklet in tracklets:
        entity_id = _entity_id_for_tracklet(tracklet)
        if not graphstore.has_node(entity_id):
            graphstore.upsert_node(entity_id, "Entity", {"entity_id": entity_id})
        graphstore.upsert_edge(
            entity_id,
            event_id,
            "belongs_to",
            {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "track_id": int(tracklet.get("track_id", 0)),
            },
        )

    for relation in event.get("spatial_relations", []):
        track_a = _relation_track_id(relation, "track_1", "track_a", "source_track_id")
        track_b = _relation_track_id(relation, "track_2", "track_b", "target_track_id")
        entity_a = track_to_entity.get(track_a, f"Entity-{track_a}")
        entity_b = track_to_entity.get(track_b, f"Entity-{track_b}")
        if entity_a == entity_b:
            continue

        src, tgt = sorted([entity_a, entity_b])
        _aggregate_interaction_edge(
            graphstore,
            src,
            tgt,
            event_id,
            frame_start,
            frame_end,
            track_a,
            track_b,
            relation,
        )


def _upsert_profile_summary(graphstore: GraphStore, profile: dict[str, Any]) -> None:
    entity_id = str(profile["entity_id"])
    properties = {
        "entity_id": entity_id,
        "track_ids": profile.get("track_ids", []),
        "first_seen_frame": profile.get("first_seen_frame", 0),
        "last_seen_frame": profile.get("last_seen_frame", 0),
        "tracking_quality": profile.get("tracking_quality", "stable"),
        "trajectory_summary": profile.get("trajectory_summary", {}),
        "merge_evidence_summary": profile.get("merge_evidence_summary", []),
        "tentative_merge_candidates": profile.get("tentative_merge_candidates", []),
    }
    graphstore.upsert_node(entity_id, "Entity", properties)


def _aggregate_interaction_edge(
    graphstore: GraphStore,
    src: str,
    tgt: str,
    event_id: str,
    frame_start: int,
    frame_end: int,
    track_a: int,
    track_b: int,
    relation: dict[str, Any],
) -> None:
    existing = graphstore.get_edge(src, tgt) or {}
    event_ids = _decoded_list(existing.get("event_ids"))
    frame_ranges = _decoded_list(existing.get("frame_ranges"))
    evidence = _decoded_list(existing.get("evidence"))
    track_pair = sorted([track_a, track_b])
    new_record = {
        "event_id": event_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "track_pair": track_pair,
        "relation_type": relation.get("relation_type", "near"),
        "distance_m": relation.get("distance_m"),
        "duration_s": relation.get("duration_s", 0.0),
    }
    evidence_by_key = {
        _evidence_key(item): item
        for item in evidence
        if isinstance(item, dict)
    }
    evidence_by_key[_evidence_key(new_record)] = new_record
    evidence = list(evidence_by_key.values())
    event_ids = sorted({str(item["event_id"]) for item in evidence})
    frame_ranges = []
    for item in evidence:
        frame_range = [int(item.get("frame_start", 0)), int(item.get("frame_end", 0))]
        if frame_range not in frame_ranges:
            frame_ranges.append(frame_range)
    duration = sum(float(item.get("duration_s", 0.0) or 0.0) for item in evidence)

    graphstore.upsert_edge(
        src,
        tgt,
        "interacts_with",
        {
            "event_ids": event_ids,
            "frame_ranges": frame_ranges,
            "duration": round(duration, 3),
            "confidence": min(1.0, round(len(event_ids) / 5.0, 3)),
            "evidence": evidence[-20:],
        },
    )


def _entity_id_for_tracklet(tracklet: dict[str, Any]) -> str:
    track_id = tracklet.get("track_id", 0)
    return str(tracklet.get("entity_id") or f"Entity-{track_id}")


def _relation_track_id(relation: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in relation:
            return int(relation[key])
    return 0


def _decoded_list(value: Any) -> list[Any]:
    decoded = GraphStore.decode_property(value)
    return list(decoded) if isinstance(decoded, list) else []


def _evidence_key(item: dict[str, Any]) -> tuple[Any, ...]:
    track_pair = item.get("track_pair")
    if not isinstance(track_pair, list):
        track_pair = sorted([
            int(item.get("track_1", 0)),
            int(item.get("track_2", 0)),
        ])
        item["track_pair"] = track_pair
        item.pop("track_1", None)
        item.pop("track_2", None)
    return (
        item.get("event_id"),
        int(item.get("frame_start", 0)),
        int(item.get("frame_end", 0)),
        tuple(track_pair),
        item.get("relation_type", "near"),
    )
