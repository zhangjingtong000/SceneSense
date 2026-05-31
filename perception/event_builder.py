import hashlib
import json
import math

import cv2
import numpy as np

from utils.serialization import to_jsonable


class EventBuilder:
    _next_id = 1

    def __init__(
        self,
        event_interval_frames: int = 30,
        spatial_neighbor_threshold_m: float = 3.0,
    ):
        self.event_interval_frames = event_interval_frames
        self.spatial_neighbor_threshold_m = spatial_neighbor_threshold_m

    def build_event(
        self,
        tracked_objects: list[dict],
        frame_start: int,
        frame_end: int,
        fps: float,
        zones: dict[str, list[tuple[float, float]]] | None = None,
    ) -> dict:
        time_start = frame_start / fps
        time_end = frame_end / fps
        timestamp = (time_start + time_end) / 2.0

        # 去重：每个 track_id 只保留最新一条（含最完整轨迹历史）
        latest: dict[int, dict] = {}
        for obj in tracked_objects:
            tid = obj["track_id"]
            latest[tid] = obj
        tracklets = list(latest.values())

        zone_id = "zone_default"
        zone_type = "full_frame"
        if zones:
            zone_counts: dict[str, int] = {}
            for obj in tracklets:
                zid = self._determine_zone(obj["centroid"], zones)
                zone_counts[zid] = zone_counts.get(zid, 0) + 1
            if zone_counts:
                zone_id = max(zone_counts, key=zone_counts.get)
                zone_type = zone_id

        event = {
            "event_id": "",
            "timestamp": round(timestamp, 2),
            "duration": [round(time_start, 2), round(time_end, 2)],
            "frame_start": frame_start,
            "frame_end": frame_end,
            "fps": fps,
            "tracklets": tracklets,
            "spatial_relations": self._compute_spatial_relations(tracklets),
            "zone_id": zone_id,
            "zone_type": zone_type,
        }
        event = to_jsonable(event)

        event["event_id"] = f"Event-{EventBuilder._next_id}"
        EventBuilder._next_id += 1

        return event

    def _compute_spatial_relations(
        self, tracklets: list[dict]
    ) -> list[dict]:
        relations: list[dict] = []
        for i, obj1 in enumerate(tracklets):
            for obj2 in tracklets[i + 1:]:
                c1 = obj1["centroid"]
                c2 = obj2["centroid"]
                dist = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
                if dist < self.spatial_neighbor_threshold_m:
                    t1 = obj1["trajectory"]
                    t2 = obj2["trajectory"]
                    if len(t1) >= 2 and len(t2) >= 2:
                        overlap_start = max(t1[0][2], t2[0][2])
                        overlap_end = min(t1[-1][2], t2[-1][2])
                        duration_s = round(max(0.0, overlap_end - overlap_start), 2)
                    else:
                        duration_s = 0.0
                    relations.append({
                        "track_1": obj1["track_id"],
                        "track_2": obj2["track_id"],
                        "distance_m": round(dist, 2),
                        "relation_type": "near",
                        "duration_s": duration_s,
                    })
        return relations

    def _determine_zone(
        self,
        centroid: tuple[float, float] | list[float],
        zones: dict[str, list[tuple[float, float]]],
    ) -> str:
        if not zones:
            return "zone_default"
        for zone_id, polygon in zones.items():
            poly = np.array(polygon, dtype=np.float32)
            if cv2.pointPolygonTest(poly, (float(centroid[0]), float(centroid[1])), False) >= 0:
                return zone_id
        return "zone_default"
