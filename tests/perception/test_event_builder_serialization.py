import json

import numpy as np

from perception.event_builder import EventBuilder


def test_event_builder_converts_numpy_scalars_before_hashing_and_agent_payload():
    builder = EventBuilder(spatial_neighbor_threshold_m=100.0)

    event = builder.build_event(
        [
            {
                "track_id": np.int64(1),
                "bbox": np.array([10, 20, 30, 60], dtype=np.float32),
                "class": "person",
                "centroid": [np.float32(20.0), np.float32(40.0)],
                "velocity_ms": np.float32(1.25),
                "direction_deg": np.float32(90.0),
                "trajectory": [
                    [np.float32(20.0), np.float32(40.0), np.float32(0.0)],
                    [np.float32(21.0), np.float32(41.0), np.float32(1.0)],
                ],
                "pose_keypoints": [],
                "entity_id": "Person-1",
            },
            {
                "track_id": np.int64(2),
                "bbox": np.array([35, 20, 55, 60], dtype=np.float32),
                "class": "person",
                "centroid": [np.float32(45.0), np.float32(40.0)],
                "velocity_ms": np.float32(0.75),
                "direction_deg": np.float32(180.0),
                "trajectory": [
                    [np.float32(45.0), np.float32(40.0), np.float32(0.0)],
                    [np.float32(44.0), np.float32(40.0), np.float32(1.0)],
                ],
                "pose_keypoints": [],
                "entity_id": "Person-2",
            },
        ],
        frame_start=0,
        frame_end=30,
        fps=np.float32(30.0),
    )

    json.dumps(event)

    assert event["tracklets"][0]["track_id"] == 1
    assert isinstance(event["tracklets"][0]["trajectory"][0][0], float)
    assert event["spatial_relations"][0]["distance_m"] == 25.0
