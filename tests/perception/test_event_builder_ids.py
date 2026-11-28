"""EventBuilder 的事件编号是类变量：Web 端同一进程连跑多个视频必须能重置。"""

from perception.event_builder import EventBuilder


def make_track(track_id=1):
    return {
        "track_id": track_id,
        "bbox": [0, 0, 30, 90],
        "class": "person",
        "centroid": [15.0, 45.0],
        "velocity_ms": 10.0,
        "direction_deg": 0.0,
        "trajectory": [[15.0, 45.0, 0.0], [16.0, 45.0, 0.1]],
        "pose_keypoints": [],
        "entity_id": f"Person-{track_id}",
    }


def build(builder, track_id=1):
    return builder.build_event([make_track(track_id)], frame_start=0, frame_end=30, fps=30.0)


def test_event_ids_increment_across_events():
    EventBuilder.reset_ids()
    builder = EventBuilder()

    first = build(builder, 1)
    second = build(builder, 2)

    assert first["event_id"] == "Event-1"
    assert second["event_id"] == "Event-2"


def test_reset_ids_restarts_numbering_for_the_next_video():
    EventBuilder.reset_ids()
    builder = EventBuilder()

    build(builder, 1)
    build(builder, 2)
    EventBuilder.reset_ids()

    assert build(builder, 3)["event_id"] == "Event-1"
