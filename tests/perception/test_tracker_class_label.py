"""检测类别必须跟着轨迹走，不能被写死成 person。"""

import numpy as np

from perception.tracker import MultiObjectTracker, Track


def _frame():
    return np.zeros((80, 80, 3), dtype=np.uint8)


def _det(bbox, cls, conf=0.9):
    return {"bbox": bbox, "class": cls, "class_id": 0, "confidence": conf}


def _run(tracker, detections, frames=4):
    outputs = []
    for i in range(frames):
        outputs = tracker.update(detections, frame_idx=i, timestamp=i / 30.0, frame=_frame())
    return outputs


def test_car_tracklet_outputs_car_not_person():
    """--classes 0,2 跑车时，输出类别只能是 car。"""
    tracker = MultiObjectTracker(with_reid=False, with_gmc=False)

    outputs = _run(tracker, [_det([10, 10, 40, 70], "car")])

    assert outputs, "轨迹应已被确认"
    assert outputs[0]["class"] == "car"


def test_person_and_car_tracks_keep_their_own_class():
    tracker = MultiObjectTracker(with_reid=False, with_gmc=False)

    outputs = _run(tracker, [
        _det([10, 10, 40, 70], "person"),
        _det([50, 10, 75, 70], "car"),
    ])

    labels = sorted(t["class"] for t in outputs)
    assert labels == ["car", "person"]


def test_track_default_class_is_person():
    """直接构造 Track（测试/回放路径）时默认仍是 person。"""
    track = Track([10, 10, 40, 70], 0.9, None, 0, 0.0)

    assert track._class_name == "person"


def test_track_update_can_change_class():
    track = Track([10, 10, 40, 70], 0.9, None, 0, 0.0, class_name="car")

    track.update([11, 10, 41, 70], 0.9, None, 1, 0.04, class_name="car")

    assert track._class_name == "car"
