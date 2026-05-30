import numpy as np
import pytest

from perception.tracker import MultiObjectTracker, Track


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


def test_tracker_output_empty_when_no_confirmed_tracks():
    tracker = MultiObjectTracker(with_reid=False, with_gmc=False)

    assert tracker._build_output(0.0, [], [], 0) == []


def test_confirmed_track_output_includes_continuity_signals_and_existing_keys():
    tracker = MultiObjectTracker(with_reid=False, with_gmc=False)
    feature = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    track = Track([10, 20, 30, 60], 0.9, feature, frame_idx=7, timestamp=0.25)
    track.state = "confirmed"
    track.last_seen = 9
    track.time_since_update = 2
    tracker.tracked_tracks.append(track)

    output = tracker._build_output(0.5, ["person"], [], 10)

    assert len(output) == 1
    result = output[0]
    assert {
        "track_id",
        "bbox",
        "class",
        "centroid",
        "velocity_ms",
        "direction_deg",
        "trajectory",
        "pose_keypoints",
    }.issubset(result)
    assert result["appearance_feature"] == feature.tolist()
    assert result["track_state"] == "confirmed"
    assert result["last_seen_frame"] == 9
    assert result["time_since_update"] == 2


def test_pipeline_safe_tracklets_strip_appearance_feature_without_mutating_source():
    from pipeline import strip_tracker_debug_fields

    tracked = [
        {
            "track_id": 1,
            "bbox": [1, 2, 3, 4],
            "appearance_feature": [0.1, 0.2],
            "track_state": "confirmed",
        }
    ]

    stripped = strip_tracker_debug_fields(tracked)

    assert "appearance_feature" not in stripped[0]
    assert tracked[0]["appearance_feature"] == [0.1, 0.2]
    assert stripped[0]["track_id"] == 1
    assert stripped[0]["track_state"] == "confirmed"


def test_resolve_runtime_fps_prefers_actual_video_fps_when_source_has_no_override():
    from pipeline import resolve_runtime_fps

    assert resolve_runtime_fps(0.0, 29.97) == pytest.approx(29.97)
    assert resolve_runtime_fps(None, 24.0) == pytest.approx(24.0)
    assert resolve_runtime_fps(30.0, 29.97) == pytest.approx(30.0)
    assert resolve_runtime_fps(0.0, 0.0) == pytest.approx(30.0)


def test_pipeline_uses_configured_detector_model_path(monkeypatch):
    import pipeline

    captured = {}

    class FakeDetector:
        def __init__(self, model_path, device, conf_threshold, classes, imgsz):
            captured["detector_model_path"] = model_path

    class FakePoseEstimator:
        def __init__(self, model_path, device):
            pass

    monkeypatch.setattr(pipeline, "YOLODetector", FakeDetector)
    monkeypatch.setattr(pipeline, "PoseEstimator", FakePoseEstimator)

    instance = pipeline.Pipeline(
        source="0",
        model_path="custom-detector.pt",
        dry_run=True,
    )

    assert instance.model_path == "custom-detector.pt"
    assert captured["detector_model_path"] == "custom-detector.pt"


def test_web_device_helper_uses_env_override(monkeypatch):
    from web.backend.pipeline_runner import select_inference_device

    monkeypatch.setenv("GP_INFERENCE_DEVICE", "cpu")

    assert select_inference_device() == "cpu"


def test_web_safe_tracklets_strip_appearance_feature_without_mutating_source():
    from web.backend.pipeline_runner import strip_tracker_debug_fields

    tracked = [{"track_id": 2, "appearance_feature": [0.3, 0.4], "bbox": [0, 1, 2, 3]}]

    stripped = strip_tracker_debug_fields(tracked)

    assert "appearance_feature" not in stripped[0]
    assert tracked[0]["appearance_feature"] == [0.3, 0.4]
    assert stripped[0]["track_id"] == 2
