import asyncio

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
        reid_interval=2,
    )

    assert tracker.track_low_thresh == 0.08
    assert tracker.lost_track_buffer == 150
    assert tracker.match_thresh == 0.65
    assert tracker.reid_interval == 2


def test_tracker_reid_interval_skips_embedding_refresh_without_erasing_feature(monkeypatch):
    import perception.tracker as tracker_module

    class FakeReIDExtractor:
        def __init__(self, device):
            self.calls = 0

        def extract(self, frame, bboxes):
            self.calls += 1
            return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(bboxes), 1))

    monkeypatch.setattr(tracker_module, "ReIDExtractor", FakeReIDExtractor)
    tracker = tracker_module.MultiObjectTracker(with_reid=True, with_gmc=False, reid_interval=2)
    detection = {"bbox": [10, 20, 30, 60], "confidence": 0.9, "class": "person"}
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    tracker.update([detection], frame_idx=0, timestamp=0.0, frame=frame)
    first_feature = tracker.tracked_tracks[0].smooth_feat.copy()
    tracker.update([detection], frame_idx=1, timestamp=0.04, frame=frame)
    tracker.update([detection], frame_idx=2, timestamp=0.08, frame=frame)

    assert tracker.reid.calls == 2
    assert np.array_equal(tracker.tracked_tracks[0].smooth_feat, first_feature)


def test_reid_extractor_uses_half_precision_on_cuda(monkeypatch):
    import perception.tracker as tracker_module

    class FakeBackbone:
        pass

    class FakeFeatureExtractor:
        def __init__(self):
            self.half_called = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def half(self):
            self.half_called = True
            return self

    fake_feature_extractor = FakeFeatureExtractor()
    monkeypatch.setattr(tracker_module, "resnet18", lambda weights: FakeBackbone())
    monkeypatch.setattr(
        tracker_module,
        "create_feature_extractor",
        lambda backbone, return_nodes: fake_feature_extractor,
    )

    extractor = tracker_module.ReIDExtractor(device="cuda", backbone="resnet18")

    assert extractor.use_half is True
    assert extractor.model.half_called is True


def test_tracker_output_empty_when_no_confirmed_tracks():
    tracker = MultiObjectTracker(with_reid=False, with_gmc=False)

    assert tracker._build_output(0.0, [], [], 0) == []


def test_track_confirms_on_third_consecutive_observation():
    feature = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    track = Track([10, 20, 30, 60], 0.9, feature, frame_idx=0, timestamp=0.0)

    track.update([11, 20, 31, 60], 0.91, feature, frame_idx=1, timestamp=0.04)
    track.update([12, 20, 32, 60], 0.92, feature, frame_idx=2, timestamp=0.08)

    assert track.is_confirmed()
    assert track.hits == 3


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


def test_web_tracker_factory_disables_gmc_for_static_camera_latency():
    from web.backend.pipeline_runner import create_web_tracker

    tracker = create_web_tracker("cpu")

    assert tracker._with_reid is True
    assert tracker._with_gmc is False
    assert tracker.gmc is None
    assert tracker.reid_interval == 2


def test_web_unmatched_detections_excludes_boxes_covered_by_tracks():
    from web.backend.pipeline_runner import find_unmatched_detections

    tracked = [{"bbox": [0, 0, 100, 100]}]
    detections = [
        {"bbox": [5, 5, 98, 98], "confidence": 0.9},
        {"bbox": [200, 200, 260, 320], "confidence": 0.42},
    ]

    unmatched = find_unmatched_detections(detections, tracked)

    assert unmatched == [detections[1]]


def test_web_display_track_memory_keeps_recent_track_visible_briefly():
    from web.backend.pipeline_runner import smooth_display_tracks

    memory = {}
    tracked = [{"track_id": 7, "bbox": [10, 20, 40, 80], "entity_id": "Entity-7"}]

    visible_now = smooth_display_tracks(tracked, memory, frame_idx=10, max_age=3)
    visible_next = smooth_display_tracks([], memory, frame_idx=11, max_age=3)

    assert visible_now[0]["track_state"] != "display_gap"
    assert visible_next[0]["track_id"] == 7
    assert visible_next[0]["track_state"] == "display_gap"
    assert visible_next[0]["display_age"] == 1


def test_web_annotated_frame_exposes_unmatched_detection_candidates():
    import numpy as np
    from web.backend.pipeline_runner import make_annotated_frame

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = [{"bbox": [20, 30, 50, 90], "confidence": 0.42}]

    _, persons = make_annotated_frame(frame, [], detections)

    assert len(persons) == 1
    assert persons[0]["candidate"] is True
    assert persons[0]["entity_id"] == "候选检测"


def test_web_candidate_tracker_promotes_repeated_unmatched_detection():
    from web.backend.pipeline_runner import CandidateDisplayTracker

    tracker = CandidateDisplayTracker(min_hits=3, max_age=2, iou_threshold=0.4)
    detections = [{"bbox": [20, 30, 50, 90], "confidence": 0.42}]

    first = tracker.update(detections, frame_idx=1)
    second = tracker.update([{"bbox": [21, 30, 51, 90], "confidence": 0.44}], frame_idx=2)
    third = tracker.update([{"bbox": [22, 30, 52, 90], "confidence": 0.46}], frame_idx=3)

    assert first[0]["entity_id"] == "候选检测"
    assert second[0]["entity_id"] == "候选检测"
    assert third[0]["entity_id"] == "Tentative-1"
    assert third[0]["tentative_entity"] is True
    assert third[0]["candidate_hits"] == 3


def test_web_annotated_frame_keeps_tentative_candidate_identity_and_pose():
    import numpy as np
    from web.backend.pipeline_runner import make_annotated_frame

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    candidate = {
        "track_id": "cand-1",
        "bbox": [20, 30, 50, 90],
        "entity_id": "Tentative-1",
        "candidate": True,
        "tentative_entity": True,
        "confidence": 0.5,
        "pose_keypoints": [[25, 35, 0.9] for _ in range(17)],
    }

    _, persons = make_annotated_frame(frame, [], candidate_objects=[candidate])

    assert persons[0]["entity_id"] == "Tentative-1"
    assert persons[0]["tentative_entity"] is True
    assert persons[0]["pose_keypoints"] == candidate["pose_keypoints"]


def test_web_pose_targets_prioritize_stable_tracks_and_limit_candidates():
    from web.backend.pipeline_runner import build_pose_targets

    stable = [
        {"track_id": 1, "entity_id": "Person-1", "bbox": [0, 0, 20, 80]},
        {"track_id": 2, "entity_id": "Person-2", "bbox": [30, 0, 50, 80]},
    ]
    candidates = [
        {
            "track_id": f"cand-{idx}",
            "entity_id": f"Tentative-{idx}",
            "bbox": [idx, idx, idx + 10, idx + 30],
            "tentative_entity": True,
        }
        for idx in range(5)
    ]

    targets = build_pose_targets(stable, candidates, frame_idx=12, max_candidates=2)

    assert [target["entity_id"] for target in targets] == [
        "Person-1",
        "Person-2",
        "Tentative-0",
        "Tentative-1",
    ]
    assert all(target["frame_idx"] == 12 for target in targets)


def test_web_pose_updates_are_sent_without_waiting_for_video_delay():
    from web.backend.pipeline_runner import drain_pose_updates

    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send_json(self, message):
            self.messages.append(message)

    async def run():
        results = asyncio.Queue(maxsize=1)
        websocket = FakeWebSocket()
        await results.put({"type": "pose_update", "frame_idx": 1, "poses": []})
        await drain_pose_updates(results, websocket)
        return websocket.messages

    messages = asyncio.run(run())

    assert messages == [{"type": "pose_update", "frame_idx": 1, "poses": []}]


def test_web_pose_worker_replaces_stale_result_in_full_queue():
    from web.backend.pipeline_runner import pose_worker

    class FakePoseEstimator:
        def estimate(self, frame, bboxes):
            return [[[1.0, 2.0, 0.9] for _ in range(17)] for _ in bboxes]

    async def run():
        requests = asyncio.Queue()
        results = asyncio.Queue(maxsize=1)
        await results.put({"type": "pose_update", "frame_idx": 1, "poses": []})
        worker = asyncio.create_task(pose_worker(FakePoseEstimator(), requests, results))
        await requests.put({
            "frame_idx": 9,
            "frame": np.zeros((20, 20, 3), dtype=np.uint8),
            "targets": [{
                "target_id": "Person-1",
                "track_id": 1,
                "entity_id": "Person-1",
                "bbox": [0, 0, 10, 10],
            }],
        })
        await asyncio.wait_for(requests.put(None), timeout=1)
        await asyncio.wait_for(worker, timeout=1)
        return results.get_nowait()

    result = asyncio.run(run())

    assert result["frame_idx"] == 9
    assert result["poses"][0]["entity_id"] == "Person-1"


def test_web_frame_writer_flushes_all_queued_frames(monkeypatch):
    import web.backend.pipeline_runner as pipeline_runner

    written = []

    def fake_write(path, frame):
        written.append((str(path), int(frame[0, 0, 0])))

    monkeypatch.setattr(pipeline_runner, "write_annotated_frame", fake_write)

    async def run():
        queue = asyncio.Queue(maxsize=2)
        worker = asyncio.create_task(pipeline_runner.frame_writer_worker(queue))
        await queue.put(("000001.jpg", np.full((2, 2, 3), 1, dtype=np.uint8)))
        await queue.put(("000002.jpg", np.full((2, 2, 3), 2, dtype=np.uint8)))
        await queue.put(None)
        await asyncio.wait_for(worker, timeout=1)

    asyncio.run(run())

    assert written == [("000001.jpg", 1), ("000002.jpg", 2)]
