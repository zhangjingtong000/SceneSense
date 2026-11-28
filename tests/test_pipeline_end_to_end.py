"""端到端：合成视频 → 感知层（假检测/假跟踪）→ 事件 → 三个 Agent → KG → GraphML。

检测器与跟踪器被替换成确定性的假实现，所以这条链路不依赖 CUDA、YOLO 权重和网络；
Agent 用真的实现，LLM 换成 stub（交互 Agent 在规则置信度 >= 0.6 时不会调它）。
"""

import cv2
import numpy as np
import pytest

import pipeline as pipeline_module
from pipeline import Pipeline


class FakeDetector:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [
            {"track_id": 0, "bbox": [10, 10, 40, 90], "score": 0.9, "class": "person"},
            {"track_id": 1, "bbox": [45, 10, 75, 90], "score": 0.9, "class": "person"},
        ]


class FakePoseEstimator:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def estimate(self, frame, bboxes):
        self.calls += 1
        return [[] for _ in bboxes]


class FakeTracker:
    """固定返回两条并排向右走的轨迹。"""

    def __init__(self, *args, **kwargs):
        self.frames = 0

    def update(self, detections, frame_idx, timestamp, frame):
        self.frames += 1
        return [self._track(1, 10 + frame_idx, timestamp), self._track(2, 40 + frame_idx, timestamp)]

    def _track(self, track_id, x, timestamp):
        return {
            "track_id": track_id,
            "bbox": [x, 10, x + 30, 90],
            "score": 0.9,
            "class": "person",
            "centroid": [x + 15, 50],
            "velocity_ms": 30.0,
            "direction_deg": 0.0,
            "trajectory": [
                [x + 5, 50, max(0.0, timestamp - 0.1)],
                [x + 15, 50, timestamp],
            ],
            "appearance_feature": [0.1] * 8,
        }

    def get_active_tracks(self):
        return {1, 2}


class StubLLM:
    """离线 LLM：交互精判直接失败，Agent 回落到规则结论。"""

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def chat_with_json_output(self, *args, **kwargs):
        self.calls += 1
        return {"error": "api_error", "raw": "stub"}

    def chat(self, *args, **kwargs):
        self.calls += 1
        return ""

    def generate(self, *args, **kwargs):
        self.calls += 1
        return ""


def write_synthetic_video(path, frames=40, size=96):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (size, size))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.full((size, size, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (10, 10), (80, 90), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()


@pytest.fixture()
def video_path(tmp_path):
    path = tmp_path / "synth.mp4"
    write_synthetic_video(path)
    return path


@pytest.fixture()
def fake_perception(monkeypatch):
    monkeypatch.setattr(pipeline_module, "YOLODetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "PoseEstimator", FakePoseEstimator)
    monkeypatch.setattr(pipeline_module, "MultiObjectTracker", FakeTracker)
    monkeypatch.setattr(pipeline_module, "LLMClient", StubLLM)


def make_pipeline(video_path, tmp_path, **overrides):
    kwargs = dict(
        source=str(video_path),
        event_interval_frames=30,
        kg_storage_path=str(tmp_path / "scene.graphml"),
        output_log=str(tmp_path / "results.json"),
        dry_run=False,
    )
    kwargs.update(overrides)
    return Pipeline(**kwargs)


def test_dry_run_chain_produces_event_and_kg_without_agents(video_path, tmp_path, fake_perception):
    pipes = make_pipeline(video_path, tmp_path, dry_run=True)

    pipes.run()

    assert len(pipes._results) == 1
    assert pipes._results[0]["tracklet_count"] == 2
    assert pipes.graphstore.query_nodes(node_type="Event")
    assert (tmp_path / "scene.graphml").exists()
    assert (tmp_path / "results.json").exists()


def test_full_chain_runs_three_agents_and_writes_kg(video_path, tmp_path, fake_perception):
    pipes = make_pipeline(video_path, tmp_path)

    pipes.run()

    assert pipes.orchestrator.route_order == [
        "trajectory_agent",
        "interaction_agent",
        "summary_agent",
    ]
    output = pipes._results[0]["output"]
    assert output["event_id"] == "Event-1"
    assert output["scene_report"]["summary"]
    assert [step["agent"] for step in output["reasoning_trail"]] == pipes.orchestrator.route_order

    entities = pipes.graphstore.query_nodes(node_type="Entity")
    assert {e["node_id"] for e in entities} == {"Person-1", "Person-2"}
    assert pipes.graphstore.query_edges(edge_type="belongs_to")
    assert pipes.graphstore.query_edges(edge_type="next") == []


def test_anomaly_agent_is_inserted_before_summary_in_full_chain(
    video_path, tmp_path, fake_perception
):
    pipes = make_pipeline(video_path, tmp_path, with_anomaly=True)

    pipes.run()

    assert pipes.orchestrator.route_order == [
        "trajectory_agent",
        "interaction_agent",
        "anomaly_agent",
        "summary_agent",
    ]
    output = pipes._results[0]["output"]
    assert "anomaly_agent_confidence" in output["agent_summary"]


def test_annotate_creates_missing_output_directory(video_path, tmp_path, fake_perception, capsys):
    """README 的 --annotate-path outputs/demo.mp4：全新 clone 时 outputs/ 不存在。

    cv2.VideoWriter 不会自己建目录，缺目录时 isOpened() 返回 False、write() 全丢。
    这里锁两件事：目录必须被自动创建；只有真的写进去过帧才打印"已保存"。
    """
    out = tmp_path / "outputs" / "nested" / "demo.mp4"
    assert not out.parent.exists()

    pipes = make_pipeline(
        video_path, tmp_path, dry_run=True, annotate=True, annotate_path=str(out)
    )
    pipes.run()

    assert out.parent.is_dir()
    assert out.is_file() and out.stat().st_size > 0
    assert pipes._annotated_frames == 40
    captured = capsys.readouterr()
    assert f"Annotated video saved to {out} (40 frames)" in captured.out
    assert captured.err == ""


class BrokenVideoWriter:
    """模拟目录不可写等导致 cv2.VideoWriter 打不开的情况。"""

    def __init__(self, *args, **kwargs):
        self.released = False

    def isOpened(self):  # noqa: N802 - 与 OpenCV API 同名
        return False

    def write(self, frame):
        raise AssertionError("打不开的 writer 不应该收到帧")

    def release(self):
        self.released = True


def test_annotate_writer_failure_is_reported_without_claiming_saved(
    video_path, tmp_path, fake_perception, monkeypatch, capsys
):
    """writer.isOpened() == False 时必须明确报错，且绝不能打印"已保存"。"""
    monkeypatch.setattr(pipeline_module.cv2, "VideoWriter", BrokenVideoWriter)
    out = tmp_path / "outputs" / "demo.mp4"

    pipes = make_pipeline(
        video_path, tmp_path, dry_run=True, annotate=True, annotate_path=str(out)
    )
    pipes.run()

    captured = capsys.readouterr()
    assert "Annotated video saved" not in captured.out
    assert "Annotated video saved" not in captured.err
    assert "无法打开" in captured.err
    assert str(out) in captured.err
    assert pipes._writer is None
    assert pipes._annotated_frames == 0
    assert pipes.annotate is False
    assert not out.exists()  # 目录建了，但没有视频文件
    assert "错误" in captured.err
