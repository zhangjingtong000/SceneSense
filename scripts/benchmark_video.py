"""Run a repeatable detector/tracker/entity/KG benchmark on one video.

Outputs:
  1. Stable JSON for diffing between parameter runs.
  2. Compact terminal tables for quick human inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg.continuity import write_continuity_event
from kg.graph_store import GraphStore
from perception.detector import YOLODetector
from perception.entity_profile import EntityProfileManager
from perception.event_builder import EventBuilder
from perception.metrics import TrackerMetrics
from perception.tracker import MultiObjectTracker
from utils.serialization import to_jsonable
from web.backend.pipeline_runner import select_inference_device, strip_tracker_debug_fields


SCHEMA_VERSION = "benchmark-video-v1"


def parse_classes(value: str) -> list[int] | None:
    if not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def benchmark_video(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = video_fps if video_fps > 0 else float(args.fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    device = args.device or select_inference_device()
    classes = parse_classes(args.classes)
    detector = YOLODetector(
        model_path=args.model,
        device=device,
        conf_threshold=args.conf,
        classes=classes,
        imgsz=args.imgsz,
    )
    tracker = MultiObjectTracker(
        track_activation_threshold=args.track_activation_threshold,
        lost_track_buffer=args.lost_buffer,
        minimum_matching_threshold=args.matching_threshold,
        reid_device=device,
        with_reid=args.with_reid,
        with_gmc=not args.no_gmc,
    )
    profiles = EntityProfileManager()
    metrics = TrackerMetrics()
    event_builder = EventBuilder(
        event_interval_frames=args.event_interval,
        spatial_neighbor_threshold_m=args.spatial_threshold,
    )
    graph = GraphStore(use_directed=True)

    start_time = time.perf_counter()
    frame_idx = 0
    event_start = 0
    previous_event_id = ""
    tracker_results: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        timestamp = frame_idx / fps
        detections = detector.detect(frame)
        tracked = tracker.update(detections, frame_idx, timestamp, frame)
        tracked = profiles.update(tracked, frame_idx, timestamp) if tracked else []
        metrics.observe(detections, tracked, profiles.to_entities())

        if tracked:
            tracker_results.extend(strip_tracker_debug_fields(tracked))

        frame_idx += 1
        if frame_idx - event_start >= args.event_interval:
            active = [
                item for item in tracker_results
                if item["track_id"] in tracker.get_active_tracks()
            ]
            if active:
                event = event_builder.build_event(active, event_start, frame_idx, fps)
                write_continuity_event(
                    graph,
                    event,
                    profiles.to_entities(),
                    previous_event_id=previous_event_id,
                )
                previous_event_id = event["event_id"]
                events.append(summarize_event(event))
            tracker_results.clear()
            event_start = frame_idx

    cap.release()

    elapsed = round(time.perf_counter() - start_time, 3)
    processed_fps = round(frame_idx / elapsed, 3) if elapsed > 0 else 0.0

    return to_jsonable({
        "schema_version": SCHEMA_VERSION,
        "run": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": str(source),
            "output": str(Path(args.output)) if args.output else "",
            "elapsed_seconds": elapsed,
            "processed_fps": processed_fps,
        },
        "video": {
            "frame_count_reported": total_frames,
            "frame_count_processed": frame_idx,
            "fps": fps,
            "width": width,
            "height": height,
        },
        "parameters": {
            "device": device,
            "detector": {
                "model": args.model,
                "conf": args.conf,
                "imgsz": args.imgsz,
                "classes": classes,
            },
            "tracker": {
                "track_activation_threshold": args.track_activation_threshold,
                "lost_track_buffer": args.lost_buffer,
                "minimum_matching_threshold": args.matching_threshold,
                "with_reid": args.with_reid,
                "with_gmc": not args.no_gmc,
            },
            "event": {
                "event_interval_frames": args.event_interval,
                "spatial_neighbor_threshold": args.spatial_threshold,
            },
        },
        "metrics": {
            "tracker": metrics.to_dict(),
            "entities": {
                "entity_count": len(profiles.to_entities()),
                "lifeline_count": len(profiles.to_lifelines()),
            },
            "events": {
                "event_count": len(events),
                "total_event_tracklets": sum(item["tracklet_count"] for item in events),
                "total_spatial_relations": sum(item["spatial_relation_count"] for item in events),
            },
            "kg": graph.stats,
        },
        "events": events,
        "entities": profiles.to_entities(),
        "lifelines": profiles.to_lifelines(),
    })


def summarize_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "frame_start": event.get("frame_start", 0),
        "frame_end": event.get("frame_end", 0),
        "duration": event.get("duration", []),
        "tracklet_count": len(event.get("tracklets", [])),
        "spatial_relation_count": len(event.get("spatial_relations", [])),
    }


def print_summary(result: dict[str, Any]) -> None:
    video = result["video"]
    run = result["run"]
    tracker = result["metrics"]["tracker"]
    entities = result["metrics"]["entities"]
    events = result["metrics"]["events"]
    kg = result["metrics"]["kg"]
    params = result["parameters"]

    print("\nBenchmark Summary")
    print_table([
        ("source", result["run"]["source"]),
        ("frames", f"{video['frame_count_processed']} / {video['frame_count_reported']}"),
        ("fps", video["fps"]),
        ("size", f"{video['width']}x{video['height']}"),
        ("elapsed_seconds", run["elapsed_seconds"]),
        ("processed_fps", run["processed_fps"]),
        ("device", params["device"]),
    ])

    print("\nParameters")
    print_table([
        ("model", params["detector"]["model"]),
        ("conf", params["detector"]["conf"]),
        ("imgsz", params["detector"]["imgsz"]),
        ("classes", params["detector"]["classes"]),
        ("lost_buffer", params["tracker"]["lost_track_buffer"]),
        ("matching_threshold", params["tracker"]["minimum_matching_threshold"]),
        ("with_reid", params["tracker"]["with_reid"]),
        ("with_gmc", params["tracker"]["with_gmc"]),
    ])

    print("\nMetrics")
    print_table([
        ("detection_total", tracker["detection_total"]),
        ("tracked_total", tracker["tracked_total"]),
        ("tracker_drop_rate", tracker["tracker_drop_rate"]),
        ("detection_gap_rate", tracker["detection_gap_rate"]),
        ("id_switch_count", tracker["id_switch_count"]),
        ("entity_count", entities["entity_count"]),
        ("event_count", events["event_count"]),
        ("interacts_with_edges", kg["edge_types"].get("interacts_with", 0)),
    ])


def print_table(rows: list[tuple[str, Any]]) -> None:
    key_width = max(len(str(key)) for key, _ in rows)
    val_width = max(len(format_value(value)) for _, value in rows)
    print(f"+-{'-' * key_width}-+-{'-' * val_width}-+")
    for key, value in rows:
        text = format_value(value)
        print(f"| {key:<{key_width}} | {text:<{val_width}} |")
    print(f"+-{'-' * key_width}-+-{'-' * val_width}-+")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ",".join(str(item) for item in value) if value else "all"
    if value is None:
        return "all"
    return str(value)


def default_output_path(source: str) -> str:
    stem = Path(source).stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("outputs") / f"{stem}_benchmark_{stamp}.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="视频检测/跟踪/Entity/KG 基线评估")
    parser.add_argument("--source", required=True, help="视频路径")
    parser.add_argument("--output", default="", help="JSON 输出路径")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 检测模型")
    parser.add_argument("--device", default="", help="推理设备，留空自动选择")
    parser.add_argument("--conf", type=float, default=0.15, help="检测置信度阈值")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO 推理尺寸")
    parser.add_argument("--classes", default="0", help="检测类别，逗号分隔；留空表示全部")
    parser.add_argument("--fps", type=float, default=25.0, help="视频无 FPS 元数据时使用")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数，0 表示不限")
    parser.add_argument("--event-interval", type=int, default=60, help="事件窗口帧数")
    parser.add_argument("--spatial-threshold", type=float, default=100.0, help="空间邻近阈值，像素")
    parser.add_argument("--track-activation-threshold", type=float, default=0.18)
    parser.add_argument("--lost-buffer", type=int, default=120)
    parser.add_argument("--matching-threshold", type=float, default=0.65)
    parser.add_argument("--with-reid", action="store_true", help="启用 tracker 内置 ReID")
    parser.add_argument("--no-gmc", action="store_true", help="关闭相机运动补偿")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.output:
        args.output = default_output_path(args.source)

    result = benchmark_video(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(result)
    print(f"\nJSON saved: {output}")


if __name__ == "__main__":
    main()
