"""
主管线编排。连接感知层 → Agent层 → KG层。

用法:
    python pipeline.py --source <video_path> [--config config.yaml]
    python pipeline.py --source 0              # 摄像头
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from perception.detector import YOLODetector, PoseEstimator
from perception.entity_profile import EntityProfileManager
from perception.metrics import TrackerMetrics
from perception.tracker import MultiObjectTracker
from perception.event_builder import EventBuilder
from agents.llm_client import LLMClient
from agents.trajectory_agent import TrajectoryAgent
from agents.interaction_agent import InteractionAgent
from agents.summary_agent import SummaryAgent
from agents.orchestrator import Orchestrator
from kg.continuity import write_continuity_event
from kg.graph_store import GraphStore

# COCO 17-point skeleton connections
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                   # torso
    (11, 13), (13, 15), (12, 14), (14, 16),       # legs
]

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
]


def strip_tracker_debug_fields(tracked: list[dict]) -> list[dict]:
    """Return tracklet copies without large per-frame debug payloads."""
    return [
        {key: value for key, value in tracklet.items() if key != "appearance_feature"}
        for tracklet in tracked
    ]


def resolve_runtime_fps(resolved_fps: float | None, video_fps: float) -> float:
    if resolved_fps and resolved_fps > 0:
        return resolved_fps
    if video_fps > 0:
        return video_fps
    return 30.0


def draw_annotations(frame: np.ndarray, tracked: list[dict]) -> np.ndarray:
    """在帧上画 bbox、骨架、track_id。

    Args:
        frame: BGR 图像
        tracked: tracker.update() 返回的结果列表

    Returns:
        标注后的帧
    """
    for t in tracked:
        color = COLORS[t.get("track_id", 0) % len(COLORS)]
        bgr = (color[2], color[1], color[0])  # RGB → BGR for OpenCV

        # bbox
        bbox = t.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

        # track_id label
        tid = t.get("track_id", "?")
        centroid = t.get("centroid", [0, 0])
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.putText(frame, f"#{tid}", (cx - 15, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

        # 骨架
        kps = t.get("pose_keypoints", [])
        if kps and len(kps) >= 17:
            for a, b in SKELETON:
                if a < len(kps) and b < len(kps):
                    if kps[a][2] > 0.3 and kps[b][2] > 0.3:
                        pt1 = (int(kps[a][0]), int(kps[a][1]))
                        pt2 = (int(kps[b][0]), int(kps[b][1]))
                        cv2.line(frame, pt1, pt2, bgr, 1)
            for kp in kps:
                if kp[2] > 0.3:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 2, bgr, -1)

    return frame


class Pipeline:
    def __init__(
        self,
        source: str,
        model_path: str = "yolov8n-pose.pt",
        device: str = "cpu",
        conf_threshold: float = 0.5,
        event_interval_frames: int = 30,
        deepseek_api_key: str = "",
        deepseek_base_url: str = "https://api.deepseek.com",
        deepseek_model: str = "deepseek-chat",
        kg_storage_path: str = "",
        output_log: str = "",
        ttl_seconds: float = 300.0,
        dry_run: bool = False,
        fps: float = 30.0,
        classes: list[int] | None = None,
        max_events: int = 0,
        imgsz: int = 640,
        annotate: bool = False,
        annotate_path: str = "annotated.mp4",
    ):
        self.source = source
        self.event_interval_frames = event_interval_frames
        self.ttl_seconds = ttl_seconds
        self.output_log = output_log
        self.model_path = model_path
        self.running = True
        self.dry_run = dry_run
        self.default_fps = fps
        self.max_events = max_events
        self.imgsz = imgsz
        self.annotate = annotate
        self.annotate_path = annotate_path
        self._writer = None  # cv2.VideoWriter, 延迟初始化
        self._last_keypoints: list[list[list[float]]] = []

        # 1. KG 层
        self.graphstore = GraphStore(use_directed=True)

        if not dry_run:
            # 2. LLM 客户端
            self.llm = LLMClient(
                api_key=deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=deepseek_base_url,
                model=deepseek_model,
            )

            # 3. 协调 Agent（注入真实 GraphStore）
            self.orchestrator = Orchestrator(llm_client=self.llm, kg_api=self.graphstore)

            # 4. 注册三个分析 Agent
            self.orchestrator.register_agent(TrajectoryAgent(self.llm))
            self.orchestrator.register_agent(InteractionAgent(self.llm))
            self.orchestrator.register_agent(SummaryAgent(self.llm))

        # 感知层（两阶段：检测 + 姿态）
        self.detector = YOLODetector(
            model_path=self.model_path, device=device, conf_threshold=conf_threshold,
            classes=classes, imgsz=imgsz,
        )
        self.pose_estimator = PoseEstimator(model_path="yolov8n-pose.pt", device=device)
        self.tracker = MultiObjectTracker(
            track_activation_threshold=0.18,
            lost_track_buffer=120,
            minimum_matching_threshold=0.65,
        )
        # spatial_neighbor_threshold 用像素值（无相机标定时 approximate 100px≈2-3m）
        self.event_builder = EventBuilder(
            event_interval_frames=event_interval_frames,
            spatial_neighbor_threshold_m=100.0,  # pixels, not meters until calibrated
        )
        self.entity_profiles = EntityProfileManager()
        self.tracker_metrics = TrackerMetrics()
        self._previous_event_id = ""

        # 状态
        self.frame_idx = 0
        self.kg_storage_path = kg_storage_path
        self._results: list[dict] = []

        # 注册信号处理
        signal.signal(signal.SIGINT, self._on_shutdown)
        signal.signal(signal.SIGTERM, self._on_shutdown)

    def _resolve_source(self) -> tuple[str, float]:
        """解析视频源。支持视频文件、摄像头ID、图片序列目录。

        Returns:
            (cv2_source_path, fps)
        """
        source = self.source

        # 摄像头
        if source.isdigit():
            return (int(source), 30.0)

        # 已经是 printf 格式模式
        if "%" in source:
            return (source, 30.0)

        # 目录：自动检测图片序列
        if os.path.isdir(source):
            jpegs = sorted(
                f for f in os.listdir(source)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if not jpegs:
                raise ValueError(f"目录 {source} 中没有图片文件")

            first = jpegs[0]
            # 推断命名模式：找数字部分并替换为 printf 通配符
            import re
            digits = re.search(r"(\d+)", first)
            if digits:
                width = len(digits.group())
                pattern = re.sub(r"\d+", f"%0{width}d", first)
            else:
                pattern = first

            return (os.path.join(source, pattern), self.default_fps)

        return (source, 0.0)

    def run(self):
        cap_source, resolved_fps = self._resolve_source()
        cap = cv2.VideoCapture(cap_source)
        # 图片序列用用户指定 fps，视频文件用实际 fps
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = resolve_runtime_fps(resolved_fps, video_fps)
        if not cap.isOpened():
            print(f"无法打开视频源: {self.source}")
            return

        print(f"Pipeline started. Source={self.source}, FPS={fps:.1f}, "
              f"EventInterval={self.event_interval_frames}frames, TTL={self.ttl_seconds}s")
        print("-" * 60)

        tracker_results: list[dict] = []
        event_start_frame = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = self.frame_idx / fps

            # 检测
            detections = self.detector.detect(frame)

            # 跟踪
            tracked = self.tracker.update(detections, self.frame_idx, timestamp, frame)
            tracked = self.entity_profiles.update(tracked, self.frame_idx, timestamp) if tracked else []
            self.tracker_metrics.observe(detections, tracked, self.entity_profiles.to_entities())
            if tracked:
                # 姿态估计：每5帧做一次，中间帧复用
                if self.frame_idx % 5 == 0:
                    bboxes = [t["bbox"] for t in tracked]
                    self._last_keypoints = self.pose_estimator.estimate(frame, bboxes)
                for i, kps in enumerate(self._last_keypoints):
                    if i < len(tracked):
                        tracked[i]["pose_keypoints"] = kps
                tracker_results.extend(strip_tracker_debug_fields(tracked))

            # 标注 + 写入视频
            if self.annotate:
                annotated = draw_annotations(frame.copy(), tracked if tracked else [])
                if self._writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self._writer = cv2.VideoWriter(self.annotate_path, fourcc, fps, (w, h))
                self._writer.write(annotated)

            self.frame_idx += 1

            # 每 N 帧生成一个事件
            if self.frame_idx - event_start_frame >= self.event_interval_frames:
                active_tracked = [
                    t for t in tracker_results
                    if t["track_id"] in self.tracker.get_active_tracks()
                ]
                if active_tracked:
                    event = self.event_builder.build_event(
                        tracked_objects=active_tracked,
                        frame_start=event_start_frame,
                        frame_end=self.frame_idx,
                        fps=fps,
                    )
                    self._process_event(event)

                tracker_results.clear()
                event_start_frame = self.frame_idx

        cap.release()
        self._on_shutdown()

    def _process_event(self, event: dict):
        if self.max_events and len(self._results) >= self.max_events:
            self.running = False
            return

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {event['event_id']} "
              f"duration={event['duration']} entities={len(event['tracklets'])} "
              f"relations={len(event['spatial_relations'])}")

        write_continuity_event(
            self.graphstore,
            event,
            self.entity_profiles.to_entities(),
            previous_event_id=self._previous_event_id,
            ttl_seconds=self.ttl_seconds,
        )
        self._previous_event_id = event["event_id"]

        if self.dry_run:
            print(f"  [DRY-RUN] tracklets: {len(event['tracklets'])} "
                  f"spatial_relations: {len(event['spatial_relations'])} "
                  f"zone={event['zone_id']}")
            for t in event["tracklets"]:
                print(f"    track_id={t['track_id']} class={t['class']} "
                      f"vel={t['velocity_ms']:.1f}m/s dir={t['direction_deg']:.0f}° "
                      f"trajectory_len={len(t['trajectory'])}")
            for r in event["spatial_relations"]:
                print(f"    relation: {r['track_1']}↔{r['track_2']} "
                      f"dist={r['distance_m']:.1f}m type={r['relation_type']}")
            self._results.append({
                "event_id": event["event_id"],
                "duration": event["duration"],
                "tracklet_count": len(event["tracklets"]),
                "relation_count": len(event["spatial_relations"]),
                "entities": self.entity_profiles.to_entities(),
                "lifelines": self.entity_profiles.to_lifelines(),
                "tracker_metrics": self.tracker_metrics.to_dict(),
            })
            return

        output = self.orchestrator.process_event(event)

        # 打印推理链
        trail = output.get("reasoning_trail", [])
        print(f"  [推理链]")
        for step in trail:
            print(f"    ├─ {step['agent']}")
            print(f"    │  输入: {step['input_summary']}")
            for line in step['output_summary'].split('\n'):
                print(f"    │  {line}")

        # 打印场景报告
        report = output.get("scene_report", {})
        print(f"  [场景报告] entities={report.get('entity_count', 0)}")
        summary = report.get('summary', '')
        for line in summary.split('\n')[:8]:
            if line.strip():
                print(f"    {line.strip()[:150]}")
        print(f"  [置信度] TA={output['agent_summary']['trajectory_agent_confidence']:.2f} "
              f"IA={output['agent_summary']['interaction_agent_confidence']:.2f} "
              f"SA={output['agent_summary']['summary_agent_confidence']:.2f}")

        # KG 清理
        now = time.time()
        evicted = self.graphstore.evict_before(now - self.ttl_seconds)
        if evicted:
            print(f"  [KG] evicted {evicted} expired events")

        self._results.append({
            "event_id": event["event_id"],
            "duration": event["duration"],
            "frame_range": [event.get("frame_start", 0), event.get("frame_end", 0)],
            "output": output,
            "entities": self.entity_profiles.to_entities(),
            "lifelines": self.entity_profiles.to_lifelines(),
            "tracker_metrics": self.tracker_metrics.to_dict(),
        })

    def _on_shutdown(self, signum=None, frame=None):
        if getattr(self, '_shutting_down', False):
            return
        self._shutting_down = True
        was_running = self.running
        self.running = False
        print(f"\n{'='*60}")
        print(f"Shutting down. Processed {len(self._results)} events.")

        # 释放 VideoWriter
        if self._writer is not None:
            self._writer.release()
            print(f"Annotated video saved to {self.annotate_path}")

        # 持久化 KG
        if self.kg_storage_path:
            self.graphstore.save(self.kg_storage_path)
            print(f"KG saved to {self.kg_storage_path}")

        # 写输出日志
        if self.output_log:
            with open(self.output_log, "w", encoding="utf-8") as f:
                json.dump(self._results, f, ensure_ascii=False, indent=2)
            print(f"Output log saved to {self.output_log}")

        stats = self.graphstore.stats
        print(f"KG stats: nodes={stats['node_count']} edges={stats['edge_count']} "
              f"types={stats['node_types']}")
        metrics = self.tracker_metrics.to_dict()
        print("Tracker metrics: "
              f"drop_rate={metrics['tracker_drop_rate']:.4f} "
              f"gap_rate={metrics['detection_gap_rate']:.4f} "
              f"id_switches={metrics['id_switch_count']} "
              f"raw_tracks={metrics['raw_track_count']} entities={metrics['entity_count']}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="YOLO + Multi-Agent + KG Pipeline")
    parser.add_argument("--source", required=True, help="视频路径或摄像头ID")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO 模型路径")
    parser.add_argument("--device", default="cpu", help="推理设备 cpu/cuda")
    parser.add_argument("--conf", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--event-interval", type=int, default=30,
                        help="事件生成间隔（帧数）")
    parser.add_argument("--api-key", default="", help="DeepSeek API Key")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--api-model", default="deepseek-chat")
    parser.add_argument("--kg-path", default="kg_output.graphml",
                        help="KG 持久化路径")
    parser.add_argument("--output", default="results.json",
                        help="输出日志路径")
    parser.add_argument("--ttl", type=float, default=300.0,
                        help="Event 节点 TTL（秒）")
    parser.add_argument("--classes", type=str, default="",
                        help="检测类别，逗号分隔 (如 '0'=person, '0,2'=person+car)")
    parser.add_argument("--dry-run", action="store_true",
                        help="跳过 Agent 推理，仅验证感知层输出")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="图片序列模式的帧率")
    parser.add_argument("--max-events", type=int, default=0,
                        help="最多处理的事件数（0=不限）")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO 推理分辨率 (默认 640)")
    parser.add_argument("--annotate", action="store_true",
                        help="输出骨架标注视频")
    parser.add_argument("--annotate-path", default="annotated.mp4",
                        help="标注视频输出路径")
    args = parser.parse_args()

    classes = [int(c.strip()) for c in args.classes.split(",") if c.strip()] if args.classes else None

    pipeline = Pipeline(
        source=args.source,
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        event_interval_frames=args.event_interval,
        deepseek_api_key=args.api_key,
        deepseek_base_url=args.api_base,
        deepseek_model=args.api_model,
        kg_storage_path=args.kg_path,
        output_log=args.output,
        ttl_seconds=args.ttl,
        dry_run=args.dry_run,
        fps=args.fps,
        classes=classes,
        max_events=args.max_events,
        imgsz=args.imgsz,
        annotate=args.annotate,
        annotate_path=args.annotate_path,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
