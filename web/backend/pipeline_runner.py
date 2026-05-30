"""后台 pipeline 执行器。实时推 base64 帧 + Agent 状态 + 事件。"""

import asyncio
import base64
import json
import os
import time
from pathlib import Path

import cv2

from perception.detector import YOLODetector, PoseEstimator
from perception.entity_profile import EntityProfileManager
from perception.metrics import TrackerMetrics
from perception.tracker import MultiObjectTracker
from perception.event_builder import EventBuilder
from agents.llm_client import LLMClient
from agents.env import get_local_env
from agents.trajectory_agent import TrajectoryAgent
from agents.interaction_agent import InteractionAgent
from agents.summary_agent import SummaryAgent
from agents.orchestrator import Orchestrator
from kg.continuity import write_continuity_event
from kg.graph_store import GraphStore

SKELETON = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
COLORS = [(0,255,0),(255,80,80),(80,80,255),(255,200,40),(200,40,255),(40,220,220),
          (140,255,80),(255,140,80),(80,160,255),(200,80,200),(220,220,80),(80,255,200)]

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

AGENT_NAMES = {"trajectory_agent": "轨迹分析", "interaction_agent": "交互分析", "summary_agent": "场景总结"}


def select_inference_device() -> str:
    configured = os.getenv("GP_INFERENCE_DEVICE", "").strip()
    if configured:
        return configured
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def strip_tracker_debug_fields(tracked: list[dict]) -> list[dict]:
    """Return tracklet copies without large per-frame debug payloads."""
    return [
        {key: value for key, value in tracklet.items() if key != "appearance_feature"}
        for tracklet in tracked
    ]


def make_annotated_frame(frame, tracked_objects):
    """返回 (annotated_frame, person_list)。person_list 用于前端的点击检测。"""
    annotated = frame.copy()
    persons = []
    for t in tracked_objects:
        tid = t.get("track_id", 0)
        color = COLORS[tid % len(COLORS)]
        bgr = (color[2], color[1], color[0])
        bbox = t.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
            persons.append({"track_id": tid, "bbox": [x1, y1, x2, y2],
                            "entity_id": t.get("entity_id", f"Entity-{tid}"),
                            "color": list(color), "centroid": t.get("centroid", [0, 0])})
        centroid = t.get("centroid", [0, 0])
        label = t.get("entity_id", f"#{tid}")
        cv2.putText(annotated, label, (int(centroid[0]) - 24, int(centroid[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
        kps = t.get("pose_keypoints", [])
        if kps and len(kps) >= 17:
            for a, b in SKELETON:
                if a < len(kps) and b < len(kps) and kps[a][2] > 0.3 and kps[b][2] > 0.3:
                    cv2.line(annotated, (int(kps[a][0]), int(kps[a][1])),
                             (int(kps[b][0]), int(kps[b][1])), bgr, 1)
            for kp in kps:
                if kp[2] > 0.3:
                    cv2.circle(annotated, (int(kp[0]), int(kp[1])), 2, bgr, -1)
    return annotated, persons


def _build_mp4(out_path: str, frame_dir, fps: float, total: int):
    """从保存的 JPEG 帧合成 MP4。"""
    if total == 0:
        return
    first = cv2.imread(str(frame_dir / "000000.jpg"))
    if first is None:
        # try 000001
        first = cv2.imread(str(frame_dir / "000001.jpg"))
    if first is None:
        return
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for i in range(total):
        fpath = frame_dir / f"{i:06d}.jpg"
        if fpath.exists():
            writer.write(cv2.imread(str(fpath)))
    writer.release()


async def run_pipeline(video_path: str, video_id: str, websocket, api_key: str = ""):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "message": "无法打开视频"})
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_dir = OUTPUT_DIR / video_id
    frame_dir.mkdir(exist_ok=True)

    await websocket.send_json({"type": "meta", "fps": fps, "total_frames": total_frames})
    await asyncio.sleep(0)

    # 初始化模型
    await websocket.send_json({"type": "agent_status", "agent": "system", "status": "loading", "message": "加载检测模型..."})
    device = select_inference_device()
    detector = YOLODetector("yolov8l.pt", device, 0.15, classes=[0], imgsz=960)
    await asyncio.sleep(0)

    await websocket.send_json({"type": "agent_status", "agent": "system", "status": "loading", "message": "加载姿态模型..."})
    pose_est = PoseEstimator("yolov8n-pose.pt", device)
    await asyncio.sleep(0)

    graphstore = GraphStore(use_directed=True)
    kit = api_key or get_local_env("DEEPSEEK_API_KEY")
    llm = LLMClient(api_key=kit) if kit else None

    orch = None
    if llm:
        orch = Orchestrator(llm_client=llm, kg_api=graphstore)
        orch.register_agent(TrajectoryAgent(llm))
        orch.register_agent(InteractionAgent(llm))
        orch.register_agent(SummaryAgent(llm))

    tracker = MultiObjectTracker(
        track_activation_threshold=0.18,
        lost_track_buffer=120,
        minimum_matching_threshold=0.65,
        reid_device=device,
    )
    event_builder = EventBuilder(event_interval_frames=60)
    entity_profiles = EntityProfileManager()
    tracker_metrics = TrackerMetrics()
    previous_event_id = ""

    frame_idx = 0
    event_start = 0
    tracker_results: list[dict] = []
    all_events: list[dict] = []
    last_keypoints: list[list[list[float]]] = []
    annotated_images: list[str] = []

    await websocket.send_json({"type": "agent_status", "agent": "system", "status": "ready", "message": "开始分析"})
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        detections = detector.detect(frame)
        tracked = tracker.update(detections, frame_idx, timestamp, frame)
        tracked = entity_profiles.update(tracked, frame_idx, timestamp) if tracked else []
        tracker_metrics.observe(detections, tracked, entity_profiles.to_entities())

        if tracked:
            if frame_idx % 3 == 0:
                bboxes = [t["bbox"] for t in tracked]
                last_keypoints = pose_est.estimate(frame, bboxes)
            for i, kps in enumerate(last_keypoints):
                if i < len(tracked):
                    tracked[i]["pose_keypoints"] = kps
            tracker_results.extend(strip_tracker_debug_fields(tracked))

        # 实时帧推送（每3帧），但磁盘保存每帧
        safe_tracked = strip_tracker_debug_fields(tracked) if tracked else []
        annotated, persons = make_annotated_frame(frame, safe_tracked)
        cv2.imwrite(str(frame_dir / f"{frame_idx:06d}.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
        annotated_images.append(str(frame_dir / f"{frame_idx:06d}.jpg"))

        if frame_idx % 3 == 0:
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 55])
            b64 = base64.b64encode(buf).decode()
            await websocket.send_json({
                "type": "frame",
                "frame_idx": frame_idx,
                "image": b64,
                "persons": persons,
                "entities": entity_profiles.to_entities(),
            })

        frame_idx += 1
        if frame_idx % 5 == 0:
            await asyncio.sleep(0)

        # 进度
        if frame_idx % 15 == 0:
            elapsed = time.time() - t_start
            remaining = elapsed / frame_idx * (total_frames - frame_idx) if frame_idx > 0 else 0
            await websocket.send_json({
                "type": "progress",
                "frame": frame_idx, "total": total_frames,
                "pct": int(frame_idx / total_frames * 100),
                "elapsed": int(elapsed), "remaining": int(remaining),
            })

        # 事件
        if frame_idx - event_start >= 60:
            active = [t for t in tracker_results if t["track_id"] in tracker.get_active_tracks()]
            if active and orch and llm:
                await websocket.send_json({"type": "agent_status", "agent": "trajectory_agent", "status": "active"})
                try:
                    event = event_builder.build_event(active, event_start, frame_idx, fps)
                    output = orch.process_event(event)
                    write_continuity_event(
                        graphstore,
                        event,
                        entity_profiles.to_entities(),
                        previous_event_id=previous_event_id,
                    )
                    previous_event_id = event["event_id"]
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Agent处理失败: {e}"})
                    tracker_results.clear()
                    event_start = frame_idx
                    continue
                await websocket.send_json({"type": "agent_status", "agent": "trajectory_agent", "status": "done"})

                report = output.get("scene_report", {})
                trail = output.get("reasoning_trail", [])

                # Agent 状态切换动画
                for t in trail:
                    name = t.get("agent", "")
                    await websocket.send_json({
                        "type": "agent_status", "agent": name,
                        "status": "done" if name != trail[-1].get("agent") else "active",
                    })

                await websocket.send_json({"type": "agent_status", "agent": "summary_agent", "status": "done"})

                motion_list = []
                for m in report.get("motion_descriptions", []):
                    motion_list.append({"entity_id": m.get("entity_id", ""), "data": m.get("description", {})})
                inter_list = [{"entities": i.get("entities", []), "desc": i.get("description", "")}
                              for i in report.get("interaction_descriptions", [])]

                event_data = {
                    "type": "event",
                    "event_id": event["event_id"],
                    "frame_start": event["frame_start"], "frame_end": event["frame_end"],
                    "duration": event["duration"],
                    "entity_count": len(event["tracklets"]),
                    "summary": report.get("summary", ""),
                    "motions": motion_list,
                    "interactions": inter_list,
                    "trail": [{"agent": t["agent"], "input": t["input_summary"], "output": t["output_summary"]}
                              for t in trail],
                    "entities": entity_profiles.to_entities(),
                    "lifelines": entity_profiles.to_lifelines(),
                    "kg_subgraph": graphstore.to_dict(),
                    "tracker_metrics": tracker_metrics.to_dict(),
                }
                all_events.append(event_data)
                await websocket.send_json(event_data)
            elif active:
                event = event_builder.build_event(active, event_start, frame_idx, fps)
                write_continuity_event(
                    graphstore,
                    event,
                    entity_profiles.to_entities(),
                    previous_event_id=previous_event_id,
                )
                previous_event_id = event["event_id"]
                ev = {"type": "event", "event_id": event["event_id"],
                      "frame_start": event["frame_start"], "frame_end": event["frame_end"],
                      "duration": event["duration"], "entity_count": len(event["tracklets"]),
                      "summary": f"检测到 {len(event['tracklets'])} 个目标",
                      "motions": [], "interactions": [], "trail": [],
                      "entities": entity_profiles.to_entities(),
                      "lifelines": entity_profiles.to_lifelines(),
                      "kg_subgraph": graphstore.to_dict(),
                      "tracker_metrics": tracker_metrics.to_dict()}
                all_events.append(ev)
                await websocket.send_json(ev)

            tracker_results.clear()
            event_start = frame_idx

    cap.release()

    # 合成标注 MP4
    await websocket.send_json({"type": "agent_status", "agent": "system", "status": "loading", "message": "合成 MP4..."})
    mp4_path = OUTPUT_DIR / f"{video_id}_annotated.mp4"
    _build_mp4(str(mp4_path), frame_dir, fps, frame_idx)

    # 保存 KG
    kg_path = str(Path(__file__).resolve().parent.parent.parent / "kg_outputs" / f"{video_id}.graphml")
    graphstore.save(kg_path)

    await websocket.send_json({
        "type": "done",
        "total_frames": frame_idx,
        "fps": fps,
        "mp4_url": f"/outputs/{video_id}_annotated.mp4",
        "frame_dir": f"/outputs/{video_id}/",
        "events": all_events,
        "tracker_metrics": tracker_metrics.to_dict(),
    })
