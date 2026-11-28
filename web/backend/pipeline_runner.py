"""后台 pipeline 执行器。实时推 base64 帧 + Agent 状态 + 事件。"""

import asyncio
import base64
import contextlib
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


def create_web_tracker(device: str) -> MultiObjectTracker:
    return MultiObjectTracker(
        track_activation_threshold=0.18,
        lost_track_buffer=120,
        minimum_matching_threshold=0.65,
        reid_device=device,
        with_gmc=False,
        reid_interval=2,
    )


def build_pose_targets(
    tracked_objects: list[dict],
    candidate_objects: list[dict],
    frame_idx: int,
    max_candidates: int = 8,
) -> list[dict]:
    targets: list[dict] = []
    for tracklet in tracked_objects:
        bbox = tracklet.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        targets.append({
            "frame_idx": frame_idx,
            "target_id": str(tracklet.get("entity_id", f"track-{tracklet.get('track_id', '')}")),
            "track_id": tracklet.get("track_id"),
            "entity_id": tracklet.get("entity_id", f"Entity-{tracklet.get('track_id', '')}"),
            "bbox": list(bbox),
            "candidate": False,
            "tentative_entity": False,
        })
    tentative_candidates = [
        candidate for candidate in candidate_objects
        if candidate.get("tentative_entity") and candidate.get("bbox")
    ][:max_candidates]
    for candidate in tentative_candidates:
        targets.append({
            "frame_idx": frame_idx,
            "target_id": str(candidate.get("entity_id", candidate.get("track_id", ""))),
            "track_id": candidate.get("track_id"),
            "entity_id": candidate.get("entity_id", "Tentative"),
            "bbox": list(candidate["bbox"]),
            "candidate": True,
            "tentative_entity": True,
        })
    return targets


def replace_latest_pose_job(queue: asyncio.Queue, job: dict | None) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    try:
        queue.put_nowait(job)
    except asyncio.QueueFull:
        pass


async def pose_worker(
    pose_estimator: PoseEstimator,
    request_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
) -> None:
    while True:
        job = await request_queue.get()
        if job is None:
            return
        targets = job.get("targets", [])
        if not targets:
            continue
        frame = job["frame"]
        keypoints_list = await asyncio.to_thread(
            pose_estimator.estimate,
            frame,
            [target["bbox"] for target in targets],
        )
        poses = []
        for target, keypoints in zip(targets, keypoints_list):
            if not keypoints:
                continue
            poses.append({
                "target_id": target["target_id"],
                "track_id": target.get("track_id"),
                "entity_id": target["entity_id"],
                "bbox": target["bbox"],
                "candidate": target.get("candidate", False),
                "tentative_entity": target.get("tentative_entity", False),
                "keypoints": keypoints,
            })
        if poses:
            replace_latest_pose_job(result_queue, {
                "type": "pose_update",
                "frame_idx": job["frame_idx"],
                "poses": poses,
            })


async def drain_pose_updates(
    result_queue: asyncio.Queue,
    websocket,
) -> None:
    while True:
        try:
            message = result_queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        await websocket.send_json(message)


def write_annotated_frame(path: str | Path, frame) -> None:
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 72])


async def frame_writer_worker(queue: asyncio.Queue) -> None:
    while True:
        job = await queue.get()
        if job is None:
            return
        path, frame = job
        await asyncio.to_thread(write_annotated_frame, path, frame)


def _warmup_models(cap, detector: YOLODetector, pose_estimator: PoseEstimator) -> None:
    """Use the first video frame for model warmup, then rewind the capture."""
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    if not ret or frame is None:
        return
    detections = detector.detect(frame)
    if detections:
        pose_estimator.estimate(frame, [detections[0]["bbox"]])


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def find_unmatched_detections(
    detections: list[dict],
    tracked_objects: list[dict],
    iou_threshold: float = 0.35,
) -> list[dict]:
    tracked_boxes = [t["bbox"] for t in tracked_objects if t.get("bbox") and len(t["bbox"]) == 4]
    unmatched = []
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        if not any(_box_iou(bbox, tracked_box) >= iou_threshold for tracked_box in tracked_boxes):
            unmatched.append(det)
    return unmatched


class CandidateDisplayTracker:
    """Display-only continuity for detections that are not stable tracker outputs."""

    def __init__(
        self,
        min_hits: int = 3,
        max_age: int = 5,
        iou_threshold: float = 0.35,
    ) -> None:
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self._candidates: dict[str, dict] = {}
        self._next_id = 1

    def update(self, detections: list[dict], frame_idx: int) -> list[dict]:
        matched_ids: set[str] = set()
        visible: list[dict] = []
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            candidate_id = self._match_candidate(bbox, matched_ids)
            if candidate_id is None:
                candidate_id = f"cand-{self._next_id}"
                self._next_id += 1
                self._candidates[candidate_id] = {
                    "candidate_id": candidate_id,
                    "hits": 0,
                    "first_seen_frame": frame_idx,
                    "pose_keypoints": [],
                }
            cand = self._candidates[candidate_id]
            cand["bbox"] = list(bbox)
            cand["confidence"] = float(det.get("confidence", 0.0))
            cand["last_seen_frame"] = frame_idx
            cand["hits"] = int(cand.get("hits", 0)) + 1
            matched_ids.add(candidate_id)
            visible.append(self._to_display(cand))

        for candidate_id, cand in list(self._candidates.items()):
            if candidate_id in matched_ids:
                continue
            if frame_idx - int(cand.get("last_seen_frame", frame_idx)) > self.max_age:
                del self._candidates[candidate_id]

        return visible

    def update_pose(self, candidate_id: str, keypoints: list[list[float]]) -> None:
        if candidate_id in self._candidates:
            self._candidates[candidate_id]["pose_keypoints"] = keypoints

    def _match_candidate(self, bbox: list[float], matched_ids: set[str]) -> str | None:
        best_id = None
        best_iou = 0.0
        for candidate_id, cand in self._candidates.items():
            if candidate_id in matched_ids:
                continue
            candidate_bbox = cand.get("bbox")
            if not candidate_bbox:
                continue
            iou = _box_iou(bbox, candidate_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = candidate_id
        return best_id if best_iou >= self.iou_threshold else None

    def _to_display(self, cand: dict) -> dict:
        hits = int(cand.get("hits", 0))
        promoted = hits >= self.min_hits
        bbox = cand["bbox"]
        entity_id = f"Tentative-{cand['candidate_id'].split('-')[-1]}" if promoted else "候选检测"
        return {
            "track_id": cand["candidate_id"],
            "bbox": list(bbox),
            "entity_id": entity_id,
            "candidate": True,
            "tentative_entity": promoted,
            "confidence": float(cand.get("confidence", 0.0)),
            "candidate_hits": hits,
            "first_seen_frame": int(cand.get("first_seen_frame", 0)),
            "last_seen_frame": int(cand.get("last_seen_frame", 0)),
            "pose_keypoints": cand.get("pose_keypoints", []),
            "centroid": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
        }


def smooth_display_tracks(
    tracked_objects: list[dict],
    memory: dict[int, dict],
    frame_idx: int,
    max_age: int = 5,
) -> list[dict]:
    visible: list[dict] = []
    current_ids: set[int] = set()
    for tracklet in tracked_objects:
        tid = int(tracklet.get("track_id", 0))
        current_ids.add(tid)
        current = dict(tracklet)
        current.setdefault("track_state", "confirmed")
        current["_display_last_seen"] = frame_idx
        memory[tid] = current
        visible.append({key: value for key, value in current.items() if not key.startswith("_display_")})

    for tid, previous in list(memory.items()):
        age = frame_idx - int(previous.get("_display_last_seen", frame_idx))
        if tid in current_ids:
            continue
        if age <= max_age:
            ghost = {key: value for key, value in previous.items() if not key.startswith("_display_")}
            ghost["track_state"] = "display_gap"
            ghost["display_age"] = age
            visible.append(ghost)
        else:
            del memory[tid]
    return visible


def make_annotated_frame(frame, tracked_objects, detections=None, candidate_objects=None):
    """返回 (annotated_frame, person_list)。person_list 用于前端的点击检测。"""
    annotated = frame.copy()
    persons = []
    raw_candidates = candidate_objects
    if raw_candidates is None:
        raw_candidates = []
        for idx, det in enumerate(find_unmatched_detections(detections or [], tracked_objects)):
            x1, y1, x2, y2 = map(int, det["bbox"])
            raw_candidates.append({
                "track_id": f"det-{idx}",
                "bbox": [x1, y1, x2, y2],
                "entity_id": "候选检测",
                "candidate": True,
                "confidence": float(det.get("confidence", 0)),
                "centroid": [(x1 + x2) / 2, (y1 + y2) / 2],
            })
    for cand in raw_candidates:
        x1, y1, x2, y2 = map(int, cand["bbox"])
        promoted = bool(cand.get("tentative_entity"))
        color = (255, 210, 80) if not promoted else (80, 230, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = str(cand.get("entity_id") or "det")
        if label == "候选检测":
            label = f"det {cand.get('confidence', 0):.2f}"
        cv2.putText(annotated, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        persons.append({
            "track_id": cand.get("track_id", "det"),
            "bbox": [x1, y1, x2, y2],
            "entity_id": cand.get("entity_id", "候选检测"),
            "candidate": True,
            "tentative_entity": bool(cand.get("tentative_entity", False)),
            "confidence": float(cand.get("confidence", 0)),
            "candidate_hits": int(cand.get("candidate_hits", 1)),
            "color": [80, 210, 255],
            "centroid": cand.get("centroid", [(x1 + x2) / 2, (y1 + y2) / 2]),
            "pose_keypoints": cand.get("pose_keypoints", []),
        })
        kps = cand.get("pose_keypoints", [])
        if kps and len(kps) >= 17:
            for a, b in SKELETON:
                if a < len(kps) and b < len(kps) and kps[a][2] > 0.3 and kps[b][2] > 0.3:
                    cv2.line(annotated, (int(kps[a][0]), int(kps[a][1])),
                             (int(kps[b][0]), int(kps[b][1])), color, 1)
            for kp in kps:
                if kp[2] > 0.3:
                    cv2.circle(annotated, (int(kp[0]), int(kp[1])), 2, color, -1)
    for t in tracked_objects:
        tid = t.get("track_id", 0)
        color = COLORS[tid % len(COLORS)]
        bgr = (color[2], color[1], color[0])
        is_gap = t.get("track_state") == "display_gap"
        bbox = t.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 1 if is_gap else 2)
            persons.append({"track_id": tid, "bbox": [x1, y1, x2, y2],
                            "entity_id": t.get("entity_id", f"Entity-{tid}"),
                            "color": list(color), "centroid": t.get("centroid", [0, 0]),
                            "display_gap": is_gap,
                            "tentative_predecessor": t.get("tentative_predecessor", "")})
        centroid = t.get("centroid", [0, 0])
        label = t.get("entity_id", f"#{tid}")
        if is_gap:
            label = f"{label} 预测"
        cv2.putText(annotated, label, (int(centroid[0]) - 24, int(centroid[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1 if is_gap else 2)
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

    await websocket.send_json({"type": "agent_status", "agent": "system", "status": "loading", "message": f"加载姿态模型... 设备 {device}"})
    pose_est = PoseEstimator("yolov8n-pose.pt", device)
    await websocket.send_json({"type": "agent_status", "agent": "system", "status": "loading", "message": "预热检测模型..."})
    _warmup_models(cap, detector, pose_est)
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

    tracker = create_web_tracker(device)
    # 事件编号是类变量，同一进程里第二次上传不重置就会从上一个视频的编号往后数
    EventBuilder.reset_ids()
    event_builder = EventBuilder(event_interval_frames=60)
    entity_profiles = EntityProfileManager()
    # 位置门限按画面宽度比例（跨遮挡合并需要，140px 默认值在大画面下过严）
    _frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    entity_profiles.max_position_gap_px = max(entity_profiles.max_position_gap_px, 0.4 * _frame_width)
    tracker_metrics = TrackerMetrics()
    previous_event_id = ""

    frame_idx = 0
    event_start = 0
    tracker_results: list[dict] = []
    all_events: list[dict] = []
    annotated_images: list[str] = []
    display_track_memory: dict[int, dict] = {}
    candidate_tracker = CandidateDisplayTracker(min_hits=3, max_age=5, iou_threshold=0.35)
    pose_request_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    pose_result_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    pose_task = asyncio.create_task(pose_worker(pose_est, pose_request_queue, pose_result_queue))
    frame_write_queue: asyncio.Queue = asyncio.Queue(maxsize=16)
    frame_writer_task = asyncio.create_task(frame_writer_worker(frame_write_queue))

    try:
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
                tracker_results.extend(strip_tracker_debug_fields(tracked))

            # 实时帧推送（每3帧），但磁盘保存每帧
            safe_tracked = strip_tracker_debug_fields(tracked) if tracked else []
            display_tracked = smooth_display_tracks(safe_tracked, display_track_memory, frame_idx)
            unmatched_detections = find_unmatched_detections(detections, display_tracked)
            candidate_objects = candidate_tracker.update(unmatched_detections, frame_idx)
            pose_targets = build_pose_targets(display_tracked, candidate_objects, frame_idx)
            if pose_targets:
                replace_latest_pose_job(
                    pose_request_queue,
                    {"frame_idx": frame_idx, "frame": frame.copy(), "targets": pose_targets},
                )
            annotated, persons = make_annotated_frame(
                frame,
                display_tracked,
                candidate_objects=candidate_objects,
            )
            frame_path = frame_dir / f"{frame_idx:06d}.jpg"
            await frame_write_queue.put((frame_path, annotated))
            annotated_images.append(str(frame_path))

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
            await drain_pose_updates(pose_result_queue, websocket)

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
                        # Agent 链路是同步 HTTP 调用，放线程里跑，别阻塞事件循环
                        output = await asyncio.to_thread(orch.process_event, event)
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
    finally:
        replace_latest_pose_job(pose_request_queue, None)
        with contextlib.suppress(Exception):
            await pose_task
        await drain_pose_updates(pose_result_queue, websocket)
        await frame_write_queue.put(None)
        await frame_writer_task

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
