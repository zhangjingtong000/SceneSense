"""流水线版预测导出：读帧、检测、跟踪三段并行。

与原版 export_predictions.py 的区别：
- 读帧线程与检测线程独立，解码不再阻塞检测
- 检测线程批量推理，并可在主线程处理上一帧时提前处理后续帧
- 主线程只负责跟踪、实体融合与写文件

用法::

    python scripts/export_predictions_pipelined.py --source <video> --out <jsonl>
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.entity_profile import EntityProfileManager  # noqa: E402
from perception.tracker import MultiObjectTracker  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="流水线预测导出")
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="yolov8l.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--reid-interval", type=int, default=3)
    parser.add_argument("--match-thresh", type=float, default=0.65)
    parser.add_argument("--proximity-thresh", type=float, default=0.5)
    parser.add_argument("--appearance-thresh", type=float, default=0.25)
    parser.add_argument("--lost-low", action="store_true",
                        help="实验项：让 lost 轨迹参与低分框匹配（多场景验证为负收益，默认关闭）")
    parser.add_argument("--match-margin", type=float, default=0.0,
                        help="匹配区分度门限（0 关闭；实验项，用相对排序替代绝对阈值）")
    parser.add_argument("--score-norm", action="store_true",
                        help="实验项：实体合并用候选分数 z-norm 相对排序替代绝对阈值")
    parser.add_argument("--correct-bbox", action="store_true",
                        help="实验项：被遮挡截断的框用历史中位高度补全")
    parser.add_argument("--median-feature", action="store_true",
                        help="实验项：外观相似度用近 15 帧中位数替代 EMA")
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--out", required=True)
    return parser


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def main() -> None:
    args = build_parser().parse_args()
    from perception.detector import YOLODetector

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"视频不存在: {src}")
    device = resolve_device(args.device)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    total = args.max_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = YOLODetector(
        model_path=args.model, device=device, conf_threshold=args.conf,
        classes=[0], imgsz=args.imgsz,
    )
    tracker = MultiObjectTracker(
        track_activation_threshold=0.18, lost_track_buffer=120,
        minimum_matching_threshold=args.match_thresh,
        reid_device=device if device.startswith("cuda") else "cpu",
        reid_interval=max(1, args.reid_interval),
        include_lost_in_low_stage=args.lost_low,
    )
    tracker.proximity_thresh = args.proximity_thresh
    tracker.appearance_thresh = args.appearance_thresh
    tracker.match_margin = args.match_margin
    tracker.use_median_feature = args.median_feature
    tracker.correct_truncated_bbox = args.correct_bbox
    profiles = EntityProfileManager()
    profiles.max_position_gap_px = max(profiles.max_position_gap_px, 0.4 * width)
    profiles.use_score_norm = args.score_norm

    frames_q: queue.Queue = queue.Queue(maxsize=args.queue_size)
    dets_q: queue.Queue = queue.Queue(maxsize=args.queue_size)
    stop = threading.Event()
    errors: list[BaseException] = []
    # ReID 独立线程：检测一完成就提交，主线程消费时通常已算好
    reid_executor = ThreadPoolExecutor(max_workers=1)
    feat_futures: dict[int, object] = {}

    def reader() -> None:
        idx = 0
        try:
            while idx < total and not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                frames_q.put((idx, frame))
                idx += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            frames_q.put(None)

    def detector_worker() -> None:
        batch: list[tuple[int, "object"]] = []
        try:
            while not stop.is_set():
                item = frames_q.get()
                if item is None:
                    break
                batch.append(item)
                if len(batch) >= args.batch_size or frames_q.empty():
                    frames = [f for _, f in batch]
                    if len(frames) > 1:
                        results = detector.detect_batch(frames)
                    else:
                        results = [detector.detect(frames[0])]
                    for (idx, frame), dets in zip(batch, results):
                        # 立即提交 ReID 任务，与后续检测重叠
                        if tracker.reid is not None and idx % args.reid_interval == 0 and len(dets):
                            boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
                            feat_futures[idx] = reid_executor.submit(
                                tracker.reid.extract, frame, boxes
                            )
                        dets_q.put((idx, frame, dets))
                    batch = []
            if batch:
                frames = [f for _, f in batch]
                results = detector.detect_batch(frames) if len(frames) > 1 else [detector.detect(frames[0])]
                for (idx, frame), dets in zip(batch, results):
                    if tracker.reid is not None and idx % args.reid_interval == 0 and len(dets):
                        boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
                        feat_futures[idx] = reid_executor.submit(
                            tracker.reid.extract, frame, boxes
                        )
                    dets_q.put((idx, frame, dets))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            dets_q.put(None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader_thread = threading.Thread(target=reader, daemon=True)
    detector_thread = threading.Thread(target=detector_worker, daemon=True)
    started = time.perf_counter()
    first_frame_ms = None
    written = 0
    processed = 0

    reader_thread.start()
    detector_thread.start()

    try:
        with out_path.open("w", encoding="utf-8") as handle:
            while True:
                item = dets_q.get()
                if item is None:
                    break
                idx, frame, detections = item
                frame_started = time.perf_counter()
                future = feat_futures.pop(idx, None)
                cached_features = future.result() if future is not None else None
                tracked = tracker.update(
                    detections, idx, idx / fps, frame, features=cached_features
                )
                enriched = profiles.update(tracked, idx, idx / fps) if tracked else []
                if first_frame_ms is None:
                    first_frame_ms = round((time.perf_counter() - frame_started) * 1000, 1)

                for obj in enriched:
                    if obj.get("class") != "person":
                        continue
                    bbox = obj.get("bbox")
                    if not bbox:
                        continue
                    handle.write(json.dumps({
                        "frame": idx,
                        "track_id": int(obj.get("track_id", -1)),
                        "entity_id": obj.get("entity_id", ""),
                        "bbox": [round(float(v), 2) for v in bbox],
                        "class": "person",
                    }, ensure_ascii=False) + "\n")
                    written += 1
                processed += 1
                if processed % 100 == 0:
                    elapsed = time.perf_counter() - started
                    print(f"  {processed} 帧 {processed/elapsed:.2f} fps", flush=True)
    finally:
        stop.set()
        reid_executor.shutdown(wait=False)
        reader_thread.join(timeout=5)
        detector_thread.join(timeout=5)
        cap.release()

    if errors:
        raise errors[0]

    elapsed = time.perf_counter() - started
    meta = {
        "source": str(src), "frames": processed, "fps": fps,
        "processed_fps": round(processed / elapsed, 2) if elapsed > 0 else 0,
        "first_frame_ms": first_frame_ms,
        "model": args.model, "imgsz": args.imgsz, "conf": args.conf,
        "batch_size": args.batch_size, "reid_interval": args.reid_interval,
        "pipeline": "reader+detector+tracker",
    }
    Path(str(out_path) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成: {processed} 帧  {meta['processed_fps']} fps -> {out_path}")


if __name__ == "__main__":
    main()
