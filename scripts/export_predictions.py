"""逐帧导出感知结果，供 Gate 评估使用。

输出 JSONL，每行::

    {"frame": 0, "track_id": 3, "entity_id": "Person-1", "bbox": [x1,y1,x2,y2], "class": "person"}

同时在 JSONL 旁写一份 ``<name>.meta.json``，记录设备、模型、首帧耗时与处理 FPS——
验收 spec 要求这些信息必须进入报告，且 **CPU 回退视为配置错误**。

用法::

    python scripts/export_predictions.py --source test_data/mot17_180.mp4 \
        --model yolov8l.pt --conf 0.15 --imgsz 960 --device cuda \
        --out outputs/pred/mot17_180.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.detector import YOLODetector  # noqa: E402
from perception.entity_profile import EntityProfileManager  # noqa: E402
from perception.tracker import MultiObjectTracker  # noqa: E402

PERSON_CLASS_ID = 0


def resolve_device(requested: str, allow_cpu: bool) -> str:
    """解析推理设备。CPU 回退默认视为配置错误。"""
    import torch

    requested = (requested or "auto").lower()
    cuda_ok = torch.cuda.is_available()

    if requested == "auto":
        if cuda_ok:
            return "cuda"
        if allow_cpu:
            print("警告: 未检测到 CUDA，按 --allow-cpu 使用 CPU（验收报告会标记为配置错误）")
            return "cpu"
        raise SystemExit(
            "配置错误: 未检测到可用 CUDA。验收要求 GPU 推理；"
            "确需 CPU 冒烟请显式加 --allow-cpu。"
        )

    if requested == "cpu" and not allow_cpu:
        raise SystemExit(
            "配置错误: 验收不接受 CPU 推理（spec Gate 1）。"
            "确需 CPU 冒烟请显式加 --allow-cpu。"
        )

    if requested.startswith("cuda") and not cuda_ok:
        raise SystemExit(f"配置错误: 请求 {requested} 但 CUDA 不可用")

    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出逐帧人物预测（JSONL）")
    parser.add_argument("--source", required=True, help="视频路径")
    parser.add_argument("--model", default="yolov8l.pt", help="YOLO 模型")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--allow-cpu", action="store_true", help="允许 CPU 回退（仅冒烟，验收会失败）")
    parser.add_argument("--conf", type=float, default=0.15, help="检测置信度阈值")
    parser.add_argument("--imgsz", type=int, default=960, help="推理尺寸")
    parser.add_argument("--out", required=True, help="输出 JSONL 路径")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数，0 为全部")
    parser.add_argument("--start-frame", type=int, default=0, help="从第几帧开始")
    parser.add_argument("--fps", type=float, default=0.0, help="覆盖视频 FPS，0 为自动")
    parser.add_argument("--reid-interval", type=int, default=1,
                        help="每 N 帧刷新一次外观特征（1 最准；实测 3 会引入错误合并）")
    parser.add_argument("--match-thresh", type=float, default=0.65,
                        help="关联匹配阈值（越大越宽松；实测 0.55 最优）")
    parser.add_argument("--proximity-thresh", type=float, default=0.5,
                        help="允许外观参与匹配的最大 IoU 距离（实测 0.35 最优）")
    parser.add_argument("--appearance-thresh", type=float, default=0.25,
                        help="外观距离上限，超过视为不匹配")
    parser.add_argument("--lost-buffer", type=int, default=120,
                        help="丢失轨迹保留帧数（决定能跨多长遮挡）")
    parser.add_argument("--lost-low", action="store_true",
                        help="实验项：让 lost 轨迹参与低分框匹配（多场景验证为负收益，默认关闭）")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="批量检测帧数（实测 4 最优，过大反而变慢）")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"视频不存在: {src}")

    device = resolve_device(args.device, args.allow_cpu)
    print(f"推理设备: {device}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {src}")
    video_fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = YOLODetector(
        model_path=args.model,
        device=device,
        conf_threshold=args.conf,
        classes=[PERSON_CLASS_ID],
        imgsz=args.imgsz,
    )
    tracker = MultiObjectTracker(
        track_activation_threshold=0.18,
        lost_track_buffer=args.lost_buffer,
        minimum_matching_threshold=args.match_thresh,
        # ReID 跟随检测设备，否则每帧都在 CPU 上跑 ResNet18（实测慢 11 倍）
        reid_device=device if device.startswith("cuda") else "cpu",
        reid_interval=max(1, args.reid_interval),
        include_lost_in_low_stage=args.lost_low,
    )
    tracker.proximity_thresh = args.proximity_thresh
    tracker.appearance_thresh = args.appearance_thresh
    profiles = EntityProfileManager()
    # 位置门限按画面宽度比例（跨遮挡合并需要；140px 默认值在 1280/1920 画面下过严）
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    profiles.max_position_gap_px = max(profiles.max_position_gap_px, 0.4 * frame_width)

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_idx = args.start_frame
    written = 0
    first_frame_ms: float | None = None
    started = time.perf_counter()
    limit = args.max_frames if args.max_frames > 0 else total_frames - args.start_frame

    with out_path.open("w", encoding="utf-8") as f:
        while frame_idx - args.start_frame < limit:
            # 批量读帧 + 一次前向，提升 GPU 利用率（实测 batch=4 约 1.5x）
            batch_frames = []
            batch_indices = []
            while len(batch_frames) < args.batch_size and frame_idx - args.start_frame < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                frame_idx += 1
            if not batch_frames:
                break

            frame_started = time.perf_counter()
            if len(batch_frames) > 1:
                batch_dets = detector.detect_batch(batch_frames)
            else:
                batch_dets = [detector.detect(batch_frames[0])]

            for frame, detections, idx in zip(batch_frames, batch_dets, batch_indices):
                timestamp = idx / video_fps
                tracked = tracker.update(detections, idx, timestamp, frame)
                enriched = profiles.update(tracked, idx, timestamp) if tracked else []
                if first_frame_ms is None:
                    first_frame_ms = round((time.perf_counter() - frame_started) * 1000, 1)

                for obj in enriched:
                    if obj.get("class") != "person":
                        continue
                    bbox = obj.get("bbox")
                    if not bbox:
                        continue
                    record = {
                        "frame": idx,
                        "track_id": int(obj.get("track_id", -1)),
                        "entity_id": obj.get("entity_id", ""),
                        "bbox": [round(float(v), 2) for v in bbox],
                        "class": "person",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

            if (frame_idx - args.start_frame) % 50 < args.batch_size:
                elapsed = time.perf_counter() - started
                done = frame_idx - args.start_frame
                print(f"  已处理 {done} 帧  记录 {written} 条  "
                      f"{done / elapsed:.2f} fps", flush=True)

    cap.release()
    elapsed = time.perf_counter() - started
    processed = frame_idx - args.start_frame
    processed_fps = round(processed / elapsed, 2) if elapsed > 0 else 0.0

    meta = {
        "device": device,
        "model": args.model,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "source": str(src),
        "fps": video_fps,
        "frames": processed,
        "records": written,
        "first_frame_ms": first_frame_ms,
        "processed_fps": processed_fps,
        "elapsed_s": round(elapsed, 2),
        "cpu_fallback": device == "cpu",
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成: {processed} 帧, {written} 条记录, 耗时 {elapsed:.1f}s "
          f"({processed_fps} fps), 首帧 {first_frame_ms} ms")
    print(f"输出: {out_path}")
    print(f"元信息: {meta_path}")


if __name__ == "__main__":
    main()
