"""一条命令生成感知 Gate 报告。

用法::

    # 已有预测文件
    python scripts/run_gate.py --video test_data/mot17_180.mp4 \
        --predictions outputs/pred/mot17_180.jsonl --out outputs/gate/mot17_180.json

    # 先导出预测再评估
    python scripts/run_gate.py --video test_data/mot17_04_raw.mp4 \
        --model yolov8l.pt --conf 0.15 --imgsz 960 --export \
        --out outputs/gate/mot17_04.json

    # 同时输出标准 MOT 指标（IDF1 / MOTA / HOTA）
    python scripts/run_gate.py --video test_data/mot17_04_raw.mp4 \
        --predictions outputs/pred/mot17_04.jsonl \
        --mot-gt MOT17/train/MOT17-04-FRCNN/gt/gt.txt \
        --out outputs/gate/mot17_04.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.report import (  # noqa: E402
    format_report,
    load_predictions_meta,
    run_evaluation,
)

DEFAULT_GT = {
    "mot17_04": "MOT17/train/MOT17-04-FRCNN/gt/gt.txt",
    "mot17_180": "MOT17/train/MOT17-04-FRCNN/gt/gt.txt",
}
VIDEO_META = {
    "mot17_04_raw": {"frame_count": 1050, "image_size": (1920, 1080), "fps": 25.0},
    "mot17_180": {"frame_count": 180, "image_size": (1920, 1080), "fps": 25.0},
}


def resolve_gt(video_path: Path) -> Path:
    stem = video_path.stem.lower()
    for key, rel in DEFAULT_GT.items():
        if key in stem:
            return Path(rel)
    raise SystemExit(f"无法为 {video_path.name} 推断真值路径，请用 --gt 指定")


def resolve_meta(video_path: Path) -> dict:
    stem = video_path.stem.lower()
    for key, meta in VIDEO_META.items():
        if key in stem:
            return dict(meta)
    return {"frame_count": 0, "image_size": (1920, 1080), "fps": 25.0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 Gate 1/2 感知评估报告")
    parser.add_argument("--video", required=True, help="验证视频路径")
    parser.add_argument("--gt", default="", help="MOT17 真值路径，缺省按视频名推断")
    parser.add_argument("--predictions", default="", help="预测 JSONL；缺省配合 --export 生成")
    parser.add_argument("--export", action="store_true", help="先跑模型导出预测")
    parser.add_argument("--model", default="yolov8l.pt")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu（CPU 需显式 --allow-cpu）")
    parser.add_argument("--allow-cpu", action="store_true", help="允许 CPU 回退（验收会失败）")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--iou", type=float, default=0.5, help="匹配 IoU 阈值")
    parser.add_argument("--min-gap", type=int, default=5, help="判定遮挡断口的最小帧间隔")
    parser.add_argument("--frame-count", type=int, default=0, help="覆盖视频帧数")
    parser.add_argument("--out", required=True, help="报告 JSON 输出路径")
    parser.add_argument("--baseline", default="", help="baseline.json，用于碎片数相对判定")
    parser.add_argument(
        "--mot-gt",
        default="",
        help="MOT17 gt.txt 路径；给了就额外算标准 MOT 指标（IDF1/MOTA/HOTA）。"
             "通常和 --gt 是同一个文件，但要显式指定才会算。",
    )
    parser.add_argument("--mot-iou", type=float, default=0.5, help="标准 MOT 指标的匹配 IoU 阈值")
    return parser


def _export_predictions(args: argparse.Namespace, out_path: Path) -> None:
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "export_predictions.py"),
        "--source", args.video,
        "--model", args.model,
        "--device", args.device,
        "--conf", str(args.conf),
        "--imgsz", str(args.imgsz),
        "--out", str(out_path),
    ]
    if args.allow_cpu:
        cmd.append("--allow-cpu")
    print("导出逐帧预测:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit("预测导出失败")


def main() -> None:
    args = build_parser().parse_args()
    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"视频不存在: {video}")

    gt_path = Path(args.gt) if args.gt else resolve_gt(video)
    if not gt_path.exists():
        raise SystemExit(f"真值不存在: {gt_path}")

    mot_gt_path = Path(args.mot_gt) if args.mot_gt else None
    if mot_gt_path is not None and not mot_gt_path.exists():
        raise SystemExit(f"标准 MOT 真值不存在: {mot_gt_path}")

    meta = resolve_meta(video)
    frame_count = args.frame_count or meta["frame_count"]

    if args.predictions:
        pred_path = Path(args.predictions)
        if not pred_path.exists():
            raise SystemExit(f"预测文件不存在: {pred_path}")
    else:
        if not args.export:
            raise SystemExit("需要 --predictions 或 --export")
        pred_path = Path("outputs/pred") / f"{video.stem}.jsonl"
        _export_predictions(args, pred_path)

    baseline_fragmentation = None
    if args.baseline and Path(args.baseline).exists():
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline_fragmentation = (
            baseline.get("gate2", {})
            .get("metrics", {})
            .get("average_entity_fragmentation")
        )

    env_meta = load_predictions_meta(pred_path)
    config = {
        "model": env_meta.get("model", args.model if args.export else "unknown"),
        "device": env_meta.get("device", args.device if args.export else "unknown"),
        "conf": env_meta.get("conf", args.conf if args.export else None),
        "imgsz": env_meta.get("imgsz", args.imgsz if args.export else None),
        "predictions": str(pred_path),
    }

    report = run_evaluation(
        gt_path=gt_path,
        predictions_path=pred_path,
        video_path=video,
        frame_count=frame_count,
        image_size=meta["image_size"],
        fps=meta["fps"],
        iou_threshold=args.iou,
        min_gap_frames=args.min_gap,
        baseline_fragmentation=baseline_fragmentation,
        config=config,
        environment=env_meta,
        mot_gt_path=mot_gt_path,
        mot_iou_threshold=args.mot_iou,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(format_report(report))
    print(f"\n报告已写入: {out_path}")


if __name__ == "__main__":
    main()
