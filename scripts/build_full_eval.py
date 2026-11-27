"""构建全序列评测集：把每个 MOT17 场景从 300 帧扩到完整序列。

目的：把短遮挡事件量从 ~88 提到 300+，使 bootstrap 置信区间收窄到可判断
5 个百分点差异的水平。

用法::

    python scripts/build_full_eval.py --root MOT17/train --out outputs/full_eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="全序列评测集构建")
    parser.add_argument("--root", default="MOT17/train")
    parser.add_argument("--pattern", default="MOT17-*-FRCNN")
    parser.add_argument("--model", default="yolov8l.engine")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reid-interval", type=int, default=3)
    parser.add_argument("--skip", default="", help="跳过的场景关键词")
    parser.add_argument("--out", default="outputs/full_eval")
    return parser


def ensure_video(seq: Path, target: Path) -> int:
    """全序列生成 mp4（缓存）。返回帧数。"""
    import cv2

    images = sorted((seq / "img1").glob("*.jpg"))
    if not images:
        raise RuntimeError(f"{seq} 无图片")
    if target.exists() and target.stat().st_size > 0:
        cap = cv2.VideoCapture(str(target))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if count == len(images):
            return count
        target.unlink()

    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]
    info: dict[str, str] = {}
    seqinfo = seq / "seqinfo.ini"
    if seqinfo.exists():
        for line in seqinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key.strip().lower()] = value.strip()
    fps = float(info.get("framerate", 25) or 25)

    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for image in images:
        writer.write(cv2.imread(str(image)))
    writer.release()
    return len(images)


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted(p for p in root.glob(args.pattern) if (p / "img1").exists())
    skip = [s for s in args.skip.split(",") if s]
    sequences = [s for s in sequences if not any(k in s.name for k in skip)]
    print(f"场景数: {len(sequences)}")

    summary = []
    for seq in sequences:
        video = out_dir / f"{seq.name}.mp4"
        frames = ensure_video(seq, video)
        pred = out_dir / f"{seq.name}.base.jsonl"
        report = out_dir / f"{seq.name}.base.gate.json"
        print(f"[{seq.name}] {frames} 帧", flush=True)

        if not pred.exists() or pred.stat().st_size == 0:
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "export_predictions_pipelined.py"),
                "--source", str(video), "--model", args.model, "--device", args.device,
                "--conf", "0.15", "--imgsz", str(args.imgsz),
                "--reid-interval", str(args.reid_interval),
                "--batch-size", str(args.batch_size),
                "--out", str(pred),
            ], check=True)
        if not report.exists() or report.stat().st_size == 0:
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "run_gate.py"),
                "--video", str(video), "--gt", str(seq / "gt" / "gt.txt"),
                "--predictions", str(pred), "--frame-count", str(frames),
                "--out", str(report),
            ], check=True, stdout=subprocess.DEVNULL)

        data = json.loads(report.read_text(encoding="utf-8"))
        detail = data["gate2"]["metrics"].get("recovery_detail") or {}
        short = detail.get("short_gap") or {}
        medium = detail.get("medium_gap") or {}
        entry = {
            "sequence": seq.name,
            "frames": frames,
            "short_rate": short.get("rate"),
            "short_events": short.get("events", 0),
            "medium_rate": medium.get("rate"),
            "medium_events": medium.get("events", 0),
            "recall": data["gate1"]["checks"]["detection_recall"]["value"],
            "precision": data["gate1"]["checks"]["detection_precision"]["value"],
            "merge": data["gate2"]["metrics"]["false_merge_count"],
        }
        summary.append(entry)
        print(f"    短遮挡 {entry['short_rate']} ({entry['short_events']} 例) "
              f"召回 {entry['recall']:.3f}", flush=True)

    out_file = out_dir / "summary.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    total_events = sum(e["short_events"] for e in summary)
    print(f"\n汇总: {len(summary)} 场景，短遮挡事件合计 {total_events}")
    print(f"明细: {out_file}")


if __name__ == "__main__":
    main()
