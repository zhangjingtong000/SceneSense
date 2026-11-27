"""在多个 MOT 格式序列上跑泛化验证（MOT17 的 7 个场景 / DanceTrack 等）。

每个序列：图片序列 -> mp4 -> 逐帧预测 -> Gate 1/2 报告，最后汇总。

用法::

    # MOT17 的 FRCNN 版本（6 个场景，04 已单独验证过）
    python scripts/run_mot_sequences.py --root MOT17/train --pattern "MOT17-*-FRCNN" --limit 6

    # DanceTrack
    python scripts/run_mot_sequences.py --root tools/dancetrack/dancetrack/val --pattern "*"
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
    parser = argparse.ArgumentParser(description="多序列 MOT 评估")
    parser.add_argument("--root", required=True, help="包含多个序列目录的根目录")
    parser.add_argument("--pattern", default="*", help="序列目录匹配模式")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip", default="", help="跳过名字包含该子串的序列（逗号分隔）")
    parser.add_argument("--model", default="yolov8l.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--max-frames", type=int, default=0, help="每个序列最多取多少帧，0 为全部")
    parser.add_argument("--pos-gap-ratio", type=float, default=0.4,
                        help="实体位置门限比例（0 表示用默认 140px）")
    parser.add_argument("--out", default="outputs/mot_sequences")
    return parser


def find_sequences(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.glob(pattern) if p.is_dir() and (p / "img1").exists())


def read_seqinfo(seq: Path) -> dict:
    info: dict[str, str] = {}
    path = seq / "seqinfo.ini"
    if not path.exists():
        return info
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key.strip().lower()] = value.strip()
    return info


def frames_to_video(seq: Path, out_video: Path, fps: float, max_frames: int) -> int:
    import cv2

    images = sorted((seq / "img1").glob("*.jpg"))
    if max_frames:
        images = images[:max_frames]
    if not images:
        raise RuntimeError(f"{seq} 没有图片")
    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for img in images:
        writer.write(cv2.imread(str(img)))
    writer.release()
    return len(images)


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = find_sequences(root, args.pattern)
    skip = [s for s in args.skip.split(",") if s]
    sequences = [s for s in sequences if not any(k in s.name for k in skip)]
    if args.limit:
        sequences = sequences[: args.limit]
    print(f"待评估序列: {len(sequences)}")

    summaries = []
    for index, seq in enumerate(sequences, start=1):
        info = read_seqinfo(seq)
        fps = float(info.get("framerate", 25) or 25)
        video = out_dir / f"{seq.name}.mp4"
        pred = out_dir / f"{seq.name}.pred.jsonl"
        report = out_dir / f"{seq.name}.gate.json"

        if not video.exists():
            frames = frames_to_video(seq, video, fps, args.max_frames)
            print(f"[{index}/{len(sequences)}] {seq.name}: {frames} 帧")

        export_cmd = [
            sys.executable, str(ROOT / "scripts" / "export_predictions.py"),
            "--source", str(video), "--model", args.model, "--device", args.device,
            "--conf", str(args.conf), "--imgsz", str(args.imgsz),
            "--fps", str(fps), "--out", str(pred),
        ]
        subprocess.run(export_cmd, check=True, stdout=subprocess.DEVNULL)

        gate_cmd = [
            sys.executable, str(ROOT / "scripts" / "run_gate.py"),
            "--video", str(video), "--gt", str(seq / "gt" / "gt.txt"),
            "--predictions", str(pred), "--out", str(report),
            "--frame-count", str(args.max_frames or int(info.get("seqlength", 0) or 0)),
        ]
        subprocess.run(gate_cmd, check=True, stdout=subprocess.DEVNULL)

        data = json.loads(report.read_text(encoding="utf-8"))
        summary = {
            "sequence": seq.name,
            "frames": data["dataset"]["frames_with_gt"],
            "core_boxes": data["dataset"]["core_gt_boxes"],
            "detection_recall": data["gate1"]["checks"]["detection_recall"]["value"],
            "detection_precision": data["gate1"]["checks"]["detection_precision"]["value"],
            "discovery": data["gate1"]["checks"]["person_discovery_recall"]["value"],
            "max_miss": data["gate1"]["checks"]["max_consecutive_miss"]["value"],
            "false_merge": data["gate2"]["metrics"]["false_merge_count"],
            "fragmentation": data["gate2"]["metrics"]["average_entity_fragmentation"],
            "persons": data["gate1"]["metrics"]["person_count"],
        }
        summaries.append(summary)
        print(f"    召回={summary['detection_recall']:.3f} "
              f"精确={summary['detection_precision']:.3f} "
              f"发现={summary['discovery']:.3f} "
              f"误合并={summary['false_merge']} 碎片={summary['fragmentation']}")

    out_summary = out_dir / "summary.json"
    out_summary.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    if summaries:
        n = len(summaries)
        print("\n===== 汇总 =====")
        print(f"场景数 {n}")
        print(f"平均召回 {sum(s['detection_recall'] for s in summaries)/n:.3f}")
        print(f"平均精确 {sum(s['detection_precision'] for s in summaries)/n:.3f}")
        print(f"平均发现率 {sum(s['discovery'] for s in summaries)/n:.3f}")
        print(f"总错误合并 {sum(s['false_merge'] for s in summaries)}")
        print(f"平均碎片 {sum(s['fragmentation'] for s in summaries)/n:.3f}")
        worst = min(summaries, key=lambda s: s["detection_recall"])
        print(f"最差召回: {worst['sequence']} ({worst['detection_recall']:.3f})")
    print(f"\n汇总文件: {out_summary}")


if __name__ == "__main__":
    main()
