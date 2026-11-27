"""用"完美 GT"自检标准 MOT 指标适配器链路。

思路很简单：拿一份预测结果，原样写成 MOT17 的 ``gt.txt``（同一批框、同一个 id、
帧号 +1）。预测相对这份 GT 就是 100% 命中、零误检、零漏检、零 ID 跳变，因此适配器
必须给出 IDF1 = 1.0、MOTA = 1.0、HOTA = 1.0。任何一个不是 1.0，就说明适配器在坐标
换算（xywh/xyxy）、帧号偏移或 ID 映射上写错了。

**这不是真实评测结果**。完美 GT 是拿预测自己生成的，只能证明"链路通"，
不能证明跟踪质量；真实数字需要 MOT17 数据集。

预测来源二选一：

- ``--demo``：按 ``assets/demo/sample.mp4`` 的合成场景几何（见
  ``assets/demo/gen_demo_video.py`` 的 ``actors_at`` / ``person_bbox``）生成一份
  1280×720、200 帧的"完美检测"。没有 GPU、没有权重也能跑，适合进 CI。
- ``--pred``：已有预测文件，``scripts/export_predictions.py`` 的 JSONL 或
  MOTChallenge txt 都行。

用法::

    python -m evaluation.make_synthetic_gt --demo --workdir outputs/mot/synthetic
    python -m evaluation.make_synthetic_gt --pred outputs/pred/mot17_04.jsonl \
        --workdir outputs/mot/mot17_04
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .mot_metrics import (
    DEFAULT_IOU_THRESHOLD,
    Box,
    evaluate,
    evaluate_files,
    format_metrics,
    load_predictions_any,
)

DEMO_FRAMES = 200  # assets/demo/sample.mp4 正好 8s × 25fps
_DEMO_DIR = Path(__file__).resolve().parent.parent / "assets" / "demo"


def demo_predictions(frame_count: int = DEMO_FRAMES) -> dict[int, list[Box]]:
    """把合成示例视频的场景几何变成"完美检测"结果（0 基帧号）。

    只是几何反推，不跑检测器：4 个行人各一个稳定 id，走出画面的 P4 在画面外
    的帧不出框，和真实检测器"看不见就不报"的行为一致。
    """
    if str(_DEMO_DIR) not in sys.path:
        sys.path.insert(0, str(_DEMO_DIR))
    from gen_demo_video import FPS, N, W, actors_at, person_bbox  # 延迟导入：避免兜走 cv2

    by_frame: dict[int, list[Box]] = {}
    for i in range(min(frame_count, N)):
        t = i / FPS
        rows: list[Box] = []
        for person_id, cx, foot_y, *_ in actors_at(t):
            if not (-80 < cx < W + 80):  # 与 gen_demo_video 的可见性判定保持一致
                continue
            bbox = person_bbox(cx, foot_y)
            rows.append(Box(frame=i, track_id=person_id, bbox=bbox))
        if rows:
            by_frame[i] = rows
    return by_frame


def perfect_gt_from_predictions(
    pred_by_frame: Mapping[int, Sequence[Box]],
    *,
    frame_offset: int = 1,
) -> dict[int, list[Box]]:
    """把预测原样转成真值；``frame_offset=1`` 得到 MOT17 的 1 基帧号。"""
    gt: dict[int, list[Box]] = {}
    for frame, boxes in pred_by_frame.items():
        gt[frame + frame_offset] = [
            Box(frame=frame + frame_offset, track_id=b.track_id, bbox=b.bbox, conf=1.0)
            for b in boxes
        ]
    return dict(sorted(gt.items()))


def write_mot_gt(path: str | Path, gt_by_frame: Mapping[int, Sequence[Box]], *, visibility: float = 1.0) -> Path:
    """写 MOT17 格式的真值：``frame,id,x,y,w,h,mark,class,visibility``。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for frame in sorted(gt_by_frame):
        for box in gt_by_frame[frame]:
            x, y, w, h = box.xywh
            lines.append(f"{frame},{box.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,{visibility:.2f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def write_mot_predictions(path: str | Path, pred_by_frame: Mapping[int, Sequence[Box]], *, conf: float = 1.0) -> Path:
    """写 MOTChallenge 结果格式：``frame,id,x,y,w,h,conf,-1,-1,-1``。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for frame in sorted(pred_by_frame):
        for box in pred_by_frame[frame]:
            x, y, w, h = box.xywh
            lines.append(f"{frame},{box.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def self_check(
    pred_by_frame: Mapping[int, Sequence[Box]],
    *,
    workdir: str | Path,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict:
    """完美 GT 下跑一遍适配器，断言 IDF1 / MOTA / HOTA 都是 1.0。

    返回的是**文件往返**那一遍的结果（写 gt.txt + 预测 txt，再读回来算），
    这样坐标与帧号偏移都被真实走了一遍。
    """
    if not pred_by_frame:
        raise ValueError("预测为空，没法构造完美 GT")

    # 1) 内存里先对一遍：帧号相同的预测与 GT
    identity_gt = perfect_gt_from_predictions(pred_by_frame, frame_offset=0)
    identity = evaluate(identity_gt, pred_by_frame, iou_threshold=iou_threshold)
    _assert_perfect(identity, label="内存")

    # 2) 文件往返：GT 用 MOT17 的 1 基帧号，帧号偏移交给适配器的 frame_offset
    workdir = Path(workdir)
    gt_path = write_mot_gt(workdir / "perfect_gt.txt", perfect_gt_from_predictions(pred_by_frame, frame_offset=1))
    pred_path = write_mot_predictions(workdir / "perfect_predictions.txt", pred_by_frame)
    result = evaluate_files(
        gt_path,
        pred_path,
        iou_threshold=iou_threshold,
        gt_frame_offset=-1,
        pred_frame_offset=0,
    )
    _assert_perfect(result, label="文件")
    result["gt_path"] = str(gt_path)
    result["pred_path"] = str(pred_path)
    return result


def _assert_perfect(result: Mapping[str, object], *, label: str) -> None:
    for key in ("idf1", "mota"):
        value = result.get(key)
        if value is None:
            raise AssertionError(f"适配器自检失败（{label}）：{key} 为 None，检查后端是否装上")
        if abs(float(value) - 1.0) > 1e-6:
            raise AssertionError(f"适配器自检失败（{label}）：{key}={value}，完美 GT 下应为 1.0")
    hota = result.get("hota")
    if hota is not None and abs(float(hota) - 1.0) > 1e-6:
        raise AssertionError(f"适配器自检失败（{label}）：hota={hota}，完美 GT 下应为 1.0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成完美 GT 并自检标准 MOT 指标适配器")
    parser.add_argument("--demo", action="store_true", help="用 assets/demo/sample.mp4 的合成场景几何生成预测")
    parser.add_argument("--pred", default="", help="已有预测文件（JSONL 或 MOTChallenge txt）")
    parser.add_argument("--workdir", default="outputs/mot/synthetic", help="gt/预测/结果 JSON 的输出目录")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="MOTA/IDF1 的匹配 IoU 阈值")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.demo == bool(args.pred):
        raise SystemExit("需要且只需要 --demo 或 --pred 之一")

    if args.demo:
        pred_by_frame = demo_predictions()
    else:
        pred_by_frame = load_predictions_any(args.pred)

    result = self_check(pred_by_frame, workdir=args.workdir, iou_threshold=args.iou)
    print("适配器自检：完美 GT 下 IDF1 / MOTA / HOTA 均为 1.0（不是真实评测结果）")
    print(format_metrics(result))

    out_path = Path(args.workdir) / "selfcheck.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
