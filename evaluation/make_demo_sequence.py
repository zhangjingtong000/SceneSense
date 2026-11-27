"""生成一份小型合成 MOT 场景，用来验证标准指标适配器**真的**能反映跟踪质量。

为什么需要这个：``make_synthetic_gt.py`` 的自检是"完美 GT = 预测"，IDF1 恒为
1.0。这只能证明链路通，不能证明指标有判别力——适配器把 IDSW 恒算 0、把 FN
恒算 0，自检照样 1.0 全过。所以这里造一段**有错误的**数据：同一批 GT，配三种
预测，看指标是否按预期分开。

场景（6 个目标，200 帧，1280×720，25 fps）：

============  ==============================================  =========================
预测          怎么造的                                         期望指标变化
============  ==============================================  =========================
``perfect``   与 GT 完全一致                                   IDF1≈1.0 / MOTA≈1.0 / IDSW=0
``idsw``      2、4 号各换 1 次 ID，6 号换 2 次（共 4 次切换）  IDF1 明显下降，IDSW>0
``dropout``   1 号漏 30 帧、5 号漏 50 帧、3 号每 7 帧漏 1 帧     FN 上升，IDR/MOTA 下降
============  ==============================================  =========================

坐标是 MOTChallenge 的左上角 + 宽高；``gt.txt`` 用 1 基帧号（MOT17 约定），三个
``*.txt`` 预测用 0 基帧号（``scripts/export_predictions.py`` 约定），正好对上
``mot_metrics`` 的默认偏移 ``--gt-frame-offset -1``，所以下面这条命令不用带偏移::

    python -m evaluation.mot_metrics \
        --gt evaluation/demo_data/gt.txt \
        --pred evaluation/demo_data/idsw.txt

**这是合成数据的演示，不是真实跟踪性能。** 它证明的是适配器能把"好"和"坏"
分开，不证明任何模型在真实视频上能达到这些数字。真实评测需要 MOT17 数据集。

重新生成::

    python -m evaluation.make_demo_sequence            # 重写 demo_data/
    python -m evaluation.make_demo_sequence --report   # 顺带打印三种场景的指标
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .mot_metrics import Box, evaluate_files

DEMO_DATA_DIR = Path(__file__).resolve().parent / "demo_data"

N_FRAMES = 200
FPS = 25.0
FRAME_W, FRAME_H = 1280, 720
SCENARIOS = ("perfect", "idsw", "dropout")

# 指标里要写进 expected.json 的字段，顺序即输出顺序
METRIC_KEYS = (
    "idf1",
    "mota",
    "hota",
    "deta",
    "assa",
    "idp",
    "idr",
    "fp",
    "fn",
    "idsw",
)


@dataclass(frozen=True)
class Target:
    """一个目标的参考轨迹：水平匀速 + 轻微正弦摆动（确定性，无随机数）。"""

    track_id: int
    x0: float
    y0: float
    vx: float
    w: float
    h: float
    phase: float


# 横向间距 200px、框宽 ≤56px、200 帧内最多漂移 32px，所以任意两帧里目标框
# 都不会重叠——同帧匹配是唯一的，指标数字可复现。
TARGETS: tuple[Target, ...] = (
    Target(1, 60.0, 300.0, 0.16, 52.0, 112.0, 0.0),
    Target(2, 260.0, 330.0, -0.11, 48.0, 104.0, 1.1),
    Target(3, 460.0, 310.0, 0.09, 56.0, 118.0, 2.3),
    Target(4, 660.0, 340.0, -0.14, 50.0, 108.0, 3.4),
    Target(5, 860.0, 320.0, 0.12, 46.0, 100.0, 4.2),
    Target(6, 1060.0, 300.0, -0.08, 54.0, 116.0, 5.0),
)

# (切换发生帧, 新 ID)。帧号 >= 该值的预测换用新 ID。
ID_SWITCHES: Mapping[int, tuple[tuple[int, int], ...]] = {
    2: ((70, 102),),
    4: ((140, 204),),
    6: ((40, 306), (160, 406)),
}

# 整段漏检区间 [lo, hi)：模拟被遮挡。1 号漏 30 帧，5 号漏 50 帧。
DROP_RANGES: Mapping[int, tuple[tuple[int, int], ...]] = {
    1: ((50, 80),),
    5: ((120, 170),),
}

# 3 号再叠一层稀疏漏检：每 7 帧掉 1 帧，模拟检测器不稳定
SPARSE_DROP_ID = 3
SPARSE_DROP_PERIOD = 7
SPARSE_DROP_PHASE = 3


def _box_for(target: Target, frame: int) -> tuple[float, float, float, float]:
    """目标在第 ``frame`` 帧的 xyxy 框。"""
    x = target.x0 + target.vx * frame + 5.0 * math.sin(frame * 0.05 + target.phase)
    y = target.y0 + 4.0 * math.sin(frame * 0.09 + target.phase * 1.3)
    x = min(max(x, 0.0), FRAME_W - target.w)
    y = min(max(y, 0.0), FRAME_H - target.h)
    return (x, y, x + target.w, y + target.h)


def gt_by_frame(n_frames: int = N_FRAMES) -> dict[int, list[Box]]:
    """真值：6 个目标全程可见，帧号 0 基（写文件时才 +1）。"""
    return {
        frame: [
            Box(frame=frame, track_id=t.track_id, bbox=_box_for(t, frame))
            for t in TARGETS
        ]
        for frame in range(n_frames)
    }


def _pred_track_id(gt_id: int, frame: int) -> int:
    """``idsw`` 场景里某个 GT 目标在 ``frame`` 帧被赋予的预测 ID。"""
    track_id = gt_id
    for start, new_id in ID_SWITCHES.get(gt_id, ()):
        if frame >= start:
            track_id = new_id
    return track_id


def _is_dropped(gt_id: int, frame: int) -> bool:
    for lo, hi in DROP_RANGES.get(gt_id, ()):
        if lo <= frame < hi:
            return True
    return (
        gt_id == SPARSE_DROP_ID
        and frame % SPARSE_DROP_PERIOD == SPARSE_DROP_PHASE
    )


def _jitter(bbox: tuple[float, float, float, float], frame: int, gt_id: int) -> tuple[float, float, float, float]:
    """±2px 的确定性抖动，模拟检测框不稳定；IoU 仍有 ~0.96，不影响匹配。"""
    dx = 2.0 * math.sin(frame * 0.31 + gt_id)
    dy = 2.0 * math.sin(frame * 0.23 + gt_id * 1.7)
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def predictions(kind: str, n_frames: int = N_FRAMES) -> dict[int, list[Box]]:
    """按场景生成预测框（帧号 0 基）。"""
    if kind not in SCENARIOS:
        raise ValueError(f"未知场景 {kind!r}，可选 {SCENARIOS}")

    by_frame: dict[int, list[Box]] = {}
    for frame in range(n_frames):
        rows: list[Box] = []
        for target in TARGETS:
            if kind == "dropout" and _is_dropped(target.track_id, frame):
                continue  # 漏检：这一帧不报框 -> FN
            bbox = _box_for(target, frame)
            if kind == "dropout":
                bbox = _jitter(bbox, frame, target.track_id)
            track_id = _pred_track_id(target.track_id, frame) if kind == "idsw" else target.track_id
            rows.append(Box(frame=frame, track_id=track_id, bbox=bbox))
        if rows:
            by_frame[frame] = rows
    return by_frame


def write_gt(path: str | Path, gt: Mapping[int, Sequence[Box]], *, visibility: float = 1.0) -> Path:
    """写 MOTChallenge 真值：``frame,id,x,y,w,h,mark,class,visibility``（帧号 1 基）。

    行尾固定 LF：``.gitattributes`` 里 ``*.txt`` 是 ``eol=lf``，clone 后工作区
    就是 LF；这里若跟着 Windows 写 CRLF，``test_demo_data_is_reproducible``
    的逐字节比对会失败。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for frame in sorted(gt):
        for box in gt[frame]:
            x, y, w, h = box.xywh
            lines.append(
                f"{frame + 1},{box.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,{visibility:.2f}"
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    return path


def write_predictions(path: str | Path, pred: Mapping[int, Sequence[Box]], *, conf: float = 1.0) -> Path:
    """写 MOTChallenge 结果：``frame,id,x,y,w,h,conf,-1,-1,-1``（帧号 0 基，行尾 LF）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for frame in sorted(pred):
        for box in pred[frame]:
            x, y, w, h = box.xywh
            lines.append(
                f"{frame},{box.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1"
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    return path


def write_demo_data(out_dir: str | Path = DEMO_DATA_DIR) -> dict[str, Path]:
    """生成 ``gt.txt`` + 三个场景的预测文件，返回路径表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {"gt": out_dir / "gt.txt"}
    write_gt(paths["gt"], gt_by_frame())
    for kind in SCENARIOS:
        paths[kind] = write_predictions(out_dir / f"{kind}.txt", predictions(kind))
    return paths


def evaluate_demo(paths: Mapping[str, Path] | None = None) -> dict[str, dict]:
    """三个场景各算一遍指标（走文件往返，和用户命令行完全同一条路径）。"""
    if paths is None:
        paths = write_demo_data()
    result: dict[str, dict] = {}
    for kind in SCENARIOS:
        metrics = evaluate_files(paths["gt"], paths[kind])
        result[kind] = {key: metrics.get(key) for key in METRIC_KEYS}
        result[kind]["gt_detections"] = metrics.get("gt_detections")
        result[kind]["pred_detections"] = metrics.get("pred_detections")
        result[kind]["backends"] = metrics.get("backends")
    return result


def build_expected(paths: Mapping[str, Path] | None = None) -> dict:
    """拼出写进 ``expected.json`` 的完整内容。"""
    if paths is None:
        paths = write_demo_data()
    scenarios = evaluate_demo(paths)
    gt_count = scenarios["perfect"]["gt_detections"]
    return {
        "generated_by": "evaluation/make_demo_sequence.py",
        "note": (
            "合成数据的适配器演示，用来证明指标能区分跟踪质量，"
            "不代表真实跟踪性能；真实评测要下载 MOT17 数据集。"
        ),
        "sequence": {
            "frames": N_FRAMES,
            "fps": FPS,
            "frame_size": [FRAME_W, FRAME_H],
            "targets": len(TARGETS),
            "gt_detections": gt_count,
            "gt_frame_offset": -1,
            "pred_frame_offset": 0,
            "iou_threshold": 0.5,
        },
        "scenarios": {
            kind: {key: value for key, value in scenarios[kind].items() if key != "backends"}
            for kind in SCENARIOS
        },
        "backends": scenarios["perfect"]["backends"],
    }


def _print_table(scenarios: Mapping[str, Mapping[str, object]]) -> None:
    header = f"{'场景':<10}{'IDF1':>8}{'MOTA':>8}{'HOTA':>8}{'IDSW':>7}{'FN':>7}{'FP':>7}"
    print(header)
    print("-" * len(header))
    for kind in SCENARIOS:
        row = scenarios[kind]

        def show(key: str) -> str:
            value = row.get(key)
            return "n/a" if value is None else f"{value}"

        print(
            f"{kind:<10}{show('idf1'):>8}{show('mota'):>8}{show('hota'):>8}"
            f"{show('idsw'):>7}{show('fn'):>7}{show('fp'):>7}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成小型合成 MOT 场景（GT + 三种预测）并算指标")
    parser.add_argument(
        "--out-dir",
        default=str(DEMO_DATA_DIR),
        help="输出目录（默认 evaluation/demo_data）",
    )
    parser.add_argument("--report", action="store_true", help="打印三种场景的指标表")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="不写文件，只在内存里算一遍并打印（用于快速核对）",
    )
    parser.add_argument("--json-out", default="", help="把 expected.json 写到指定路径")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)

    if args.print_only:
        scenarios = evaluate_demo(write_demo_data(out_dir))
    else:
        paths = write_demo_data(out_dir)
        expected = build_expected(paths)
        json_path = Path(args.json_out) if args.json_out else out_dir / "expected.json"
        json_path.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
        )
        print(f"已写入 {out_dir}（gt.txt / perfect.txt / idsw.txt / dropout.txt）")
        print(f"expected.json: {json_path}")
        scenarios = expected["scenarios"]

    if args.report or args.print_only:
        _print_table(scenarios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
