"""标准 MOT 指标适配器：IDF1 / MOTA / HOTA / FP / FN / IDSW。

为什么单独一个模块：``gates.py`` 用的是本项目自建的验收口径（核心目标筛选、贪心
IoU 匹配、Entity 级连续性）。标准指标要求**全部** GT 参与、匈牙利匹配、按 ``id``
统计 ID 跳变，两者口径不同，混在一个函数里数字就不可比了。这里只做格式对齐 +
后端调用，不重复实现匹配算法。

输入统一是 MOTChallenge 行格式（逗号分隔、无表头）::

    真值  gt.txt     : frame, id, x, y, w, h, mark, class, visibility
    预测  result.txt : frame, id, x, y, w, h, conf[, -1, -1, -1]

坐标是左上角 + 宽高（MOTChallenge 约定），内部转成 xyxy 算 IoU。真值里
``mark == 1`` 且 ``class == 1``（行人）的行才计入；只有 7 列、没有 mark/class
的文件按"全是有效行人"处理（合成 GT 走这条路）。

后端：

- IDF1 / MOTA / IDP / IDR / FP / FN / IDSW：py-motmetrics（pip 包名 ``motmetrics``）
- HOTA：TrackEval 的 ``HOTA`` 实现（pip 包名 ``trackeval``）

任一后端没装，对应字段返回 ``None``、``backends`` 里写明原因，另一种指标照常算。

用法::

    python -m evaluation.mot_metrics \
        --gt MOT17/train/MOT17-04-FRCNN/gt/gt.txt \
        --pred outputs/pred/mot17_04.txt --out outputs/mot/mot17_04.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .matcher import iou

DEFAULT_IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class Box:
    """一帧里的一个框，统一用 xyxy。"""

    frame: int
    track_id: int
    bbox: tuple[float, float, float, float]
    conf: float = 1.0

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)


def _split(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",")]


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (x, y, x + w, y + h)


def load_mot_gt(
    path: str | Path,
    *,
    frame_offset: int = -1,
    classes: Sequence[int] = (1,),
    require_mark: bool = True,
    min_visibility: float = 0.0,
) -> dict[int, list[Box]]:
    """读取 MOT17 ``gt.txt``。

    Args:
        frame_offset: MOT 帧号 -> 预测帧号的偏移。仓库约定预测是 0 基
            （``scripts/export_predictions.py``），MOT17 gt 是 1 基，故默认 -1。
        classes: 计入的类别号；9 列格式里第 8 列。
        require_mark: 只保留 ``mark == 1`` 的行（MOT17 用它标"这一帧是否标注"）。
        min_visibility: 可见度下限，0 表示不筛。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"真值文件不存在: {path}")

    by_frame: dict[int, list[Box]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            parts = _split(raw)
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no} 字段不足 6 个: {raw!r}")
            try:
                mot_frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x, y, w, h = (float(parts[i]) for i in range(2, 6))
                if len(parts) >= 9:
                    mark = int(float(parts[6]))
                    class_id = int(float(parts[7]))
                    visibility = float(parts[8])
                    if require_mark and mark != 1:
                        continue
                    if classes and class_id not in classes:
                        continue
                    if visibility < min_visibility:
                        continue
                else:
                    visibility = 1.0
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} 解析失败: {raw!r}") from exc

            frame = mot_frame + frame_offset
            if frame < 0 or w <= 0 or h <= 0:
                continue
            by_frame.setdefault(frame, []).append(
                Box(frame=frame, track_id=track_id, bbox=_xywh_to_xyxy(x, y, w, h), conf=visibility)
            )

    for boxes in by_frame.values():
        boxes.sort(key=lambda b: b.track_id)
    return dict(sorted(by_frame.items()))


def load_mot_predictions(
    path: str | Path,
    *,
    frame_offset: int = 0,
    min_conf: float | None = None,
) -> dict[int, list[Box]]:
    """读取 MOTChallenge 结果文件（第 7 列是置信度）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"预测文件不存在: {path}")

    by_frame: dict[int, list[Box]] = {}
    seen: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            parts = _split(raw)
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no} 字段不足 6 个: {raw!r}")
            try:
                mot_frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x, y, w, h = (float(parts[i]) for i in range(2, 6))
                conf = float(parts[6]) if len(parts) >= 7 else 1.0
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} 解析失败: {raw!r}") from exc

            # conf < 0 是 MOT gt 的占位值，当作"未知置信度"而不是低分
            if min_conf is not None and conf >= 0 and conf < min_conf:
                continue
            frame = mot_frame + frame_offset
            if frame < 0 or w <= 0 or h <= 0:
                continue
            key = (frame, track_id)
            if key in seen:
                raise ValueError(f"{path}:{line_no} 同一帧出现重复 track_id={track_id}（frame={frame}）")
            seen.add(key)
            by_frame.setdefault(frame, []).append(
                Box(frame=frame, track_id=track_id, bbox=_xywh_to_xyxy(x, y, w, h), conf=conf)
            )

    for boxes in by_frame.values():
        boxes.sort(key=lambda b: b.track_id)
    return dict(sorted(by_frame.items()))


def load_predictions_jsonl(path: str | Path, *, min_conf: float | None = None) -> dict[int, list[Box]]:
    """读取 ``scripts/export_predictions.py`` 输出的 JSONL（帧号 0 基、bbox 是 xyxy）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"预测文件不存在: {path}")

    by_frame: dict[int, list[Box]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
                frame = int(item["frame"])
                track_id = int(item.get("track_id", -1))
                bbox = item["bbox"]
                x1, y1, x2, y2 = (float(bbox[i]) for i in range(4))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no} 解析失败: {raw!r}") from exc
            if x2 <= x1 or y2 <= y1:
                continue
            by_frame.setdefault(frame, []).append(
                Box(frame=frame, track_id=track_id, bbox=(x1, y1, x2, y2))
            )

    for boxes in by_frame.values():
        boxes.sort(key=lambda b: b.track_id)
    return dict(sorted(by_frame.items()))


def load_predictions_any(path: str | Path, *, min_conf: float | None = None) -> dict[int, list[Box]]:
    """按后缀分派：``.jsonl`` 走逐帧导出格式，其余按 MOTChallenge txt 读。"""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return load_predictions_jsonl(path, min_conf=min_conf)
    return load_mot_predictions(path, min_conf=min_conf)


def _iou_matrix(gt_boxes: Sequence[Box], pred_boxes: Sequence[Box]) -> list[list[float]]:
    return [[iou(g.bbox, p.bbox) for p in pred_boxes] for g in gt_boxes]


def _motmetrics_block(
    gt_by_frame: Mapping[int, Sequence[Box]],
    pred_by_frame: Mapping[int, Sequence[Box]],
    frames: Sequence[int],
    iou_threshold: float,
) -> tuple[dict, str]:
    """IDF1 / MOTA / IDP / IDR / FP / FN / IDSW（py-motmetrics）。"""
    try:
        import motmetrics as mm
    except ImportError:
        return (
            {k: None for k in ("idf1", "mota", "idp", "idr", "fp", "fn", "idsw")},
            "unavailable（pip install motmetrics 后才能算 IDF1/MOTA）",
        )

    import numpy as np

    acc = mm.MOTAccumulator(auto_id=True)
    for frame in frames:
        gts = gt_by_frame.get(frame, ())
        preds = pred_by_frame.get(frame, ())
        scores = _iou_matrix(gts, preds)
        # 1 - IoU 作代价；低于阈值的配对用 NaN 标记为"不可匹配"
        dists = [
            [math.nan if score < iou_threshold else 1.0 - score for score in row]
            for row in scores
        ]
        acc.update(
            [g.track_id for g in gts],
            [p.track_id for p in preds],
            dists,
        )

    metrics = mm.metrics.create()
    summary = metrics.compute(
        acc,
        metrics=[
            "idf1",
            "mota",
            "idp",
            "idr",
            "num_false_positives",
            "num_misses",
            "num_switches",
        ],
        name="scenesense",
    )

    def value(column: str) -> float | None:
        raw = float(summary.loc["scenesense", column])
        # 空序列会出现 inf/NaN（MOTA 在 GT 为 0 时是 -inf），JSON 里不能留这种值
        if math.isnan(raw) or math.isinf(raw):
            return None
        return round(raw, 4)

    def count(column: str) -> int | None:
        raw = value(column)
        return None if raw is None else int(raw)

    block = {
        "idf1": value("idf1"),
        "mota": value("mota"),
        "idp": value("idp"),
        "idr": value("idr"),
        "fp": count("num_false_positives"),
        "fn": count("num_misses"),
        "idsw": count("num_switches"),
    }
    return block, f"motmetrics {mm.__version__}"


def _hota_block(
    gt_by_frame: Mapping[int, Sequence[Box]],
    pred_by_frame: Mapping[int, Sequence[Box]],
    frames: Sequence[int],
) -> tuple[dict, str]:
    """HOTA / DetA / AssA（TrackEval 的 HOTA 实现）。

    HOTA 自己扫 IoU 阈值（0.05 到 0.95），所以这里传原始 IoU，不用 ``--iou``。
    """
    empty = {"hota": None, "deta": None, "assa": None}
    if not gt_by_frame or not pred_by_frame:
        return empty, "skipped（真值或预测为空）"
    try:
        from trackeval.metrics.hota import HOTA
    except ImportError:
        return empty, "unavailable（pip install trackeval 后才能算 HOTA）"

    import numpy as np

    gt_ids = sorted({b.track_id for boxes in gt_by_frame.values() for b in boxes})
    pred_ids = sorted({b.track_id for boxes in pred_by_frame.values() for b in boxes})
    gt_index = {tid: i for i, tid in enumerate(gt_ids)}
    pred_index = {tid: i for i, tid in enumerate(pred_ids)}

    data = {
        "num_timesteps": len(frames),
        "num_gt_dets": sum(len(gt_by_frame.get(f, ())) for f in frames),
        "num_tracker_dets": sum(len(pred_by_frame.get(f, ())) for f in frames),
        "num_gt_ids": len(gt_ids),
        "num_tracker_ids": len(pred_ids),
        "gt_ids": [
            np.array([gt_index[b.track_id] for b in gt_by_frame.get(f, ())], dtype=int)
            for f in frames
        ],
        "tracker_ids": [
            np.array([pred_index[b.track_id] for b in pred_by_frame.get(f, ())], dtype=int)
            for f in frames
        ],
        "similarity_scores": [
            np.array(_iou_matrix(gt_by_frame.get(f, ()), pred_by_frame.get(f, ())), dtype=float)
            .reshape(len(gt_by_frame.get(f, ())), len(pred_by_frame.get(f, ())))
            for f in frames
        ],
    }

    result = HOTA().eval_sequence(data)
    block = {
        "hota": round(float(np.mean(result["HOTA"])), 4),
        "deta": round(float(np.mean(result["DetA"])), 4),
        "assa": round(float(np.mean(result["AssA"])), 4),
    }
    try:
        from importlib.metadata import version

        backend = f"trackeval {version('trackeval')}"
    except Exception:  # pragma: no cover - 元数据缺失时仍能用
        backend = "trackeval"
    return block, backend


def evaluate(
    gt_by_frame: Mapping[int, Sequence[Box]],
    pred_by_frame: Mapping[int, Sequence[Box]],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict:
    """在内存中的逐帧框上算标准指标。

    帧集合取真值与预测的并集：GT 没有框而预测有框，按误检计。
    """
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    mot_block, mot_backend = _motmetrics_block(gt_by_frame, pred_by_frame, frames, iou_threshold)
    hota_block, hota_backend = _hota_block(gt_by_frame, pred_by_frame, frames)

    return {
        **mot_block,
        **hota_block,
        "gt_detections": sum(len(gt_by_frame.get(f, ())) for f in frames),
        "pred_detections": sum(len(pred_by_frame.get(f, ())) for f in frames),
        "frames": len(frames),
        "iou_threshold": iou_threshold,
        "backends": {"mota_idf1": mot_backend, "hota": hota_backend},
    }


def evaluate_files(
    gt_path: str | Path,
    pred_path: str | Path,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    gt_frame_offset: int = -1,
    pred_frame_offset: int = 0,
    min_conf: float | None = None,
) -> dict:
    """直接从两个文件算标准指标。"""
    gt = load_mot_gt(gt_path, frame_offset=gt_frame_offset)
    pred = load_predictions_any(pred_path, min_conf=min_conf)
    if pred_frame_offset:
        pred = {
            frame + pred_frame_offset: [
                Box(frame=frame + pred_frame_offset, track_id=b.track_id, bbox=b.bbox, conf=b.conf)
                for b in boxes
            ]
            for frame, boxes in pred.items()
        }
    result = evaluate(gt, pred, iou_threshold=iou_threshold)
    result["gt_path"] = str(gt_path)
    result["pred_path"] = str(pred_path)
    return result


def format_metrics(result: Mapping[str, object]) -> str:
    """终端摘要。"""
    def show(key: str) -> str:
        value = result.get(key)
        return "n/a" if value is None else f"{value}"

    lines = [
        f"IDF1={show('idf1')}  MOTA={show('mota')}  HOTA={show('hota')}"
        f"  (DetA={show('deta')}  AssA={show('assa')})",
        f"IDP={show('idp')}  IDR={show('idr')}  "
        f"FP={show('fp')}  FN={show('fn')}  IDSW={show('idsw')}",
        f"GT 框={result.get('gt_detections')}  预测框={result.get('pred_detections')}  "
        f"帧数={result.get('frames')}  IoU 阈值={result.get('iou_threshold')}",
        f"后端: {result.get('backends')}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 MOTChallenge 口径计算 IDF1 / MOTA / HOTA")
    parser.add_argument("--gt", required=True, help="MOT17 gt.txt 路径")
    parser.add_argument("--pred", required=True, help="预测文件：MOTChallenge txt 或 export_predictions.py 的 JSONL")
    parser.add_argument("--out", default="", help="结果 JSON 输出路径")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="MOTA/IDF1 的匹配 IoU 阈值")
    parser.add_argument("--gt-frame-offset", type=int, default=-1, help="GT 帧号偏移（MOT17 1 基 -> 预测 0 基，默认 -1）")
    parser.add_argument("--pred-frame-offset", type=int, default=0, help="预测帧号偏移（标准 MOTChallenge 文件为 0）")
    parser.add_argument("--min-conf", type=float, default=None, help="预测置信度下限")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_files(
        args.gt,
        args.pred,
        iou_threshold=args.iou,
        gt_frame_offset=args.gt_frame_offset,
        pred_frame_offset=args.pred_frame_offset,
        min_conf=args.min_conf,
    )
    print(format_metrics(result))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
