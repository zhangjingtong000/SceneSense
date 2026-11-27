"""评估报告：把真值、预测和 Gate 指标组合成一份可复现的 JSON + 终端摘要。"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .gates import (
    MAX_CONSECUTIVE_MISS,
    TARGET_DISCOVERY_RECALL,
    TARGET_FRAME_PRECISION,
    TARGET_FRAME_RECALL,
    evaluate_frames,
    gate1_detection,
    gate2_continuity,
)
from .matcher import DEFAULT_IOU_THRESHOLD
from .mot17_gt import load_gt

SCHEMA_VERSION = "scenesense-gate-report-v1"


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def load_predictions(path: str | Path) -> dict[int, list[dict]]:
    """读取 JSONL 预测文件，按帧分组。"""
    path = Path(path)
    by_frame: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 解析失败") from exc
            frame = int(item["frame"])
            by_frame.setdefault(frame, []).append(item)
    return by_frame


def load_predictions_meta(predictions_path: str | Path) -> dict[str, Any]:
    """读取预测导出时记录的 meta.json（设备、耗时、FPS）。缺失返回空字典。"""
    path = Path(predictions_path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def gate0_environment(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Gate 0 环境检查：GPU 推理、首帧耗时、处理 FPS 必须可追溯。"""
    device = str(meta.get("device", "unknown"))
    cpu_fallback = bool(meta.get("cpu_fallback", device == "cpu"))
    gpu_ok = device.startswith("cuda") and not cpu_fallback

    checks = {
        "gpu_inference": {
            "value": device,
            "target": "cuda",
            "passed": gpu_ok,
        },
        "first_frame_latency_recorded": {
            "value": meta.get("first_frame_ms"),
            "target": "not null",
            "passed": meta.get("first_frame_ms") is not None,
        },
        "processed_fps_recorded": {
            "value": meta.get("processed_fps"),
            "target": "> 0",
            "passed": bool(meta.get("processed_fps")),
        },
    }
    return {
        "gate": 0,
        "name": "基准可重复",
        "metrics": dict(meta),
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
        "notes": [
            "CPU 回退视为配置错误（spec Gate 1 性能预算）。",
            "三次运行一致性需要多次运行后比较，本次只记录单次配置。",
        ],
    }



def run_evaluation(
    gt_path: str | Path,
    predictions_path: str | Path,
    video_path: str | Path,
    frame_count: int,
    image_size: tuple[int, int],
    fps: float,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    baseline_fragmentation: float | None = None,
    min_gap_frames: int = 5,
    config: dict[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """执行一次完整评估，返回报告字典。"""
    gt_by_frame = load_gt(gt_path, frame_count=frame_count, image_size=image_size)
    pred_by_frame = load_predictions(predictions_path)
    env_meta = dict(environment) if environment is not None else load_predictions_meta(predictions_path)

    frame_results, person_results, false_merges, entity_persons = evaluate_frames(
        gt_by_frame, pred_by_frame, iou_threshold=iou_threshold
    )

    gate0 = gate0_environment(env_meta)

    gate1 = gate1_detection(
        frame_results,
        person_results,
        target_discovery=TARGET_DISCOVERY_RECALL,
        target_recall=TARGET_FRAME_RECALL,
        target_precision=TARGET_FRAME_PRECISION,
        max_miss=MAX_CONSECUTIVE_MISS,
    )
    gate2 = gate2_continuity(
        frame_results,
        person_results,
        false_merges,
        entity_persons,
        baseline_fragmentation=baseline_fragmentation,
        gt_by_frame=gt_by_frame,
        pred_by_frame=pred_by_frame,
        fps=fps,
        iou_threshold=iou_threshold,
        min_gap_frames=min_gap_frames,
    )

    core_frames = sum(1 for boxes in gt_by_frame.values() for b in boxes if b.is_core)
    challenge_frames = sum(1 for boxes in gt_by_frame.values() for b in boxes if not b.is_core)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "code_revision": _git_revision(),
        "video": {
            "path": str(video_path),
            "frame_count": frame_count,
            "fps": fps,
            "image_size": list(image_size),
        },
        "config": config or {},
        "dataset": {
            "gt_path": str(gt_path),
            "core_gt_boxes": core_frames,
            "challenge_gt_boxes": challenge_frames,
            "frames_with_gt": len(gt_by_frame),
            "frames_with_predictions": len(pred_by_frame),
            "iou_threshold": iou_threshold,
        },
        "gate0": gate0,
        "gate1": gate1,
        "gate2": gate2,
        "summary": {
            "gate0_passed": gate0["passed"],
            "gate1_passed": gate1["passed"],
            "gate2_passed": gate2["passed"],
            "all_passed": gate0["passed"] and gate1["passed"] and gate2["passed"],
        },
    }


def format_report(report: dict[str, Any]) -> str:
    """生成终端摘要。"""
    g1 = report["gate1"]
    g2 = report["gate2"]
    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("SceneSense 感知评估报告")
    lines.append("=" * 62)
    lines.append(f"视频: {report['video']['path']}  frames={report['video']['frame_count']} "
                 f"fps={report['video']['fps']}")
    lines.append(f"代码版本: {report['code_revision']}   生成时间: {report['generated_at']}")
    lines.append(f"核心GT框: {report['dataset']['core_gt_boxes']}  "
                 f"挑战GT框: {report['dataset']['challenge_gt_boxes']}  "
                 f"IoU阈值: {report['dataset']['iou_threshold']}")
    lines.append("")
    g0 = report["gate0"]
    lines.append("[Gate 0 基准可重复 / 环境]")
    for name, check in g0["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"  {mark}  {name:28s} value={check['value']}  target={check['target']}")
    lines.append("")
    lines.append("[Gate 1 人物检测]")
    for name, check in g1["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"  {mark}  {name:28s} value={check['value']}  target={check['target']}")
    m1 = g1["metrics"]
    lines.append(f"  人物数={m1['person_count']} 已发现={m1['discovered_person_count']} "
                 f"预测框={m1['predictions']}")
    if m1["undiscovered_persons"]:
        preview = m1["undiscovered_persons"][:20]
        lines.append(f"  未发现人物(前20): {preview}")
    if m1["persons_over_miss_limit"]:
        lines.append(f"  连续漏检超限人物数: {len(m1['persons_over_miss_limit'])}")
    lines.append("")
    lines.append("[Gate 2 单目标连续性]")
    for name, check in g2["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"  {mark}  {name:28s} value={check['value']}  target={check['target']}")
    m2 = g2["metrics"]
    lines.append(f"  错误合并={m2['false_merge_count']}  "
                 f"平均碎片数={m2['average_entity_fragmentation']}")
    lines.append(f"  恢复率: 未执行标注（需 annotations/mot17_04/annotation.json）")
    lines.append("")
    lines.append(f"总体: {'通过' if report['summary']['all_passed'] else '未通过'}")
    lines.append("=" * 62)
    return "\n".join(lines)
