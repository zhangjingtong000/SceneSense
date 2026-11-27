"""MOT17 真值加载与核心目标筛选。

真值格式（MOT17 gt.txt，逗号分隔，无表头）::

    frame, id, x, y, w, h, mark, class, visibility

核心目标口径如下:

- 帧号统一使用视频的 0 基帧号：``video_frame = mot_frame - 1``。
- 只有 ``mark == 1`` 且 ``class == 1`` 的行才视为行人。
- 核心目标（计入硬门槛）：visibility >= 0.5，bbox 高度 >= 40px，
  bbox 与画面交集面积占原 bbox 面积 >= 60%。
- 挑战目标（单独报告，不阻塞闭合）：其余行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MIN_CORE_VISIBILITY = 0.5
MIN_CORE_HEIGHT = 40.0
MIN_CORE_VISIBLE_RATIO = 0.6


@dataclass(frozen=True)
class GTBox:
    """一帧里的一个真值行人框。"""

    frame: int
    person_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    visibility: float
    visible_ratio: float
    is_core: bool

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


def _clip_box(
    x: float, y: float, w: float, h: float, image_w: int, image_h: int
) -> tuple[tuple[float, float, float, float], float]:
    """裁剪到画面内，返回 (x1,y1,x2,y2) 与可见面积占比。"""
    x1, y1, x2, y2 = x, y, x + w, y + h
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(image_w), x2), min(float(image_h), y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return (cx1, cy1, cx2, cy2), 0.0
    original = max(w, 1e-6) * max(h, 1e-6)
    visible = (cx2 - cx1) * (cy2 - cy1)
    return (cx1, cy1, cx2, cy2), min(1.0, visible / original)


def load_gt(
    path: str | Path,
    frame_count: int | None = None,
    image_size: tuple[int, int] = (1920, 1080),
    frame_offset: int = -1,
) -> dict[int, list[GTBox]]:
    """加载 MOT17 真值，按 0 基帧号分组。

    Args:
        path: gt.txt 路径。
        frame_count: 只保留 [0, frame_count) 的帧；None 表示不限制。
        image_size: 视频画面宽高，用于计算可见面积与裁剪。
        frame_offset: mot_frame -> video_frame 的偏移，默认 -1。

    Returns:
        {video_frame: [GTBox, ...]}，按帧号升序。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"真值文件不存在: {path}")

    image_w, image_h = image_size
    by_frame: dict[int, list[GTBox]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) < 9:
                raise ValueError(f"{path}:{line_no} 字段不足 9 个: {raw!r}")
            try:
                mot_frame = int(float(parts[0]))
                person_id = int(float(parts[1]))
                x, y, w, h = (float(parts[i]) for i in range(2, 6))
                mark = int(float(parts[6]))
                class_id = int(float(parts[7]))
                visibility = float(parts[8])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} 解析失败: {raw!r}") from exc

            if mark != 1 or class_id != 1:
                continue

            frame = mot_frame + frame_offset
            if frame < 0:
                continue
            if frame_count is not None and frame >= frame_count:
                continue

            bbox, visible_ratio = _clip_box(x, y, w, h, image_w, image_h)
            height = max(0.0, bbox[3] - bbox[1])
            is_core = (
                visibility >= MIN_CORE_VISIBILITY
                and height >= MIN_CORE_HEIGHT
                and visible_ratio >= MIN_CORE_VISIBLE_RATIO
            )
            by_frame.setdefault(frame, []).append(
                GTBox(
                    frame=frame,
                    person_id=person_id,
                    bbox=bbox,
                    visibility=visibility,
                    visible_ratio=visible_ratio,
                    is_core=is_core,
                )
            )

    for boxes in by_frame.values():
        boxes.sort(key=lambda b: b.person_id)
    return dict(sorted(by_frame.items()))


def core_frames_by_person(gt_by_frame: dict[int, list[GTBox]]) -> dict[int, list[int]]:
    """返回每个 GT 人物出现过的核心帧列表（升序）。"""
    frames: dict[int, list[int]] = {}
    for frame, boxes in gt_by_frame.items():
        for box in boxes:
            if box.is_core:
                frames.setdefault(box.person_id, []).append(frame)
    for pid in frames:
        frames[pid].sort()
    return frames
