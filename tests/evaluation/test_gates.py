"""Gate 评估模块的单元测试（合成真值与预测，不依赖视频/模型）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.gates import evaluate_frames, gate1_detection, gate2_continuity  # noqa: E402
from evaluation.matcher import iou, match_frame  # noqa: E402
from evaluation.mot17_gt import GTBox, core_frames_by_person, load_gt  # noqa: E402


def _gt(frame: int, pid: int, bbox=(10, 10, 60, 110), core=True, vis=0.9) -> GTBox:
    return GTBox(
        frame=frame,
        person_id=pid,
        bbox=bbox,
        visibility=vis,
        visible_ratio=1.0,
        is_core=core,
    )


def _pred(frame: int, track_id: int, bbox=(10, 10, 60, 110), entity_id="") -> dict:
    return {
        "frame": frame,
        "track_id": track_id,
        "entity_id": entity_id or f"Person-{track_id}",
        "bbox": list(bbox),
        "class": "person",
    }


class TestIou:
    def test_identical_boxes(self):
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_disjoint(self):
        assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_half_overlap(self):
        # 重叠 5x10=50，并集 150
        assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 0.3333) < 1e-3


class TestMatching:
    def test_one_to_one_greedy(self):
        gts = [_gt(0, 1, (0, 0, 10, 10)), _gt(0, 2, (100, 100, 110, 110))]
        preds = [_pred(0, 5, (1, 0, 11, 10)), _pred(0, 6, (100, 100, 110, 110))]
        matches = match_frame(0, gts, preds)
        assert len(matches) == 2
        assert {m.person_id for m in matches} == {1, 2}

    def test_below_threshold_not_matched(self):
        gts = [_gt(0, 1, (0, 0, 10, 10))]
        preds = [_pred(0, 5, (8, 8, 18, 18))]
        assert match_frame(0, gts, preds) == []


class TestGate1:
    def test_perfect_detection_passes(self):
        gt = {i: [_gt(i, 1)] for i in range(5)}
        pred = {i: [_pred(i, 1)] for i in range(5)}
        frames, persons, merges, entities = evaluate_frames(gt, pred)
        result = gate1_detection(frames, persons)
        assert result["passed"] is True
        assert result["metrics"]["precision_any_gt"] == 1.0

    def test_undiscovered_person_fails(self):
        gt = {0: [_gt(0, 1), _gt(0, 2, (500, 500, 550, 600))]}
        pred = {0: [_pred(0, 1)]}
        frames, persons, _, _ = evaluate_frames(gt, pred)
        result = gate1_detection(frames, persons)
        assert result["passed"] is False
        assert result["metrics"]["undiscovered_persons"] == [2]

    def test_consecutive_miss_counted(self):
        # person 1 在 0..15 帧，只有前 2 帧匹配
        gt = {i: [_gt(i, 1)] for i in range(16)}
        pred = {0: [_pred(0, 1)], 1: [_pred(1, 1)]}
        frames, persons, _, _ = evaluate_frames(gt, pred)
        result = gate1_detection(frames, persons)
        assert result["metrics"]["persons_over_miss_limit"][0]["max_consecutive_miss"] == 14

    def test_gap_breaks_consecutive_run(self):
        # 帧 0 和 10 有 GT，中间无 GT：两次单帧漏检不构成连续漏检
        gt = {0: [_gt(0, 1)], 10: [_gt(10, 1)]}
        frames, persons, _, _ = evaluate_frames(gt, {})
        result = gate1_detection(frames, persons)
        assert result["metrics"]["persons_over_miss_limit"] == []


class TestGate2:
    def test_false_merge_detected(self):
        # 两个 GT 在同一帧匹配到同一 Entity（两个 track 属于同一实体）
        gt = {0: [_gt(0, 1, (0, 0, 10, 10)), _gt(0, 2, (20, 0, 30, 10))]}
        pred = {
            0: [
                _pred(0, 5, (0, 0, 10, 10), entity_id="Person-9"),
                _pred(0, 7, (20, 0, 30, 10), entity_id="Person-9"),
            ]
        }
        frames, persons, merges, entity_persons = evaluate_frames(gt, pred)
        result = gate2_continuity(frames, persons, merges, entity_persons)
        assert result["metrics"]["false_merge_count"] == 1
        assert result["passed"] is False

    def test_fragmentation_counted(self):
        gt = {i: [_gt(i, 1)] for i in range(4)}
        pred = {
            0: [_pred(0, 5, entity_id="Person-1")],
            1: [_pred(1, 5, entity_id="Person-1")],
            2: [_pred(2, 8, entity_id="Person-2")],
            3: [_pred(3, 8, entity_id="Person-2")],
        }
        frames, persons, merges, entity_persons = evaluate_frames(gt, pred)
        result = gate2_continuity(frames, persons, merges, entity_persons)
        assert result["metrics"]["average_entity_fragmentation"] == 2.0
        assert result["passed"] is True


class TestGtLoading:
    def test_filters_mark_and_class_and_offsets_frame(self, tmp_path):
        gt_file = tmp_path / "gt.txt"
        gt_file.write_text(
            "1,1,10,10,50,100,1,1,0.9\n"      # 保留 -> frame 0
            "1,2,10,10,50,100,0,1,0.9\n"      # mark=0 -> 丢弃
            "2,3,10,10,50,100,1,2,0.9\n"      # class=2 -> 丢弃
            "1,4,10,10,50,30,1,1,0.9\n",      # 高度 30 < 40 -> 挑战目标
            encoding="utf-8",
        )
        frames = load_gt(gt_file, frame_count=10, image_size=(1920, 1080))
        assert 0 in frames
        assert len(frames[0]) == 2
        core = [b for b in frames[0] if b.is_core]
        assert len(core) == 1 and core[0].person_id == 1
        assert core_frames_by_person(frames)[1] == [0]

    def test_out_of_frame_visibility(self, tmp_path):
        gt_file = tmp_path / "gt.txt"
        # bbox 大部分在画面外：可见比例 < 0.6
        gt_file.write_text("1,1,-300,10,500,100,1,1,0.9\n", encoding="utf-8")
        frames = load_gt(gt_file, frame_count=10, image_size=(1920, 1080))
        assert frames[0][0].is_core is False
