"""标准 MOT 指标适配器（``evaluation/mot_metrics.py``）的测试。

验四件事：

1. MOTChallenge 行格式的解析（mark/class 过滤、帧号偏移、xywh→xyxy、置信度过滤）。
2. **适配器自检**：完美 GT（预测即 GT）下必须是 IDF1=1.0 / MOTA=1.0 / HOTA=1.0。
   这是链路自检，不是真实评测结果。
3. 已知错误模式（ID 跳变、误检、漏检、只有预测的帧）必须被如实计入。
4. **合成序列的判别力**（``evaluation/make_demo_sequence.py`` 的 200 帧 / 6 目标
   数据）：同一份 GT 配三种预测，指标必须按缺陷类型分开。这组断言能抓住
   "适配器把 IDSW 恒算 0 / 把 FN 恒算 0 / 把 IDF1 恒算 1" 这类回归。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from evaluation import make_demo_sequence, make_synthetic_gt  # noqa: E402
from evaluation.mot_metrics import (  # noqa: E402
    Box,
    evaluate,
    evaluate_files,
    load_mot_gt,
    load_mot_predictions,
    load_predictions_jsonl,
)
from evaluation.report import run_evaluation  # noqa: E402


def _box(frame: int, track_id: int, bbox=(0.0, 0.0, 50.0, 100.0)) -> Box:
    return Box(frame=frame, track_id=track_id, bbox=tuple(float(v) for v in bbox))


def _lf(data: bytes) -> bytes:
    """把 CRLF 归一成 LF。

    ``.gitattributes`` 给 ``*.txt`` / ``*.json`` 定了 ``eol=lf``，生成器也固定写
    LF；这里归一化只是让"不同机器 checkout 出行尾不同"不至于误报，内容仍逐字节比。
    """
    return data.replace(b"\r\n", b"\n")


class TestMotFormatParsing:
    def test_gt_filters_mark_and_class_and_applies_default_offset(self, tmp_path):
        gt_file = tmp_path / "gt.txt"
        gt_file.write_text(
            "1,1,10,10,50,100,1,1,0.9\n"  # mark=1 class=1 -> 保留，frame 0
            "1,2,10,10,50,100,0,1,0.9\n"  # mark=0 -> 丢
            "2,3,10,10,50,100,1,2,0.9\n"  # class=2 -> 丢
            "3,4,5,5,20,30,1,1,0.3\n",  # 保留：默认不限可见度
            encoding="utf-8",
        )
        frames = load_mot_gt(gt_file)
        assert sorted(frames) == [0, 2]
        assert [b.track_id for b in frames[0]] == [1]
        assert frames[2][0].bbox == (5.0, 5.0, 25.0, 35.0)  # xywh -> xyxy

    def test_gt_min_visibility_and_frame_offset(self, tmp_path):
        gt_file = tmp_path / "gt.txt"
        gt_file.write_text("1,1,0,0,10,20,1,1,0.2\n1,2,0,0,10,20,1,1,0.9\n", encoding="utf-8")
        frames = load_mot_gt(gt_file, min_visibility=0.5, frame_offset=0)
        assert list(frames) == [1]
        assert [b.track_id for b in frames[1]] == [2]

    def test_six_column_gt_keeps_every_row(self, tmp_path):
        gt_file = tmp_path / "gt.txt"
        gt_file.write_text("1,1,0,0,10,20\n1,2,30,40,10,20\n", encoding="utf-8")
        frames = load_mot_gt(gt_file)
        assert len(frames[0]) == 2

    def test_prediction_confidence_filter(self, tmp_path):
        pred_file = tmp_path / "pred.txt"
        pred_file.write_text(
            "1,5,0,0,10,20,0.9,-1,-1,-1\n"
            "1,6,50,50,10,20,0.1,-1,-1,-1\n",
            encoding="utf-8",
        )
        kept = load_mot_predictions(pred_file, min_conf=0.5)
        assert [b.track_id for b in kept[1]] == [5]

    def test_duplicate_track_id_in_one_frame_rejected(self, tmp_path):
        pred_file = tmp_path / "pred.txt"
        pred_file.write_text(
            "1,5,0,0,10,20,0.9,-1,-1,-1\n1,5,50,50,10,20,0.9,-1,-1,-1\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="重复 track_id"):
            load_mot_predictions(pred_file)

    def test_jsonl_loader_reads_export_predictions_format(self, tmp_path):
        jsonl = tmp_path / "pred.jsonl"
        jsonl.write_text(
            '{"frame": 0, "track_id": 3, "entity_id": "Person-1", '
            '"bbox": [10.0, 20.0, 60.0, 120.0], "class": "person"}\n',
            encoding="utf-8",
        )
        frames = load_predictions_jsonl(jsonl)
        assert frames[0][0].track_id == 3
        assert frames[0][0].bbox == (10.0, 20.0, 60.0, 120.0)


class TestAdapterSelfCheck:
    """适配器自检：完美 GT 必须是满分。**不是真实评测结果。**"""

    def test_demo_scene_perfect_gt_gives_unit_scores(self, tmp_path):
        pred = make_synthetic_gt.demo_predictions(frame_count=40)
        result = make_synthetic_gt.self_check(pred, workdir=tmp_path)

        assert result["idf1"] == pytest.approx(1.0)
        assert result["mota"] == pytest.approx(1.0)
        assert result["hota"] == pytest.approx(1.0)
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["idsw"] == 0
        assert result["frames"] == 40
        assert (tmp_path / "perfect_gt.txt").exists()
        assert (tmp_path / "perfect_predictions.txt").exists()

    def test_demo_predictions_follow_sample_video_scene(self):
        frames = make_synthetic_gt.demo_predictions()
        assert len(frames) == make_synthetic_gt.DEMO_FRAMES
        assert {b.track_id for boxes in frames.values() for b in boxes} == {1, 2, 3, 4}
        # P4 走出画面右缘的那段（约 2.0–5.5 秒）不应该有框
        assert all(4 not in {b.track_id for b in boxes} for boxes in (frames[80], frames[100]))

    def test_file_roundtrip_matches_in_memory_result(self, tmp_path):
        pred = make_synthetic_gt.demo_predictions(frame_count=20)
        gt_path = make_synthetic_gt.write_mot_gt(
            tmp_path / "gt.txt", make_synthetic_gt.perfect_gt_from_predictions(pred)
        )
        pred_path = make_synthetic_gt.write_mot_predictions(tmp_path / "pred.txt", pred)
        result = evaluate_files(gt_path, pred_path)
        assert result["idf1"] == pytest.approx(1.0)
        assert result["mota"] == pytest.approx(1.0)
        assert result["fp"] == 0 and result["fn"] == 0


class TestKnownErrorPatterns:
    def test_id_switch_lowers_idf1_and_counts_switch(self):
        gt = {frame: [_box(frame, 1)] for frame in range(10)}
        pred = {frame: [_box(frame, 1 if frame < 5 else 2)] for frame in range(10)}
        result = evaluate(gt, pred)

        assert result["idsw"] >= 1
        assert result["idf1"] < 1.0
        assert result["mota"] < 1.0
        assert result["hota"] < 1.0  # AssA 会被 ID 跳变拉低

    def test_false_positive_and_miss_counted(self):
        gt = {0: [_box(0, 1)]}
        pred = {0: [_box(0, 9, (500, 500, 550, 600))]}
        result = evaluate(gt, pred)

        assert result["fp"] == 1
        assert result["fn"] == 1
        assert result["idf1"] == 0.0
        # 1 个 GT 框上 1 误检 + 1 漏检：MOTA = 1 - (fp + fn) / gt = -1.0
        assert result["mota"] == -1.0

    def test_prediction_only_frame_counts_as_false_positive(self):
        gt = {0: [_box(0, 1)]}
        pred = {0: [_box(0, 1)], 2: [_box(2, 1)]}
        result = evaluate(gt, pred)

        assert result["frames"] == 2
        assert result["fp"] == 1
        assert result["fn"] == 0
        assert result["mota"] < 1.0

    def test_no_gt_or_no_prediction_returns_none_instead_of_inf(self):
        box = _box(0, 1)
        only_gt = evaluate({0: [box]}, {})
        only_pred = evaluate({}, {0: [box]})
        empty = evaluate({}, {})

        assert (only_gt["fn"], only_gt["mota"]) == (1, 0.0)
        assert only_pred["fp"] == 1
        assert only_pred["mota"] is None  # GT 为 0 时 MOTA 是 -inf，不允许写进 JSON
        assert empty["idf1"] is None and empty["mota"] is None
        assert empty["frames"] == 0

    def test_iou_threshold_controls_matching(self):
        gt = {0: [_box(0, 1, (0, 0, 100, 100))]}
        pred = {0: [_box(0, 1, (20, 0, 120, 100))]}  # IoU = 0.667（80/120）
        matched = evaluate(gt, pred, iou_threshold=0.5)
        unmatched = evaluate(gt, pred, iou_threshold=0.7)
        assert matched["mota"] == 1.0
        # 阈值卡掉之后同一帧变成 1 漏检 + 1 误检
        assert (unmatched["fn"], unmatched["fp"]) == (1, 1)
        assert unmatched["mota"] == -1.0


class TestReportWiring:
    def test_run_evaluation_attaches_mot_standard(self, tmp_path):
        gt_file = tmp_path / "gt.txt"
        gt_file.write_text(
            "".join(f"{i + 1},1,10,10,50,100,1,1,0.9\n" for i in range(4)),
            encoding="utf-8",
        )
        pred_file = tmp_path / "pred.jsonl"
        pred_file.write_text(
            "".join(
                f'{{"frame": {i}, "track_id": 1, "entity_id": "Person-1", '
                f'"bbox": [10.0, 10.0, 60.0, 110.0], "class": "person"}}\n'
                for i in range(4)
            ),
            encoding="utf-8",
        )
        report = run_evaluation(
            gt_path=gt_file,
            predictions_path=pred_file,
            video_path="video.mp4",
            frame_count=4,
            image_size=(1920, 1080),
            fps=25.0,
            mot_gt_path=gt_file,
        )
        mot = report["mot_standard"]
        assert mot["idf1"] == pytest.approx(1.0)
        assert mot["mota"] == pytest.approx(1.0)
        assert mot["frames"] == 4

    def test_run_gate_parser_exposes_mot_gt(self):
        from scripts.run_gate import build_parser

        args = build_parser().parse_args(["--video", "v.mp4", "--out", "o.json", "--mot-gt", "gt.txt"])
        assert args.mot_gt == "gt.txt"
        assert args.mot_iou == 0.5


class TestDemoSequence:
    """200 帧 / 6 目标合成序列上的指标**判别力**测试。

    和上面的满分自检互补：``TestAdapterSelfCheck`` 只证明链路通（完美 GT 恒为
    1.0，指标写死成 1.0 也能过），这里用同一份 ``gt.txt`` 配三种**有错误**的
    预测，要求指标按缺陷类型分开。因此下面任何一条失败，都说明适配器算错了，
    而不只是"数据变了"。

    数字锚点与 ``evaluation/demo_data/expected.json`` 一致；改生成器要一并
    重跑 ``python -m evaluation.make_demo_sequence``。
    """

    @pytest.fixture(scope="class")
    def scenarios(self):
        """三个场景各跑一遍文件往返（与用户命令行同一条路径），类内只算一次。"""
        demo_dir = make_demo_sequence.DEMO_DATA_DIR
        paths = {"gt": demo_dir / "gt.txt"}
        for kind in make_demo_sequence.SCENARIOS:
            paths[kind] = demo_dir / f"{kind}.txt"
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            pytest.skip(f"demo_data 缺失，先跑 make_demo_sequence：{missing}")
        return make_demo_sequence.evaluate_demo(paths)

    def test_demo_files_are_committed(self):
        for name in ("gt.txt", "perfect.txt", "idsw.txt", "dropout.txt", "expected.json"):
            assert (make_demo_sequence.DEMO_DATA_DIR / name).is_file(), name
        # 用户要求体积控制在 1MB 内
        total = sum(
            p.stat().st_size for p in make_demo_sequence.DEMO_DATA_DIR.glob("*.txt")
        )
        assert total < 1024 * 1024

    def test_demo_data_is_reproducible(self, tmp_path):
        """重新生成必须与仓库里的文件一致——防止改了生成器忘了落数据。"""
        make_demo_sequence.write_demo_data(tmp_path)
        demo_dir = make_demo_sequence.DEMO_DATA_DIR
        for name in ("gt.txt", "perfect.txt", "idsw.txt", "dropout.txt"):
            assert _lf((tmp_path / name).read_bytes()) == _lf(
                (demo_dir / name).read_bytes()
            ), f"{name} 与生成器输出不一致，重跑 make_demo_sequence"

    def test_perfect_scenario_scores_one(self, scenarios):
        perfect = scenarios["perfect"]
        assert perfect["idf1"] == pytest.approx(1.0, abs=1e-6)
        assert perfect["mota"] == pytest.approx(1.0, abs=1e-6)
        assert (perfect["fp"], perfect["fn"], perfect["idsw"]) == (0, 0, 0)
        assert perfect["gt_detections"] == 200 * len(make_demo_sequence.TARGETS)

    def test_idsw_lowers_idf1_and_counts_switches(self, scenarios):
        perfect, idsw = scenarios["perfect"], scenarios["idsw"]

        # 用户要的核心关系
        assert idsw["idf1"] < perfect["idf1"]
        assert idsw["idsw"] > 0
        # 换 ID 不产生误检/漏检，只伤关联：IDP == IDR，FP == FN == 0
        assert (idsw["fp"], idsw["fn"]) == (0, 0)
        assert idsw["idp"] == pytest.approx(idsw["idr"])
        # 2、4 号各换 1 次 + 6 号换 2 次 = 4 次
        assert idsw["idsw"] == 4
        # ID 全换掉的话 IDF1 会掉到 0.825；再高就不对劲了
        assert idsw["idf1"] < 0.9
        # 关联错误也要压低 HOTA 的 AssA
        assert idsw["assa"] < perfect["assa"]

    def test_dropout_raises_fn_and_lowers_recall(self, scenarios):
        perfect, dropout = scenarios["perfect"], scenarios["dropout"]

        # 用户要的核心关系
        assert dropout["fn"] > perfect["fn"]
        # 1 号漏 30 帧 + 5 号漏 50 帧 + 3 号每 7 帧漏 1 帧（29 帧）= 109
        assert dropout["fn"] == 109
        assert dropout["fp"] == 0  # 只漏不误
        assert dropout["idsw"] == 0  # 没有换 ID
        assert dropout["idr"] < perfect["idr"]
        assert dropout["mota"] < perfect["mota"]
        # MOTA = 1 - FN/GT 在这个无 FP 无 IDSW 的场景里成立
        assert dropout["mota"] == pytest.approx(1.0 - dropout["fn"] / dropout["gt_detections"], abs=1e-3)

    def test_three_scenarios_are_distinguishable(self, scenarios):
        """三种预测必须给出三组可区分的数字，否则指标没有判别力。"""
        perfect = scenarios["perfect"]
        assert perfect["idf1"] > scenarios["idsw"]["idf1"]
        assert perfect["idf1"] >= scenarios["dropout"]["idf1"]
        assert scenarios["idsw"]["idsw"] > 0
        assert scenarios["dropout"]["fn"] > 0
        assert scenarios["idsw"] != scenarios["dropout"]

    def test_expected_json_agrees_with_live_run(self, scenarios):
        """提交的 expected.json 必须和现在算出来的一致（浮点留 0.02 余量）。"""
        recorded = json.loads(
            (make_demo_sequence.DEMO_DATA_DIR / "expected.json").read_text(encoding="utf-8")
        )
        assert recorded["scenarios"].keys() == set(make_demo_sequence.SCENARIOS)

        for kind in make_demo_sequence.SCENARIOS:
            live = scenarios[kind]
            saved = recorded["scenarios"][kind]
            for key in make_demo_sequence.METRIC_KEYS:
                if live.get(key) is None or saved.get(key) is None:
                    continue  # 后端没装时字段为 None，不参与比对
                assert live[key] == pytest.approx(saved[key], abs=0.02), f"{kind}.{key}"

    def test_demo_gt_and_predictions_align_by_default_offsets(self):
        """文件按 1 基 GT / 0 基预测写，必须直接喂给 evaluate_files 的默认偏移。"""
        demo_dir = make_demo_sequence.DEMO_DATA_DIR
        result = evaluate_files(demo_dir / "gt.txt", demo_dir / "perfect.txt")
        assert result["frames"] == make_demo_sequence.N_FRAMES
        assert result["idf1"] == pytest.approx(1.0)
