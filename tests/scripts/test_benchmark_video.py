from scripts.benchmark_video import (
    SCHEMA_VERSION,
    build_parser,
    format_value,
    parse_classes,
    print_summary,
    summarize_event,
)


def test_parse_classes_supports_all_or_comma_list():
    assert parse_classes("") is None
    assert parse_classes("0") == [0]
    assert parse_classes("0, 2") == [0, 2]


def test_summarize_event_has_stable_fields():
    event = {
        "event_id": "Event-1",
        "frame_start": 0,
        "frame_end": 60,
        "duration": [0.0, 2.4],
        "tracklets": [{"track_id": 1}],
        "spatial_relations": [{"track_1": 1, "track_2": 2}],
    }

    assert summarize_event(event) == {
        "event_id": "Event-1",
        "frame_start": 0,
        "frame_end": 60,
        "duration": [0.0, 2.4],
        "tracklet_count": 1,
        "spatial_relation_count": 1,
    }


def test_print_summary_outputs_terminal_tables(capsys):
    result = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "source": "test.mp4",
            "elapsed_seconds": 1.25,
            "processed_fps": 100.0,
        },
        "video": {
            "frame_count_processed": 10,
            "frame_count_reported": 10,
            "fps": 25.0,
            "width": 1920,
            "height": 1080,
        },
        "parameters": {
            "device": "cuda",
            "detector": {"model": "yolov8n.pt", "conf": 0.15, "imgsz": 960, "classes": [0]},
            "tracker": {
                "lost_track_buffer": 120,
                "minimum_matching_threshold": 0.65,
                "with_reid": False,
                "with_gmc": True,
            },
            "entity": {
                "auto_merge_threshold": 0.78,
                "tentative_threshold": 0.3,
            },
        },
        "metrics": {
            "tracker": {
                "detection_total": 20,
                "tracked_total": 10,
                "tracker_drop_rate": 0.5,
                "detection_gap_rate": 0.0,
                "id_switch_count": 1,
            },
            "entities": {"entity_count": 3},
            "events": {"event_count": 1},
            "kg": {"edge_types": {"interacts_with": 2}},
        },
    }

    print_summary(result)
    output = capsys.readouterr().out

    assert "Benchmark Summary" in output
    assert "Parameters" in output
    assert "Metrics" in output
    assert "tracker_drop_rate" in output


def test_format_value_keeps_diff_friendly_scalars():
    assert format_value(0.5) == "0.5"
    assert format_value([0, 2]) == "0,2"
    assert format_value(None) == "all"


def test_parser_supports_optional_entity_debug():
    args = build_parser().parse_args([
        "--source", "video.mp4",
        "--entity-debug",
        "--entity-auto-threshold", "0.72",
    ])

    assert args.entity_debug is True
    assert args.entity_auto_threshold == 0.72
