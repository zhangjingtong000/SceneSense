from perception.metrics import TrackerMetrics


def test_tracker_metrics_counts_detection_drop_and_gap_rates():
    metrics = TrackerMetrics()

    metrics.observe(
        detections=[{"bbox": [0, 0, 10, 10]}, {"bbox": [20, 0, 30, 10]}],
        tracked_objects=[{"track_id": 1, "entity_id": "Person-1"}],
        entities=[{"entity_id": "Person-1", "track_ids": [1]}],
    )
    metrics.observe(
        detections=[{"bbox": [0, 0, 10, 10]}],
        tracked_objects=[],
        entities=[{"entity_id": "Person-1", "track_ids": [1]}],
    )

    result = metrics.to_dict()

    assert result["frame_count"] == 2
    assert result["detection_total"] == 3
    assert result["tracked_total"] == 1
    assert result["tracker_drop_count"] == 2
    assert result["tracker_drop_rate"] == 0.6667
    assert result["frames_with_detections_no_tracks"] == 1
    assert result["detection_gap_rate"] == 0.5


def test_tracker_metrics_estimates_raw_id_switches_from_entity_track_ids():
    metrics = TrackerMetrics()

    metrics.observe(
        detections=[{}],
        tracked_objects=[{"track_id": 3, "entity_id": "Person-1"}],
        entities=[{"entity_id": "Person-1", "track_ids": [3]}],
    )
    metrics.observe(
        detections=[{}],
        tracked_objects=[{"track_id": 17, "entity_id": "Person-1"}],
        entities=[{"entity_id": "Person-1", "track_ids": [3, 17]}],
    )

    result = metrics.to_dict()

    assert result["entity_count"] == 1
    assert result["raw_track_count"] == 2
    assert result["id_switch_count"] == 1
    assert result["id_switch_per_entity"] == 1.0


def test_tracker_metrics_acceptance_anchor_can_compare_to_baseline():
    metrics = TrackerMetrics()
    metrics.observe(
        detections=[{}, {}],
        tracked_objects=[{"track_id": 1, "entity_id": "Person-1"}],
        entities=[{"entity_id": "Person-1", "track_ids": [1, 2]}],
    )

    anchor = metrics.acceptance_anchor({
        "tracker_drop_rate": 0.75,
        "detection_gap_rate": 0.1,
        "id_switch_count": 2,
        "id_switch_per_entity": 2.0,
    })

    assert "绝对值或相对变化" in anchor["requirement"]
    assert anchor["comparison"]["tracker_drop_rate"]["absolute_change"] == -0.25
    assert anchor["comparison"]["id_switch_count"]["absolute_change"] == -1.0
