from perception.entity_profile import EntityProfileManager


def track(track_id, frame, x, y, feature=None, direction=0.0):
    return {
        "track_id": track_id,
        "bbox": [x - 5, y - 10, x + 5, y + 10],
        "centroid": [x, y],
        "trajectory": [[x, y, frame / 30.0]],
        "velocity_ms": 1.0,
        "direction_deg": direction,
        "appearance_feature": feature or [1.0, 0.0, 0.0],
    }


def test_stable_track_maps_to_same_person():
    mgr = EntityProfileManager(auto_merge_threshold=0.8)
    enriched = mgr.update([track(3, 10, 100, 100)], frame_idx=10, timestamp=10 / 30.0)
    assert enriched[0]["entity_id"] == "Person-1"

    enriched = mgr.update([track(3, 11, 102, 101)], frame_idx=11, timestamp=11 / 30.0)
    assert enriched[0]["entity_id"] == "Person-1"
    assert mgr.get_profile("Person-1").track_ids == [3]


def test_high_confidence_new_track_merges_after_occlusion_gap():
    mgr = EntityProfileManager(auto_merge_threshold=0.65)
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)
    enriched = mgr.update([track(17, 18, 106, 102, [0.98, 0.02])], frame_idx=18, timestamp=18 / 30.0)

    assert enriched[0]["entity_id"] == "Person-1"
    profile = mgr.get_profile("Person-1")
    assert profile.track_ids == [3, 17]
    assert profile.occlusion_gaps[0]["from_frame"] == 10
    assert profile.occlusion_gaps[0]["to_frame"] == 18
    assert profile.merge_evidence[0]["track_b"] == 17


def test_close_short_gap_track_merges_without_appearance_feature_under_defaults():
    mgr = EntityProfileManager()
    first = track(3, 10, 100, 100)
    second = track(17, 18, 104, 102)
    first.pop("appearance_feature")
    second.pop("appearance_feature")

    mgr.update([first], frame_idx=10, timestamp=10 / 30.0)
    enriched = mgr.update([second], frame_idx=18, timestamp=18 / 30.0)

    assert enriched[0]["entity_id"] == "Person-1"
    assert mgr.get_profile("Person-1").track_ids == [3, 17]


def test_low_confidence_new_track_becomes_tentative_candidate():
    mgr = EntityProfileManager(auto_merge_threshold=0.8, tentative_threshold=0.35)
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)
    enriched = mgr.update([track(17, 18, 260, 240, [0.2, 0.98])], frame_idx=18, timestamp=18 / 30.0)

    assert enriched[0]["entity_id"] == "Person-2"
    assert mgr.get_profile("Person-2").tentative_merge_candidates == ["Person-1"]


def test_distinct_tracks_outside_tentative_path_get_sequential_person_ids():
    mgr = EntityProfileManager(auto_merge_threshold=0.8, tentative_threshold=0.35)
    first = mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)
    second = mgr.update([track(17, 18, 500, 500, [-1.0, 0.0])], frame_idx=18, timestamp=18 / 30.0)

    assert first[0]["entity_id"] == "Person-1"
    assert second[0]["entity_id"] == "Person-2"
    assert mgr.get_profile("Person-2").tentative_merge_candidates == []


def test_same_frame_new_tracks_do_not_merge_with_each_other():
    mgr = EntityProfileManager(auto_merge_threshold=0.65)
    enriched = mgr.update(
        [
            track(3, 10, 100, 100, [1.0, 0.0]),
            track(17, 10, 102, 101, [0.99, 0.01]),
        ],
        frame_idx=10,
        timestamp=10 / 30.0,
    )

    assert [obj["entity_id"] for obj in enriched] == ["Person-1", "Person-2"]
    assert mgr.get_profile("Person-1").track_ids == [3]
    assert mgr.get_profile("Person-2").track_ids == [17]


def test_same_frame_new_tracks_in_separate_updates_do_not_merge():
    mgr = EntityProfileManager(auto_merge_threshold=0.65)
    first = mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)
    second = mgr.update([track(17, 10, 102, 101, [0.99, 0.01])], frame_idx=10, timestamp=10 / 30.0)

    assert first[0]["entity_id"] == "Person-1"
    assert second[0]["entity_id"] == "Person-2"
    assert mgr.get_profile("Person-1").track_ids == [3]
    assert mgr.get_profile("Person-2").track_ids == [17]


def test_update_returns_enriched_copies_without_mutating_input():
    mgr = EntityProfileManager()
    tracked = [track(3, 10, 100, 100)]
    original = dict(tracked[0])

    enriched = mgr.update(tracked, frame_idx=10, timestamp=10 / 30.0)

    assert tracked[0] == original
    assert enriched[0] is not tracked[0]
    assert enriched[0] == {
        **original,
        "entity_id": "Person-1",
        "tracking_quality": "stable",
    }


def test_tracking_quality_uses_allowed_enum_for_tentative_and_merged_profiles():
    allowed = {"stable", "occluded", "fragmented", "tentative"}
    tentative_mgr = EntityProfileManager(auto_merge_threshold=0.8, tentative_threshold=0.35)
    tentative_mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)
    tentative = tentative_mgr.update([track(17, 18, 260, 240, [0.2, 0.98])], frame_idx=18, timestamp=18 / 30.0)

    assert tentative[0]["tracking_quality"] == "tentative"
    assert tentative_mgr.get_profile("Person-2").tracking_quality == "tentative"

    merged_mgr = EntityProfileManager(auto_merge_threshold=0.65)
    merged_mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)
    merged = merged_mgr.update([track(17, 18, 106, 102, [0.98, 0.02])], frame_idx=18, timestamp=18 / 30.0)

    assert merged[0]["tracking_quality"] in {"occluded", "fragmented"}
    assert merged_mgr.get_profile("Person-1").tracking_quality in {"occluded", "fragmented"}
    assert {p.tracking_quality for p in tentative_mgr.profiles()}.issubset(allowed)
    assert {p.tracking_quality for p in merged_mgr.profiles()}.issubset(allowed)


def test_lifeline_export_uses_frame_segments_and_quality():
    mgr = EntityProfileManager(auto_merge_threshold=0.65)
    mgr.update([track(3, 10, 100, 100)], frame_idx=10, timestamp=10 / 30.0)
    mgr.update([track(3, 20, 120, 110)], frame_idx=20, timestamp=20 / 30.0)

    lifelines = mgr.to_lifelines()
    assert lifelines == [{
        "entity_id": "Person-1",
        "segments": [{"track_id": 3, "frame_start": 10, "frame_end": 20, "quality": "stable"}],
        "occlusion_gaps": [],
    }]


def test_lifeline_export_after_merge_has_multiple_segments_and_gap():
    mgr = EntityProfileManager(auto_merge_threshold=0.65)
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)
    mgr.update([track(3, 12, 102, 101, [1.0, 0.0])], frame_idx=12, timestamp=12 / 30.0)
    mgr.update([track(17, 18, 106, 102, [0.98, 0.02])], frame_idx=18, timestamp=18 / 30.0)
    mgr.update([track(17, 20, 108, 103, [0.98, 0.02])], frame_idx=20, timestamp=20 / 30.0)

    lifelines = mgr.to_lifelines()
    assert lifelines == [{
        "entity_id": "Person-1",
        "segments": [
            {"track_id": 3, "frame_start": 10, "frame_end": 12, "quality": "stable"},
            {"track_id": 17, "frame_start": 18, "frame_end": 20, "quality": "stable"},
        ],
        "occlusion_gaps": [{
            "from_frame": 12,
            "to_frame": 18,
            "reason": "track_reappeared",
            "reid_score": mgr.get_profile("Person-1").merge_evidence[0]["score"],
        }],
    }]


def test_to_entities_returns_summary_only_without_full_global_trajectory():
    mgr = EntityProfileManager()
    mgr.update([track(3, 10, 100, 100)], frame_idx=10, timestamp=10 / 30.0)

    entities = mgr.to_entities()

    assert entities[0]["entity_id"] == "Person-1"
    assert "trajectory_summary" in entities[0]
    assert "global_trajectory" not in entities[0]


def test_exports_deep_copy_nested_evidence_and_gap_data():
    mgr = EntityProfileManager(auto_merge_threshold=0.65)
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)
    mgr.update([track(17, 18, 106, 102, [0.98, 0.02])], frame_idx=18, timestamp=18 / 30.0)

    entities = mgr.to_entities()
    lifelines = mgr.to_lifelines()
    entities[0]["merge_evidence_summary"][0]["track_b"] = 999
    lifelines[0]["occlusion_gaps"][0]["to_frame"] = 999

    profile = mgr.get_profile("Person-1")
    assert profile.merge_evidence[0]["track_b"] == 17
    assert profile.occlusion_gaps[0]["to_frame"] == 18


def test_debug_summary_records_candidate_scores_and_blockers():
    mgr = EntityProfileManager(enable_debug=True, auto_merge_threshold=0.99, tentative_threshold=0.3)
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)
    mgr.update([track(17, 18, 106, 102, [0.98, 0.02])], frame_idx=18, timestamp=18 / 30.0)

    summary = mgr.debug_summary()

    assert summary["enabled"] is True
    assert summary["stats"]["tentative"] == 1
    assert summary["blockers"]["score_below_auto_merge_threshold"] == 1
    assert summary["records"][1]["track_id"] == 17
    assert summary["records"][1]["top_candidates"][0]["entity_id"] == "Person-1"
    assert "position_score" in summary["records"][1]["top_candidates"][0]["factors"]


def test_tentative_candidate_promotes_after_sustained_match():
    mgr = EntityProfileManager(
        auto_merge_threshold=0.99,
        tentative_threshold=0.3,
        promotion_min_observations=3,
        promotion_min_score=0.5,
        promotion_min_position_score=0.2,
        promotion_min_direction_score=0.7,
        promotion_min_appearance_score=0.7,
        enable_debug=True,
    )
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0], direction=12)], frame_idx=10, timestamp=10 / 30.0)
    first = mgr.update([track(17, 18, 106, 102, [0.98, 0.02], direction=14)], frame_idx=18, timestamp=18 / 30.0)
    second = mgr.update([track(17, 19, 108, 103, [0.98, 0.02], direction=13)], frame_idx=19, timestamp=19 / 30.0)
    third = mgr.update([track(17, 20, 110, 104, [0.98, 0.02], direction=12)], frame_idx=20, timestamp=20 / 30.0)

    assert first[0]["entity_id"] == "Person-2"
    assert second[0]["entity_id"] == "Person-2"
    assert third[0]["entity_id"] == "Person-1"
    assert "Person-2" not in {profile.entity_id for profile in mgr.profiles()}
    assert mgr.get_profile("Person-1").track_ids == [3, 17]
    assert mgr.get_profile("Person-1").merge_evidence[-1]["promotion"] == "tentative_sustained_match"
    assert mgr.debug_summary()["stats"]["promoted_merge"] == 1


def test_tentative_candidate_does_not_promote_when_direction_is_inconsistent():
    mgr = EntityProfileManager(
        auto_merge_threshold=0.99,
        tentative_threshold=0.3,
        promotion_min_observations=3,
        promotion_min_score=0.5,
        promotion_min_position_score=0.2,
        promotion_min_direction_score=0.7,
        promotion_min_appearance_score=0.7,
    )
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0], direction=0)], frame_idx=10, timestamp=10 / 30.0)
    for frame in [18, 19, 20, 21]:
        enriched = mgr.update(
            [track(17, frame, 106 + frame - 18, 102, [0.98, 0.02], direction=180)],
            frame_idx=frame,
            timestamp=frame / 30.0,
        )

    assert enriched[0]["entity_id"] == "Person-2"
    assert mgr.get_profile("Person-2").tentative_merge_candidates == ["Person-1"]


def test_tentative_candidate_does_not_promote_when_position_is_far():
    mgr = EntityProfileManager(
        auto_merge_threshold=0.99,
        tentative_threshold=0.3,
        max_position_gap_px=1000.0,
        promotion_min_observations=3,
        promotion_min_score=0.5,
        promotion_min_position_score=0.5,
        promotion_min_direction_score=0.7,
        promotion_min_appearance_score=0.7,
    )
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0], direction=0)], frame_idx=10, timestamp=10 / 30.0)
    for frame in [18, 19, 20, 21]:
        enriched = mgr.update(
            [track(17, frame, 850, 100, [0.98, 0.02], direction=0)],
            frame_idx=frame,
            timestamp=frame / 30.0,
        )

    assert enriched[0]["entity_id"] == "Person-2"
    assert mgr.get_profile("Person-2").tentative_merge_candidates == ["Person-1"]


def test_moving_person_reappearance_uses_predicted_position_for_conservative_merge():
    mgr = EntityProfileManager(auto_merge_threshold=0.72, max_position_gap_px=40.0)
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0], direction=0)], frame_idx=10, timestamp=10 / 30.0)
    mgr.update([track(3, 11, 110, 100, [1.0, 0.0], direction=0)], frame_idx=11, timestamp=11 / 30.0)

    enriched = mgr.update(
        [track(17, 16, 160, 100, [0.99, 0.01], direction=0)],
        frame_idx=16,
        timestamp=16 / 30.0,
    )

    assert enriched[0]["entity_id"] == "Person-1"
    assert mgr.get_profile("Person-1").track_ids == [3, 17]
    assert mgr.get_profile("Person-1").merge_evidence[-1]["factors"]["predicted_distance_px"] == 0.0


def test_appearance_only_does_not_auto_merge_when_spatial_gap_is_implausible():
    mgr = EntityProfileManager(auto_merge_threshold=0.60, tentative_threshold=0.30, max_position_gap_px=100.0)
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)

    enriched = mgr.update(
        [track(17, 11, 1000, 1000, [1.0, 0.0])],
        frame_idx=11,
        timestamp=11 / 30.0,
    )

    assert enriched[0]["entity_id"] == "Person-2"
    assert mgr.get_profile("Person-1").track_ids == [3]


def test_tentative_entity_output_exposes_predecessor_for_ui():
    mgr = EntityProfileManager(auto_merge_threshold=0.8, tentative_threshold=0.35)
    mgr.update([track(3, 10, 100, 100, [1.0, 0.0])], frame_idx=10, timestamp=10 / 30.0)

    enriched = mgr.update(
        [track(17, 18, 210, 180, [0.7, 0.3])],
        frame_idx=18,
        timestamp=18 / 30.0,
    )

    assert enriched[0]["entity_id"] == "Person-2"
    assert enriched[0]["tentative_predecessor"] == "Person-1"
