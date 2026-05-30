import time

from kg.continuity import write_continuity_event
from kg.graph_store import GraphStore


def make_event(event_id="Event-1", frame_start=0, frame_end=60):
    return {
        "event_id": event_id,
        "timestamp": 1.0,
        "duration": [0.0, 2.0],
        "frame_start": frame_start,
        "frame_end": frame_end,
        "tracklets": [
            {"track_id": 3, "entity_id": "Person-1", "centroid": [10, 20]},
            {"track_id": 4, "entity_id": "Person-2", "centroid": [30, 20]},
        ],
        "spatial_relations": [
            {
                "track_1": 3,
                "track_2": 4,
                "distance_m": 20.0,
                "relation_type": "near",
                "duration_s": 1.5,
            }
        ],
        "zone_id": "zone_default",
        "zone_type": "full_frame",
    }


def make_profiles():
    return [
        {
            "entity_id": "Person-1",
            "track_ids": [3],
            "first_seen_frame": 0,
            "last_seen_frame": 60,
            "tracking_quality": "stable",
            "trajectory_summary": {"point_count": 2, "activity_bbox": [0, 0, 10, 10]},
            "merge_evidence_summary": [],
        },
        {
            "entity_id": "Person-2",
            "track_ids": [4],
            "first_seen_frame": 0,
            "last_seen_frame": 60,
            "tracking_quality": "stable",
            "trajectory_summary": {"point_count": 2, "activity_bbox": [20, 0, 30, 10]},
            "merge_evidence_summary": [],
        },
    ]


def test_write_event_entity_and_edges():
    graphstore = GraphStore()

    before = time.time()
    write_continuity_event(graphstore, make_event(), make_profiles(), ttl_seconds=30.0)
    after = time.time()

    event_node = graphstore.get_node("Event-1")
    assert event_node["node_type"] == "Event"
    assert event_node["timestamp"] == 1.0
    assert event_node["duration"] == "[0.0, 2.0]"
    assert event_node["frame_start"] == 0
    assert event_node["frame_end"] == 60
    assert event_node["zone_id"] == "zone_default"
    assert event_node["zone_type"] == "full_frame"
    assert event_node["entity_count"] == 2
    assert before + 30.0 <= event_node["expires_at"] <= after + 30.0 + 0.1

    entity_node = graphstore.get_node("Person-1")
    assert entity_node["node_type"] == "Entity"
    assert entity_node["entity_id"] == "Person-1"
    assert entity_node["track_ids"] == "[3]"
    assert GraphStore.decode_property(entity_node["trajectory_summary"]) == {
        "point_count": 2,
        "activity_bbox": [0, 0, 10, 10],
    }

    belongs = graphstore.get_edge("Person-1", "Event-1")
    assert belongs == {
        "src": "Person-1",
        "tgt": "Event-1",
        "edge_type": "belongs_to",
        "frame_start": 0,
        "frame_end": 60,
        "track_id": 3,
    }

    interaction = graphstore.get_edge("Person-1", "Person-2")
    assert interaction["edge_type"] == "interacts_with"
    assert GraphStore.decode_property(interaction["event_ids"]) == ["Event-1"]
    assert GraphStore.decode_property(interaction["frame_ranges"]) == [[0, 60]]
    assert interaction["duration"] == 1.5
    assert interaction["confidence"] == 0.2
    assert GraphStore.decode_property(interaction["evidence"])[0]["relation_type"] == "near"


def test_interaction_edge_aggregates_evidence_over_events_and_next_edge():
    graphstore = GraphStore()
    write_continuity_event(graphstore, make_event("Event-1", 0, 60), make_profiles())
    write_continuity_event(
        graphstore,
        make_event("Event-2", 61, 120),
        make_profiles(),
        previous_event_id="Event-1",
    )

    interaction = graphstore.get_edge("Person-1", "Person-2")
    assert GraphStore.decode_property(interaction["event_ids"]) == ["Event-1", "Event-2"]
    assert GraphStore.decode_property(interaction["frame_ranges"]) == [[0, 60], [61, 120]]
    assert interaction["duration"] == 3.0
    assert interaction["confidence"] == 0.4
    evidence = GraphStore.decode_property(interaction["evidence"])
    assert [item["event_id"] for item in evidence] == ["Event-1", "Event-2"]

    next_edge = graphstore.get_edge("Event-1", "Event-2")
    assert next_edge["edge_type"] == "next"
    assert next_edge["scene_continuity"] is True


def test_query_filters_compare_decoded_encoded_properties():
    graphstore = GraphStore()
    write_continuity_event(graphstore, make_event(), make_profiles())

    assert [node["node_id"] for node in graphstore.query_nodes(filters={"track_ids": [3]})] == ["Person-1"]
    assert [
        (edge["src"], edge["tgt"])
        for edge in graphstore.query_edges(filters={"event_ids": ["Event-1"]})
    ] == [("Person-1", "Person-2")]


def test_interaction_aggregation_is_idempotent_for_repeat_and_duplicate_relations():
    graphstore = GraphStore()
    event = make_event()
    event["spatial_relations"].append(
        {
            "track_1": 4,
            "track_2": 3,
            "distance_m": 20.0,
            "relation_type": "near",
            "duration_s": 1.5,
        }
    )

    write_continuity_event(graphstore, event, make_profiles())
    write_continuity_event(graphstore, event, make_profiles())

    interaction = graphstore.get_edge("Person-1", "Person-2")
    assert GraphStore.decode_property(interaction["event_ids"]) == ["Event-1"]
    assert GraphStore.decode_property(interaction["frame_ranges"]) == [[0, 60]]
    assert interaction["duration"] == 1.5
    evidence = GraphStore.decode_property(interaction["evidence"])
    assert len(evidence) == 1
    assert evidence[0]["track_pair"] == [3, 4]


def test_graph_store_save_load_round_trips_encoded_continuity_properties(tmp_path):
    path = tmp_path / "continuity.graphml"
    graphstore = GraphStore()
    write_continuity_event(graphstore, make_event(), make_profiles())

    graphstore.save(str(path))
    loaded = GraphStore()

    assert loaded.load(str(path)) is True
    assert GraphStore.decode_property(loaded.get_node("Person-1")["trajectory_summary"]) == {
        "point_count": 2,
        "activity_bbox": [0, 0, 10, 10],
    }
    assert GraphStore.decode_property(loaded.get_edge("Person-1", "Person-2")["event_ids"]) == ["Event-1"]


def test_entity_node_does_not_store_global_trajectory_when_profile_contains_one():
    graphstore = GraphStore()
    profiles = make_profiles()
    profiles[0]["global_trajectory"] = [[10, 20, 0.0], [11, 21, 1.0]]

    write_continuity_event(graphstore, make_event(), profiles)

    entity_node = graphstore.get_node("Person-1")
    assert "trajectory_summary" in entity_node
    assert "global_trajectory" not in entity_node
