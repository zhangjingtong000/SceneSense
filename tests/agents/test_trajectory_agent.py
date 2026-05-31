from agents.trajectory_agent import TrajectoryAgent, VelocityNormalizer


def make_tracklet(
    track_id=1,
    entity_id="Person-1",
    velocity=60.0,
    bbox=None,
    trajectory=None,
):
    return {
        "track_id": track_id,
        "entity_id": entity_id,
        "bbox": bbox or [0, 0, 30, 100],
        "velocity_ms": velocity,
        "trajectory": trajectory or [
            [0, 0, 0.0],
            [5, 0, 0.1],
            [10, 0, 0.2],
            [15, 0, 0.3],
            [20, 0, 0.4],
        ],
        "direction_deg": 0.0,
    }


def make_agent():
    agent = TrajectoryAgent.__new__(TrajectoryAgent)
    agent.normalizer = VelocityNormalizer()
    return agent


def test_equal_relative_motion_at_different_bbox_heights_has_equal_normalized_speed():
    agent = make_agent()

    near = agent._describe_motion(make_tracklet(velocity=100.0, bbox=[0, 0, 40, 200]))
    far = agent._describe_motion(make_tracklet(velocity=50.0, bbox=[0, 0, 20, 100]))

    assert near["normalized_speed"] == 0.5
    assert far["normalized_speed"] == 0.5


def test_high_normalized_speed_is_not_overwritten_by_small_net_displacement():
    agent = make_agent()
    agent.normalizer.observe_batch([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    tracklet = make_tracklet(
        velocity=180.0,
        bbox=[0, 0, 30, 100],
        trajectory=[
            [0, 0, 0.0],
            [8, 0, 0.1],
            [0, 0, 0.2],
            [8, 0, 0.3],
            [0, 0, 0.4],
        ],
    )

    result = agent._describe_motion(tracklet)

    assert result["normalized_speed"] == 1.8
    assert result["speed_level"] in {"快速", "奔跑"}
    assert result["label"] in {"快速", "奔跑"}


def test_trajectory_agent_findings_use_existing_person_id():
    agent = make_agent()
    agent.context = {
        "tracklets": [make_tracklet(track_id=17, entity_id="Person-3")],
        "scene_type": "full_frame",
    }

    response = agent.think()

    assert response["findings"][0]["entity_id"] == "Person-3"
    assert response["findings"][0]["kg_operations"][0]["node_id"] == "Person-3"
