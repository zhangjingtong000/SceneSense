"""AnomalyAgent：关键词模板打分，只给 raw_score，不调模型。"""

from agents.anomaly_agent import AnomalyAgent


class ExplodingLLM:
    def chat_with_json_output(self, *args, **kwargs):
        raise AssertionError("AnomalyAgent 不应调用 LLM")

    def chat(self, *args, **kwargs):
        raise AssertionError("AnomalyAgent 不应调用 LLM")

    def generate(self, *args, **kwargs):
        raise AssertionError("AnomalyAgent 不应调用 LLM")


def running_context():
    return {
        "tracklets": [
            {
                "track_id": 1,
                "class": "person",
                "velocity_ms": 200.0,
                "direction_deg": 0.0,
                "centroid": [10.0, 10.0],
            }
        ],
        "all_agent_responses": {
            "trajectory_agent": {
                "findings": [
                    {
                        "entity_id": "Entity-1",
                        "finding_type": "motion_pattern",
                        "value": {"label": "奔跑", "speed_level": "奔跑"},
                        "confidence": 0.9,
                    }
                ]
            },
            "interaction_agent": {"findings": []},
        },
    }


def test_fast_entity_matches_running_template_with_raw_score():
    agent = AnomalyAgent(ExplodingLLM())
    agent.context = running_context()

    response = agent.think()

    assert len(response["findings"]) == 1
    finding = response["findings"][0]
    assert finding["entity_id"] == "Entity-1"
    assert finding["finding_type"] == "anomaly"
    assert finding["value"] == "running"
    # 只输出匹配度，严重度留给协调层融合
    assert finding["confidence"] == response["agent_confidence"]
    assert 0.0 <= finding["confidence"] <= 1.0


def test_match_emits_kg_edge_to_behavior_node():
    agent = AnomalyAgent(ExplodingLLM())
    agent.context = running_context()

    ops = agent.think()["findings"][0]["kg_operations"]

    assert [op["edge_type"] for op in ops] == ["matches"]
    assert ops[0]["from"] == "Entity-1"
    assert ops[0]["to"].startswith("Behavior-")
    assert "confidence" in ops[0]["properties"]


def test_empty_context_returns_neutral_confidence_and_no_findings():
    agent = AnomalyAgent(ExplodingLLM())
    agent.context = {"tracklets": [], "all_agent_responses": {}}

    response = agent.think()

    assert response["findings"] == []
    assert response["agent_confidence"] == 0.5
    assert response["warnings"] == []


def test_entity_without_motion_findings_is_not_flagged():
    agent = AnomalyAgent(ExplodingLLM())
    agent.context = {
        "tracklets": [
            {"track_id": 2, "class": "person", "velocity_ms": 10.0, "centroid": [0.0, 0.0]}
        ],
        "all_agent_responses": {},
    }

    response = agent.think()

    assert response["findings"] == []
    assert response["agent_confidence"] == 0.5
