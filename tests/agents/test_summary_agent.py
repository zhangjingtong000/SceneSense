"""SummaryAgent：纯模板拼接，不调模型。"""

from agents.summary_agent import SummaryAgent


class ExplodingLLM:
    """一旦被调用就报错，用来证明 SummaryAgent 不碰模型。"""

    def chat_with_json_output(self, *args, **kwargs):
        raise AssertionError("SummaryAgent 不应调用 LLM")

    def chat(self, *args, **kwargs):
        raise AssertionError("SummaryAgent 不应调用 LLM")

    def generate(self, *args, **kwargs):
        raise AssertionError("SummaryAgent 不应调用 LLM")


def make_context():
    return {
        "scene_type": "full_frame",
        "tracklets": [
            {"track_id": 1, "trajectory": [[0.0, 0.0, 0.0], [10.0, 0.0, 2.0]]},
            {"track_id": 2, "trajectory": [[5.0, 0.0, 0.0], [6.0, 0.0, 2.0]]},
        ],
        "previous_findings": [
            {
                "source": "trajectory_agent",
                "finding": {
                    "findings": [
                        {
                            "finding_type": "motion_pattern",
                            "value": {"label": "步行", "pose": "站姿端正"},
                            "evidence": ["speed=30px/s"],
                        },
                        {
                            "finding_type": "motion_pattern",
                            "value": {"label": "步行", "pose": "站姿端正"},
                            "evidence": ["speed=35px/s"],
                        },
                    ]
                },
            },
            {
                "source": "interaction_agent",
                "finding": {
                    "findings": [
                        {
                            "finding_type": "interaction",
                            "value": "结伴同行",
                            "evidence": ["distance=10px"],
                        }
                    ]
                },
            },
        ],
    }


def test_summary_is_template_rendered_from_previous_findings():
    agent = SummaryAgent(ExplodingLLM())
    agent.context = make_context()

    response = agent.think()

    summary = response["findings"][0]["value"]
    assert response["findings"][0]["finding_type"] == "scene_summary"
    assert "场景类型 full_frame" in summary
    assert "2 个目标" in summary
    assert "运动分布：步行 2" in summary
    assert "姿态分布：站姿端正 2" in summary
    assert "交互关系 1 条" in summary
    assert 0.6 <= response["agent_confidence"] <= 1.0


def test_summary_evidence_is_capped_and_keeps_source_evidence():
    agent = SummaryAgent(ExplodingLLM())
    agent.context = make_context()

    response = agent.think()

    evidence = response["findings"][0]["evidence"]
    assert "speed=30px/s" in evidence
    assert "distance=10px" in evidence
    assert len(evidence) <= 20


def test_scene_node_operation_is_emitted_for_kg():
    agent = SummaryAgent(ExplodingLLM())
    agent.context = make_context()

    ops = agent.think()["findings"][0]["kg_operations"]

    assert len(ops) == 1
    assert ops[0]["op"] == "upsert_node"
    assert ops[0]["node_type"] == "Scene"
    assert ops[0]["node_id"].startswith("Scene-")
    assert ops[0]["properties"]["entity_count"] == 2
    assert ops[0]["properties"]["scene_type"] == "full_frame"


def test_empty_context_still_produces_a_summary():
    agent = SummaryAgent(ExplodingLLM())
    agent.context = {}

    response = agent.think()

    assert "0 个目标" in response["findings"][0]["value"]
    assert response["agent_confidence"] >= 0.6
