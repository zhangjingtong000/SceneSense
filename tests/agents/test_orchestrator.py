"""Orchestrator（星型协调节点）的路由、上下文传递与 KG 落库。"""

from agents.orchestrator import MockKGAPI, Orchestrator


class FakeAgent:
    """最小 Agent：记录 observe 到的上下文，act 返回预置 findings。"""

    def __init__(self, name, findings=None, confidence=0.8):
        self.name = name
        self.system_prompt = ""
        self.contexts = []
        self._findings = findings or []
        self._confidence = confidence

    def observe(self, context):
        self.contexts.append(context)

    def act(self):
        return {
            "message_id": f"msg-{self.name}",
            "sender": self.name,
            "target": "organizer",
            "timestamp": 0.0,
            "agent_confidence": self._confidence,
            "warnings": [f"{self.name} warning"] if self.name == "trajectory_agent" else [],
            "findings": list(self._findings),
        }


def make_event(event_id="Event-1"):
    return {
        "event_id": event_id,
        "timestamp": 1.0,
        "duration": [0.0, 1.0],
        "frame_start": 0,
        "frame_end": 30,
        "zone_id": "zone_default",
        "zone_type": "full_frame",
        "tracklets": [
            {
                "track_id": 1,
                "entity_id": "Person-1",
                "velocity_ms": 30.0,
                "trajectory": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1]],
            }
        ],
        "spatial_relations": [
            {"track_1": 1, "track_2": 2, "distance_m": 10.0, "relation_type": "near"}
        ],
    }


def motion_finding(kg_operations=None):
    return {
        "entity_id": "Person-1",
        "finding_type": "motion_description",
        "value": {"label": "步行", "speed_px": 30.0},
        "confidence": 0.8,
        "evidence": ["speed=30px/s"],
        "kg_operations": kg_operations or [],
    }


def interaction_finding():
    return {
        "entity_id": "Person-1",
        "related_entities": ["Person-2"],
        "finding_type": "interaction",
        "value": "结伴同行",
        "confidence": 0.7,
        "evidence": [],
        "kg_operations": [],
    }


def summary_finding():
    return {
        "entity_id": "",
        "finding_type": "scene_summary",
        "value": "两个人一起走过画面",
        "confidence": 0.8,
        "evidence": [],
        "kg_operations": [],
    }


def build_orchestrator(fake_agents, kg_api=None):
    orch = Orchestrator(llm_client=object(), kg_api=kg_api or MockKGAPI())
    for agent in fake_agents:
        orch.register_agent(agent)
    return orch


def test_default_route_order_is_trajectory_then_interaction_then_summary():
    orch = build_orchestrator([])
    assert orch.route_order == ["trajectory_agent", "interaction_agent", "summary_agent"]


def test_optional_agent_can_be_inserted_before_summary():
    orch = build_orchestrator([])
    thrice = FakeAgent("thrice_agent")
    orch.register_agent(thrice, position=2)
    assert orch.route_order == [
        "trajectory_agent",
        "interaction_agent",
        "thrice_agent",
        "summary_agent",
    ]


def test_registering_known_agent_does_not_duplicate_route():
    orch = build_orchestrator([])
    traj = FakeAgent("trajectory_agent")
    orch.register_agent(traj)
    orch.register_agent(traj)
    assert orch.route_order.count("trajectory_agent") == 1


def test_each_agent_sees_previous_findings_in_registration_order():
    traj = FakeAgent("trajectory_agent", [motion_finding()])
    inter = FakeAgent("interaction_agent", [interaction_finding()])
    summary = FakeAgent("summary_agent", [summary_finding()])
    orch = build_orchestrator([traj, inter, summary])

    orch.process_event(make_event())

    # 第一个 Agent 拿不到前序发现，第二个能看到第一个的响应，第三个能看到前两个
    assert traj.contexts[0]["previous_findings"] == []
    assert [p["source"] for p in inter.contexts[0]["previous_findings"]] == [
        "trajectory_agent"
    ]
    assert [p["source"] for p in summary.contexts[0]["previous_findings"]] == [
        "trajectory_agent",
        "interaction_agent",
    ]
    assert inter.contexts[0]["tracklets"][0]["entity_id"] == "Person-1"
    assert inter.contexts[0]["spatial_relations"][0]["distance_m"] == 10.0


def test_all_agent_responses_expose_structured_findings_to_later_agents():
    traj = FakeAgent("trajectory_agent", [motion_finding()])
    inter = FakeAgent("interaction_agent", [interaction_finding()])
    orch = build_orchestrator([traj, inter])

    orch.process_event(make_event())

    assert set(inter.contexts[0]["all_agent_responses"]) == {"trajectory_agent"}


def test_scene_report_aggregates_agents_and_keeps_route_history():
    traj = FakeAgent("trajectory_agent", [motion_finding()])
    inter = FakeAgent("interaction_agent", [interaction_finding()])
    summary = FakeAgent("summary_agent", [summary_finding()])
    orch = build_orchestrator([traj, inter, summary])

    report = orch.process_event(make_event())

    assert report["event_id"] == "Event-1"
    assert report["frame_range"] == [0, 30]
    assert report["scene_report"]["summary"] == "两个人一起走过画面"
    assert report["scene_report"]["entity_count"] == 1
    assert report["scene_report"]["interaction_descriptions"][0]["entities"] == [
        "Person-1",
        "Person-2",
    ]
    assert report["agent_summary"]["trajectory_agent_confidence"] == 0.8
    assert report["agent_summary"]["trajectory_agent_warnings"] == ["trajectory_agent warning"]
    assert orch.route_history == [
        "trajectory_agent",
        "interaction_agent",
        "summary_agent",
    ]
    assert [step["agent"] for step in report["reasoning_trail"]] == orch.route_history
    assert "1 个目标轨迹" in report["reasoning_trail"][0]["input_summary"]


def test_kg_operations_from_findings_are_written_to_kg_api():
    kg_ops = [
        {
            "op": "upsert_node",
            "node_type": "Entity",
            "node_id": "Person-1",
            "properties": {"entity_id": "Person-1", "class": "person"},
        },
        {
            "op": "upsert_edge",
            "edge_type": "belongs_to",
            "from": "Person-1",
            "to": "Event-1",
            "properties": {"frame_start": 0},
        },
    ]
    traj = FakeAgent("trajectory_agent", [motion_finding(kg_operations=kg_ops)])
    kg = MockKGAPI()
    orch = build_orchestrator([traj], kg_api=kg)

    report = orch.process_event(make_event())

    assert len(report["kg_updates"]) == 2
    assert kg.nodes["Person-1"]["node_type"] == "Entity"
    assert kg.edges[0]["type"] == "belongs_to"
    assert kg.edges[0]["from"] == "Person-1"


def test_failing_kg_operation_does_not_break_the_report():
    class ExplodingKG(MockKGAPI):
        def upsert_node(self, node_id, node_type, properties):
            raise RuntimeError("kg down")

    kg_ops = [{"op": "upsert_node", "node_type": "Entity", "node_id": "Person-1", "properties": {}}]
    traj = FakeAgent("trajectory_agent", [motion_finding(kg_operations=kg_ops)])
    orch = build_orchestrator([traj], kg_api=ExplodingKG())

    report = orch.process_event(make_event())

    assert report["event_id"] == "Event-1"


def test_process_event_resets_state_between_events():
    traj = FakeAgent("trajectory_agent", [motion_finding()])
    orch = build_orchestrator([traj])

    first = orch.process_event(make_event("Event-1"))
    second = orch.process_event(make_event("Event-2"))

    assert first["event_id"] == "Event-1"
    assert second["event_id"] == "Event-2"
    assert orch.route_history == ["trajectory_agent"]
    assert traj.contexts[1]["event_ids"] == ["Event-2"]
