"""scripts/qa_repl.KGQA：/chat 背后的问答逻辑（无网络）。"""

import networkx as nx

from scripts.qa_repl import KGQA


def _graph():
    g = nx.DiGraph()
    g.add_node(
        "Person-1",
        **{"node_type": "Entity", "class": "person",
           "motion_profile": "walking", "motion_description": "行走"}
    )
    g.add_node(
        "Person-3",
        **{"node_type": "Entity", "class": "person",
           "motion_profile": "loitering", "motion_description": "缓慢徘徊",
           "velocity": "0.4"}
    )
    g.add_node(
        "Person-5",
        **{"node_type": "Entity", "class": "person",
           "motion_profile": "running", "motion_description": "奔跑"}
    )
    g.add_node("Event-1", node_type="Event")
    g.add_edge("Person-1", "Person-3", edge_type="interacts_with", distance=42, duration=3.5)
    # 方向相反的一条边：DiGraph 下 `G.edges(nid)` 只给 out 边，入边也要能查到
    g.add_edge("Person-5", "Person-3", edge_type="interacts_with", distance=88, duration=1.0)
    return g


def test_entity_status_resolves_person_numbering():
    answer = KGQA(_graph()).answer("3号在干什么")

    assert "Person-3" in answer
    assert "徘徊" in answer
    assert "未找到" not in answer


def test_nearby_follows_incoming_and_outgoing_edges():
    answer = KGQA(_graph()).answer("3号附近有谁")

    assert "Person-1" in answer
    assert "Person-5" in answer


def test_entity_relation_reports_edge_attributes():
    answer = KGQA(_graph()).answer("1号和3号什么关系")

    assert "interacts_with" in answer
    assert "42" in answer


def test_scene_stats_counts_entities_only():
    answer = KGQA(_graph()).answer("场景里有多少人")

    assert "3 个目标" in answer


def test_who_is_matches_motion_label():
    answer = KGQA(_graph()).answer("谁在跑动")

    assert "Person-5" in answer


def test_legacy_entity_prefix_still_resolves():
    graph = nx.DiGraph()
    graph.add_node(
        "Entity-9",
        **{"node_type": "Entity", "class": "person",
           "motion_profile": "stationary", "motion_description": "站立"}
    )

    answer = KGQA(graph).answer("9号在干什么")

    assert "Entity-9" in answer
    assert "未找到" not in answer


def test_merge_command_does_not_hit_llm():
    class ExplodingLLM:
        def chat(self, messages):
            raise AssertionError("__merge__ 不应该触发 LLM")

    answer = KGQA(_graph(), ExplodingLLM()).answer("__merge__ Person-1 Person-3")

    assert "前端" in answer


def test_unknown_question_falls_back_to_llm():
    calls = []

    class FakeLLM:
        def chat(self, messages):
            calls.append(messages)
            return "  场景里只有行人。  "

    answer = KGQA(_graph(), FakeLLM()).answer("今天天气怎么样")

    assert answer == "场景里只有行人。"
    assert len(calls) == 1


def test_unknown_question_without_llm_returns_hint():
    answer = KGQA(_graph()).answer("今天天气怎么样")

    assert "抱歉" in answer
