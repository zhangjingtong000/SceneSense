# -*- coding: utf-8 -*-
"""规则引擎的回归测试。

这套代码此前零测试、零调用点：`kg/rule_engine.py` 写得挺完整，但没有任何地方
`import` 它，所以「能匹配规则」这件事从来没被验证过。同时 `Entity` 节点缺少
`motion_profile` / `velocity` / `first_seen` 等字段，即使调用了也一条都匹配不上。

现在两件事都补了：`trajectory_agent` 把字段写进 Entity，pipeline 加 `--with-rules`
在 Agent 之后调用引擎。这些测试盯的就是这条链。
"""
import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from kg.graph_store import GraphStore          # noqa: E402
from kg.rule_engine import RuleEngine, default_rules  # noqa: E402


@pytest.fixture
def engine():
    gs = GraphStore(use_directed=True)
    eng = RuleEngine(gs)
    for r in default_rules():
        eng.add_rule(r)
    return eng


def _put(gs, eid, **props):
    gs.upsert_node(eid, "Entity", props)


def test_default_rules_are_loadable():
    rules = default_rules()
    assert len(rules) == 3
    assert {r["rule_id"] for r in rules} == {"R-LOITER-60", "R-RUN", "R-STATIC-120"}
    for r in rules:
        assert r["conditions"], f"{r['rule_id']} 没有匹配条件，永远不可能命中"
        assert "rule_id" in r and "severity" in r


def test_loiter_rule_hits_after_threshold(engine):
    """徘徊 + 停留超过 60 秒 → R-LOITER-60 命中。"""
    gs = engine.gs
    _put(gs, "Entity-1", motion_profile="loitering", first_seen=0.0,
         **{"class": "person"}, active=True)
    hits = engine.check_entity("Entity-1", 100.0)
    matched = {h["rule_id"] for h in hits if h["matched"]}
    assert "R-LOITER-60" in matched, f"100s > 60s 却没命中，实际命中 {matched}"


def test_loiter_rule_misses_before_threshold(engine):
    """停留不足 60 秒时不该命中——否则规则会把刚出现的人直接判成徘徊。"""
    gs = engine.gs
    _put(gs, "Entity-2", motion_profile="loitering", first_seen=0.0,
         **{"class": "person"}, active=True)
    hits = engine.check_entity("Entity-2", 30.0)
    matched = {h["rule_id"] for h in hits if h["matched"]}
    assert "R-LOITER-60" not in matched


def test_run_rule_hits_on_running(engine):
    gs = engine.gs
    _put(gs, "Entity-3", motion_profile="running", first_seen=0.0,
         **{"class": "person"}, active=True)
    hits = engine.check_entity("Entity-3", 5.0)
    matched = {h["rule_id"] for h in hits if h["matched"]}
    assert "R-RUN" in matched


def test_no_rule_hits_when_fields_missing(engine):
    """如果 Entity 没写 motion_profile（老数据），所有规则都不该命中。

    这条钉住的是「字段缺失时引擎静默失效」——它不报错，所以必须有测试兜住。
    """
    gs = engine.gs
    _put(gs, "Entity-4", **{"class": "person"}, active=True)
    hits = engine.check_entity("Entity-4", 100.0)
    assert hits, "至少应该返回每条规则的判定结果"
    assert not any(h["matched"] for h in hits), (
        "字段缺失却命中了规则，说明条件是 None 比较时被当成相等")


def test_check_all_skips_inactive_entities(engine):
    gs = engine.gs
    _put(gs, "Entity-5", motion_profile="running", first_seen=0.0,
         **{"class": "person"}, active=True)
    _put(gs, "Entity-6", motion_profile="running", first_seen=0.0,
         **{"class": "person"}, active=False)
    hits = engine.check_all(5.0)
    assert hits, "至少有一条命中"
    assert {h["entity_id"] for h in hits} == {"Entity-5"}
    assert not any(h["entity_id"] == "Entity-6" for h in hits)


def test_trajectory_agent_writes_rule_fields():
    """`trajectory_agent` 建 Entity 时必须写全规则引擎要用的字段。"""
    from agents.trajectory_agent import TrajectoryAgent

    agent = TrajectoryAgent(llm_client=None)
    tracklet = {
        "entity_id": "Entity-7",
        "track_id": 7,
        "class": "person",
        "velocity_ms": 0.4,
        "trajectory": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.033], [2.0, 2.1, 0.066]],
    }
    agent.context = {"tracklets": [tracklet], "scene_type": "outdoor"}
    out = agent.think()
    findings = out["findings"]
    assert findings, "没有产出 finding"
    props = findings[0]["kg_operations"][0]["properties"]
    for key in ("motion_profile", "velocity", "direction_change_count",
                "first_seen", "class", "active"):
        assert key in props, f"Entity 属性缺少 {key}，规则引擎无法匹配"
    assert props["motion_profile"], "motion_profile 不能为空"
    assert props["active"] is True
