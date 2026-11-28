from .graph_store import GraphStore


class RuleEngine:
    def __init__(self, graph_store: GraphStore):
        self.gs = graph_store
        self._rules: dict[str, dict] = {}

    def add_rule(self, rule: dict):
        self._rules[rule["rule_id"]] = rule

    def check_entity(self, entity_id: str, timestamp: float) -> list[dict]:
        entity = self.gs.get_node(entity_id)
        if entity is None:
            return []

        results: list[dict] = []
        for rule_id, rule in self._rules.items():
            conditions = rule.get("conditions", {})
            matched = True
            details: dict = {}

            if "zone_type" in conditions:
                entity_zone = self._resolve_entity_zone(entity_id)
                details["entity_zone"] = entity_zone
                if entity_zone != conditions["zone_type"]:
                    matched = False

            if "motion_profile" in conditions:
                ep = entity.get("motion_profile")
                details["entity_motion_profile"] = ep
                if ep != conditions["motion_profile"]:
                    matched = False

            if "max_stay_seconds" in conditions:
                first_seen = entity.get("first_seen")
                if first_seen is not None:
                    stay = timestamp - first_seen
                    details["stay_seconds"] = stay
                    if stay <= conditions["max_stay_seconds"]:
                        matched = False
                else:
                    matched = False

            if "velocity_max" in conditions:
                vel = entity.get("velocity")
                details["entity_velocity"] = vel
                if vel is None or vel > conditions["velocity_max"]:
                    matched = False

            if "velocity_min" in conditions:
                vel = entity.get("velocity")
                details["entity_velocity"] = details.get(
                    "entity_velocity", vel
                )
                if vel is None or vel < conditions["velocity_min"]:
                    matched = False

            if "direction_change_max" in conditions:
                dc = entity.get("direction_change_count")
                details["direction_change_count"] = dc
                if dc is not None and dc > conditions["direction_change_max"]:
                    matched = False

            if "class" in conditions:
                cls = entity.get("class")
                details["entity_class"] = cls
                if cls != conditions["class"]:
                    matched = False

            results.append({
                "rule_id": rule_id,
                "rule_description": rule.get("description", ""),
                "matched": matched,
                "severity": rule.get("severity", 0.5),
                "details": details,
            })

        return results

    def check_all(self, timestamp: float) -> list[dict]:
        entities = self.gs.query_nodes(node_type="Entity")
        results: list[dict] = []
        for ent in entities:
            eid = ent["node_id"]
            if ent.get("active") is False:
                continue
            for r in self.check_entity(eid, timestamp):
                if r["matched"]:
                    r["entity_id"] = eid
                    results.append(r)
        return results

    def get_rules_for_zone(self, zone_type: str) -> list[dict]:
        return [
            r
            for r in self._rules.values()
            if r.get("conditions", {}).get("zone_type") == zone_type
        ]

    # --- 内部辅助 ---

    def _resolve_entity_zone(self, entity_id: str) -> str | None:
        """Traverse belongs_to edge from entity to its event and return zone_type."""
        edges = self.gs.get_node_edges(entity_id)
        for src, tgt, data in edges:
            if data.get("edge_type") == "belongs_to":
                event = self.gs.get_node(tgt)
                if event is not None:
                    return event.get("zone_type")
        return None


def default_rules() -> list[dict]:
    """内置的一组演示规则。

    这些规则是「规则引擎能跑起来」的最小集合，不是某个场景的完整规则库。
    每条规则的 `conditions` 键与 `RuleEngine.check_entity` 里的分支一一对应：
    `motion_profile`（英文运动标签）、`max_stay_seconds`（停留超过阈值才算命中）、
    `velocity_max` / `velocity_min`、`direction_change_max`、`zone_type`、`class`。
    """
    return [
        {
            "rule_id": "R-LOITER-60",
            "description": "在区域内徘徊且停留超过 60 秒",
            "severity": 0.6,
            "conditions": {"motion_profile": "loitering", "max_stay_seconds": 60},
        },
        {
            "rule_id": "R-RUN",
            "description": "出现奔跑（可能是追逐或逃离）",
            "severity": 0.7,
            "conditions": {"motion_profile": "running"},
        },
        {
            "rule_id": "R-STATIC-120",
            "description": "静止超过 120 秒（可能晕倒或物品遗弃）",
            "severity": 0.5,
            "conditions": {"motion_profile": "stationary", "max_stay_seconds": 120},
        },
    ]
