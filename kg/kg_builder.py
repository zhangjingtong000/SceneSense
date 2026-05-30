import json
import time
import uuid
from typing import Any

from .graph_store import GraphStore


class KGBuilder:
    def __init__(self, llm_client: Any, graph_store: GraphStore):
        self.llm = llm_client
        self.gs = graph_store

    def build_behavior_templates(self, domain_description: str) -> list[str]:
        prompt = f"""You are a security surveillance domain expert. Given a domain description, you need to generate a list of behavior templates for video behavior analysis.

Each behavior template must be a JSON object with these fields:
- "behavior_id": a unique id like "Behavior-{uuid4().hex[:8]}"
- "pattern_type": short name of the behavior pattern (e.g., "fighting", "loitering", "tailgating")
- "description": a brief description of what this behavior looks like
- "rule_definition": an object describing the spatial-temporal conditions, including:
  - "zone_types": list of applicable zone types
  - "motion_profiles": list of matching motion profiles
  - "velocity_range": [min, max] in m/s
  - "duration_threshold": minimum duration in seconds for the behavior to be meaningful
  - "interaction_conditions": optional, conditions on entity-entity interactions
- "severity": a float in [0, 1] indicating severity level
- "preconditions": an object with conditions under which this template should be checked (e.g., "min_entities": 1, "required_class": "person")
- "examples": list of string examples describing concrete instances
- "source": "llm"

Domain description: {domain_description}

Return a JSON list of behavior templates. Only return the JSON array, no other text."""

        response = self.llm.generate(prompt)
        templates = self._parse_json(response)

        behavior_ids: list[str] = []
        if isinstance(templates, dict) and "templates" in templates:
            templates = templates["templates"]
        if isinstance(templates, dict):
            templates = [templates]

        for tmpl in templates:
            bid = tmpl.get(
                "behavior_id",
                f"Behavior-{uuid.uuid4().hex[:8]}",
            )
            self.gs.upsert_node(
                bid,
                "Behavior",
                {
                    "pattern_type": tmpl.get("pattern_type", ""),
                    "rule_definition": json.dumps(
                        tmpl.get("rule_definition", {}), ensure_ascii=False
                    ),
                    "severity": tmpl.get("severity", 0.5),
                    "preconditions": json.dumps(
                        tmpl.get("preconditions", {}), ensure_ascii=False
                    ),
                    "examples": json.dumps(
                        tmpl.get("examples", []), ensure_ascii=False
                    ),
                    "description": tmpl.get("description", ""),
                    "source": "llm",
                },
            )
            behavior_ids.append(bid)

        return behavior_ids

    def build_scene_knowledge(self, scene_description: str) -> str:
        prompt = f"""You are a scene knowledge builder for a video surveillance system. Given a scene description, generate a single Scene node.

The Scene node must be a JSON object with these fields:
- "scene_id": a unique id like "Scene-{uuid4().hex[:8]}"
- "scene_type": type of scene (e.g., "outdoor_plaza", "indoor_corridor", "parking_lot")
- "time_range": [start_hour, end_hour] or null if unknown
- "zone_id": primary zone id
- "entity_count": expected number of entities (int)
- "global_description": detailed description of the scene including spatial layout, key areas, typical activities
- "weather": weather condition if outdoor (e.g., "sunny", "rainy", null)
- "lighting": lighting condition (e.g., "daylight", "night", "artificial")

Scene description: {scene_description}

Return a single JSON object representing the Scene node. Only return the JSON object, no other text."""

        response = self.llm.generate(prompt)
        scene = self._parse_json(response)

        if isinstance(scene, list) and len(scene) > 0:
            scene = scene[0]

        sid = scene.get(
            "scene_id",
            f"Scene-{uuid.uuid4().hex[:8]}",
        )
        self.gs.upsert_node(
            sid,
            "Scene",
            {
                "scene_type": scene.get("scene_type", ""),
                "time_range": json.dumps(
                    scene.get("time_range"), ensure_ascii=False
                )
                if scene.get("time_range")
                else "[]",
                "zone_id": scene.get("zone_id", ""),
                "entity_count": scene.get("entity_count", 0),
                "global_description": scene.get("global_description", ""),
                "weather": scene.get("weather"),
                "lighting": scene.get("lighting"),
            },
        )
        return sid

    def generate_rules_from_template(self, behavior_id: str) -> list[dict]:
        node = self.gs.get_node(behavior_id)
        if node is None:
            return []

        rule_def_raw = node.get("rule_definition", "{}")
        try:
            rule_def = (
                json.loads(rule_def_raw)
                if isinstance(rule_def_raw, str)
                else rule_def_raw
            )
        except (json.JSONDecodeError, TypeError):
            rule_def = {}

        pattern_type = node.get("pattern_type", "unknown")
        severity = node.get("severity", 0.5)
        description = node.get("description", "")

        rules: list[dict] = []
        zone_types = rule_def.get("zone_types", [])
        motion_profiles = rule_def.get("motion_profiles", [])
        velocity_range = rule_def.get("velocity_range", [0, 100])
        duration_threshold = rule_def.get("duration_threshold", 0)
        interaction_conditions = rule_def.get("interaction_conditions", {})

        if not zone_types:
            zone_types = ["*"]

        for zt in zone_types:
            conditions: dict = {"zone_type": zt}

            if motion_profiles:
                conditions["motion_profile"] = motion_profiles[0]

            if velocity_range[0] > 0:
                conditions["velocity_min"] = velocity_range[0]
            if velocity_range[1] < 100:
                conditions["velocity_max"] = velocity_range[1]

            if duration_threshold > 0:
                conditions["max_stay_seconds"] = duration_threshold

            rules.append({
                "rule_id": f"rule-{behavior_id}-{len(rules) + 1}",
                "description": f"{description} (zone: {zt})",
                "conditions": conditions,
                "severity": severity,
                "source_template": behavior_id,
            })

        if interaction_conditions:
            rules.append({
                "rule_id": f"rule-{behavior_id}-interaction",
                "description": f"{description} (interaction rule)",
                "conditions": {
                    "zone_type": zone_types[0] if zone_types else "*",
                    "interaction": interaction_conditions,
                },
                "severity": severity,
                "source_template": behavior_id,
            })

        return rules

    # --- 内部辅助 ---

    def _parse_json(self, text: str) -> Any:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2:
                text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return {}
