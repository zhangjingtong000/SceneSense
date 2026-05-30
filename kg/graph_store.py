import json
import os
import time
from typing import Any

import networkx as nx
from networkx.algorithms import isomorphism

from utils.serialization import to_jsonable


class GraphStore:
    def __init__(self, use_directed: bool = True):
        self._graph: nx.DiGraph | nx.Graph = (
            nx.DiGraph() if use_directed else nx.Graph()
        )

    # --- 节点操作 ---

    @staticmethod
    def encode_property(value: Any) -> Any:
        value = to_jsonable(value)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return value

    @staticmethod
    def decode_property(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text or text[0] not in "[{":
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value

    @classmethod
    def _encode_properties(cls, properties: dict) -> dict:
        return {
            key: cls.encode_property(value)
            for key, value in properties.items()
        }

    @classmethod
    def _matches_filters(cls, data: dict, filters: dict | None) -> bool:
        if not filters:
            return True
        return all(
            cls.decode_property(data.get(key)) == value
            for key, value in filters.items()
        )

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def get_node(self, node_id: str) -> dict | None:
        data = self._graph.nodes.get(node_id)
        if data is not None:
            return {"node_id": node_id, **data}
        return None

    def upsert_node(self, node_id: str, node_type: str, properties: dict):
        encoded = self._encode_properties(properties)
        if self._graph.has_node(node_id):
            existing = self._graph.nodes[node_id]
            existing.update(encoded)
            existing["node_type"] = node_type
        else:
            self._graph.add_node(node_id, node_type=node_type, **encoded)

    def delete_node(self, node_id: str):
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)

    def query_nodes(
        self, node_type: str | None = None, filters: dict | None = None
    ) -> list[dict]:
        results: list[dict] = []
        for nid, data in self._graph.nodes(data=True):
            if node_type is not None and data.get("node_type") != node_type:
                continue
            if not self._matches_filters(data, filters):
                continue
            results.append({"node_id": nid, **data})
        return results

    def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    # --- 边操作 ---

    def has_edge(self, src: str, tgt: str) -> bool:
        return self._graph.has_edge(src, tgt)

    def get_edge(self, src: str, tgt: str) -> dict | None:
        data = self._graph.edges.get((src, tgt))
        if data is not None:
            return {"src": src, "tgt": tgt, **data}
        return None

    def upsert_edge(self, src: str, tgt: str, edge_type: str, properties: dict):
        encoded = self._encode_properties(properties)
        if self._graph.has_edge(src, tgt):
            existing = self._graph.edges[src, tgt]
            existing.update(encoded)
            existing["edge_type"] = edge_type
        else:
            self._graph.add_edge(src, tgt, edge_type=edge_type, **encoded)

    def query_edges(
        self, edge_type: str | None = None, filters: dict | None = None
    ) -> list[dict]:
        results: list[dict] = []
        for src, tgt, data in self._graph.edges(data=True):
            if edge_type is not None and data.get("edge_type") != edge_type:
                continue
            if not self._matches_filters(data, filters):
                continue
            results.append({"src": src, "tgt": tgt, **data})
        return results

    def get_node_edges(self, node_id: str) -> list:
        if self._graph.has_node(node_id):
            return list(self._graph.edges(node_id, data=True))
        return []

    # --- 子图 ---

    def get_subgraph(self, node_ids: list[str], depth: int = 1) -> nx.Graph:
        if depth < 0:
            return self._graph.copy()
        nodes_to_include: set[str] = set(node_ids)
        frontier: set[str] = set(node_ids)
        for _ in range(depth):
            neighbors: set[str] = set()
            for nid in frontier:
                neighbors.update(self._graph.neighbors(nid))
            nodes_to_include.update(neighbors)
            frontier = neighbors
        return self._graph.subgraph(nodes_to_include).copy()

    def match_pattern(self, pattern: nx.Graph) -> list[dict]:
        if pattern.number_of_nodes() == 0:
            return []

        def _node_match(n1_attrs: dict, n2_attrs: dict) -> bool:
            for k, v in n1_attrs.items():
                if k == "node_id":
                    continue
                if k not in n2_attrs or n2_attrs[k] != v:
                    return False
            return True

        def _edge_match(e1_attrs: dict, e2_attrs: dict) -> bool:
            for k, v in e1_attrs.items():
                if k in ("src", "tgt"):
                    continue
                if k not in e2_attrs or e2_attrs[k] != v:
                    return False
            return True

        results: list[dict] = []
        matcher = isomorphism.GraphMatcher(
            self._graph,
            pattern,
            node_match=_node_match,
            edge_match=_edge_match,
        )
        for mapping in matcher.subgraph_isomorphisms_iter():
            results.append({
                "node_mapping": mapping.copy(),
                "score": 1.0,
            })
        return results

    # --- 生命周期 ---

    def set_ttl(self, node_id: str, ttl_seconds: float):
        if self._graph.has_node(node_id):
            self._graph.nodes[node_id]["expires_at"] = time.time() + ttl_seconds

    def evict_before(self, timestamp: float) -> int:
        to_remove: list[str] = []
        for nid, data in self._graph.nodes(data=True):
            if data.get("node_type") != "Event":
                continue
            expires_at = data.get("expires_at")
            if expires_at is not None and expires_at < timestamp:
                to_remove.append(nid)

        for nid in to_remove:
            self._graph.remove_node(nid)
        return len(to_remove)

    # --- 持久化 ---

    def save(self, path: str):
        nx.write_graphml(self._graph, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        self._graph = nx.read_graphml(path)
        return True

    def to_dict(self) -> dict:
        nodes: list[dict] = []
        for nid, data in self._graph.nodes(data=True):
            entry: dict[str, Any] = {"node_id": nid}
            entry.update(data)
            nodes.append(entry)

        edges: list[dict] = []
        for src, tgt, data in self._graph.edges(data=True):
            entry: dict[str, Any] = {"src": src, "tgt": tgt}
            entry.update(data)
            edges.append(entry)

        return {"nodes": nodes, "edges": edges}

    @property
    def stats(self) -> dict:
        node_types: dict[str, int] = {}
        edge_types: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1
        for _, _, data in self._graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1
        return {
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
        }
