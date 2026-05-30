"""
Agent 基类。

每个 Agent 的生命周期:
  observe(接收事件) → think(调用LLM推理) → act(输出findings+kg_operations)
"""

import uuid
from datetime import datetime
from typing import Any

from .llm_client import LLMClient


class BaseAgent:
    """Agent 基类。

    所有分析 Agent 都继承此类。子类需重写 think() 方法。
    """

    def __init__(self, name: str, llm_client: LLMClient, system_prompt: str) -> None:
        self.name = name
        self.llm = llm_client
        self.system_prompt = system_prompt
        self.context: dict[str, Any] = {}

    def observe(self, context: dict) -> None:
        """接收协调Agent传来的上下文（包含事件数据、先前发现等）。"""
        self.context = context

    def think(self) -> dict:
        """调用 LLM 进行推理。子类需要重写此方法。"""
        raise NotImplementedError

    def act(self) -> dict:
        """输出标准化的 AgentResponse (§4.3)。

        Returns:
            AgentResponse dict:
            {
                "message_id": str,
                "sender": str,
                "target": "organizer",
                "timestamp": float,
                "agent_confidence": float,
                "warnings": list[str],
                "findings": list[dict]
            }
        """
        result = self.think()
        return self._format_response(result)

    def _format_response(self, raw_result: dict) -> dict:
        """将 LLM 原始输出格式化为符合 AgentResponse 契约的 dict。"""
        return {
            "message_id": f"msg-{uuid.uuid4().hex[:8]}",
            "sender": self.name,
            "target": "organizer",
            "timestamp": datetime.now().timestamp(),
            "agent_confidence": raw_result.get("agent_confidence", 0.5),
            "warnings": raw_result.get("warnings", []),
            "findings": raw_result.get("findings", []),
        }

    def _load_prompt(self, name: str) -> str:
        """从 prompts/{name}.md 加载 system prompt。"""
        import os

        prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts", f"{name}.md"
        )
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return f"You are the {name} analysis agent. Analyze the given data."
