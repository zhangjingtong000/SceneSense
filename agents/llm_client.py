"""
DeepSeek API 封装（openai SDK 兼容格式）。

提供普通 chat 和 JSON 结构化输出两种调用方式。
"""

import json
from typing import Any, Optional

import openai


class LLMClient:
    """封装 DeepSeek API 调用。

    使用 openai SDK（DeepSeek 兼容 openai 格式）。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """单字符串快捷调用（不带 system prompt 的场景用这一条）。

        Args:
            prompt: 完整 prompt 文本
            **kwargs: 可覆盖 model/temperature/max_tokens

        Returns:
            LLM 响应文本
        """
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        """单轮对话。

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            **kwargs: 可覆盖 model/temperature/max_tokens

        Returns:
            response.choices[0].message.content
        """
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content

    def chat_with_json_output(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Optional[dict] = None,
    ) -> dict:
        """调用 LLM 并要求返回 JSON。

        prompt 末尾自动添加 "Output valid JSON only."。
        解析失败返回 {"error": "parse_failed", "raw": str}。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            output_schema: 可选，JSON schema 约束

        Returns:
            解析后的 dict，或错误 dict
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt + "\n\nOutput valid JSON only.",
            },
        ]

        try:
            kwargs = {}
            if output_schema is not None:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
            content: str = response.choices[0].message.content

            # 处理 markdown 代码块包裹的情况
            content_stripped = content.strip()
            if content_stripped.startswith("```"):
                lines = content_stripped.split("\n")
                fence_starts = [
                    i for i, line in enumerate(lines) if line.startswith("```")
                ]
                if len(fence_starts) >= 2:
                    start = fence_starts[0]
                    end = fence_starts[-1]
                    content_stripped = "\n".join(lines[start + 1 : end])
                elif len(fence_starts) == 1:
                    content_stripped = "\n".join(lines[fence_starts[0] + 1 :])

            return json.loads(content_stripped)

        except (json.JSONDecodeError, AttributeError) as exc:
            return {"error": "parse_failed", "raw": str(exc)}
        except Exception as exc:
            return {"error": "api_error", "raw": str(exc)}
