"""Doubao Seed 2.0 Lite VLM client via Volcengine Ark API (OpenAI-compatible)."""

import base64
import json
import time
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image


def encode_frame_jpeg(frame: np.ndarray, quality: int = 75) -> str:
    """numpy BGR frame -> base64 JPEG string."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class DoubaoVLMClient:
    """Volcengine Ark API for Doubao Seed 2.0 Lite multimodal."""

    def __init__(
        self,
        api_key: str,
        model: str = "doubao-seed-2-0-lite-260428",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        timeout: float = 30.0,
    ):
        import openai
        self.model = model
        self.timeout = timeout
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def analyze_pair(
        self,
        encoded_image: str,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Send image + geometric features, get interaction classification."""

        def fmt(v, unit=""):
            if v is None:
                return "未知"
            if isinstance(v, float):
                return f"{v:.1f}{unit}"
            return f"{v}{unit}"

        feature_text = (
            f"距离均值: {fmt(features.get('distance_mean'), 'px')}, "
            f"距离标准差: {fmt(features.get('distance_std'), 'px')}, "
            f"轨迹方向余弦: {fmt(features.get('trajectory_cosine'))}, "
            f"朝向差: {fmt(features.get('orientation_diff'), '°')}, "
            f"共存时长: {fmt(features.get('co_present_seconds'), 's')}, "
            f"距离趋势: {fmt(features.get('distance_trend'))} (负=靠近 正=远离)"
        )

        prompt = (
            "你是视频监控交互分析助手。看这张两人同框截图，结合以下时空特征，判断交互类型：\n\n"
            f"时空特征：{feature_text}\n\n"
            "重要提示：轨迹方向余弦≈1.0=同向移动，≈-1.0=相向移动，≈0=垂直交叉。\n"
            "距离趋势斜率为负=两人在靠近，为正=两人在远离。\n\n"
            "选项：结伴同行 / 面对面交谈 / 短暂接触 / 无交互\n\n"
            "只输出 JSON，interaction_type 必须是这四个之一：\n"
            '{"interaction_type": "结伴同行", "confidence": 0.xx, "reasoning": "..."}'
        )

        try:
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                temperature=0.1,
                max_tokens=300,
            )
            elapsed = time.perf_counter() - t0
            raw = resp.choices[0].message.content
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(raw[start:end])
                result["_latency_ms"] = round(elapsed * 1000, 1)
                return result
            return {"interaction_type": "未知", "confidence": 0.3, "reasoning": raw[:200], "_latency_ms": round(elapsed * 1000, 1)}
        except Exception as e:
            return {"interaction_type": "未知", "confidence": 0.0, "reasoning": str(e)[:200], "_latency_ms": 0.0}
