"""Qwen3-VL 多模态交互分析客户端 (Ollama API)."""

import base64
import json
import math
import time as _time
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image


# ── 几何特征提取 ──────────────────────────────────────────────

def _compute_orientation(keypoints: list[list[float]]) -> float | None:
    """从 COCO 17 关键点计算身体朝向角度 (0-360, 0=右)."""
    if len(keypoints) < 17:
        return None
    ls = keypoints[5]  # 左肩
    rs = keypoints[6]  # 右肩
    if ls[2] < 0.3 or rs[2] < 0.3:
        return None
    mid_shoulder = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
    nose = keypoints[0]
    if nose[2] < 0.3:
        return None
    vec = np.array([nose[0] - mid_shoulder[0], nose[1] - mid_shoulder[1]])
    norm = float(np.linalg.norm(vec))
    if norm < 1.0:
        return None
    angle = math.degrees(math.atan2(vec[1], vec[0])) % 360
    return round(angle, 1)


def _orientation_diff(a: float, b: float) -> float:
    """两个朝向之间的最小角度差 (0-180)."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def geometric_filter(
    person_a: dict,
    person_b: dict,
    min_frames: int = 10,
    max_distance_px: float = 200.0,
) -> dict[str, Any]:
    """几何预过滤 + 特征提取.

    先基于轨迹/朝向/距离做硬过滤，避免无意义的 VLM 调用。
    返回 pass_filter、reason 和 features 字典。
    """
    traj_a = person_a.get("trajectory", [])
    traj_b = person_b.get("trajectory", [])
    kps_a = person_a.get("pose_keypoints", [])
    kps_b = person_b.get("pose_keypoints", [])

    if len(traj_a) < 2 or len(traj_b) < 2:
        return {"pass_filter": False, "reason": "轨迹点不足", "features": {}}

    # 共现时长 (秒)
    t_a_start = float(traj_a[0][2])
    t_a_end = float(traj_a[-1][2])
    t_b_start = float(traj_b[0][2])
    t_b_end = float(traj_b[-1][2])
    overlap = max(0.0, min(t_a_end, t_b_end) - max(t_a_start, t_b_start))
    min_seconds = min_frames * 0.04
    if overlap < min_seconds:
        return {"pass_filter": False, "reason": f"共现时长 {overlap:.1f}s < {min_seconds:.1f}s", "features": {}}

    # 同步帧距离
    distances: list[float] = []
    for pa in traj_a:
        for pb in traj_b:
            if abs(pa[2] - pb[2]) < 0.08:
                distances.append(math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2))
                break
    if not distances:
        return {"pass_filter": False, "reason": "无同步帧", "features": {}}

    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances)) if len(distances) > 1 else 0.0

    if mean_dist > max_distance_px:
        return {"pass_filter": False, "reason": f"距离 {mean_dist:.0f}px > {max_distance_px}px", "features": {}}
    if mean_dist < 10 and std_dist < 5 and overlap < 0.5:
        return {"pass_filter": False, "reason": "极短擦肩", "features": {}}

    # 轨迹方向余弦相似度
    vec_a = np.array([traj_a[-1][0] - traj_a[0][0], traj_a[-1][1] - traj_a[0][1]])
    vec_b = np.array([traj_b[-1][0] - traj_b[0][0], traj_b[-1][1] - traj_b[0][1]])
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    traj_cosine = float(np.dot(vec_a, vec_b) / max(norm_a * norm_b, 1e-6))

    # 身体朝向
    ori_a = _compute_orientation(kps_a)
    ori_b = _compute_orientation(kps_b)
    ori_diff = _orientation_diff(ori_a, ori_b) if (ori_a is not None and ori_b is not None) else None

    features = {
        "distance_mean": round(mean_dist, 1),
        "distance_std": round(std_dist, 1),
        "trajectory_cosine": round(traj_cosine, 3),
        "orientation_diff": ori_diff,
        "co_present_seconds": round(overlap, 1),
    }

    # 同向同行
    if traj_cosine > 0.85 and mean_dist < 150 and std_dist < 40:
        return {"pass_filter": True, "reason": "同向同行候选", "features": features}

    # 面对面 / 相近朝向
    if ori_diff is not None and ori_diff < 45 and mean_dist < 180:
        return {"pass_filter": True, "reason": "面对面/相近朝向候选", "features": features}

    # 相向擦肩 — 直接判定无需 VLM
    if traj_cosine < -0.5:
        return {"pass_filter": False, "reason": "相向擦肩", "features": features}

    return {"pass_filter": False, "reason": "未达交互阈值", "features": features}


# ── 图像处理 ──────────────────────────────────────────────────

def encode_frame_jpeg(frame: np.ndarray, quality: int = 75) -> str:
    """numpy BGR 帧 → base64 JPEG 字符串."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def crop_roi(
    frame: np.ndarray,
    bbox_a: list[float],
    bbox_b: list[float],
    padding: float = 0.3,
) -> np.ndarray | None:
    """裁剪两人同框区域."""
    h, w = frame.shape[:2]
    bw_a = bbox_a[2] - bbox_a[0]
    bw_b = bbox_b[2] - bbox_b[0]
    pad_a = bw_a * padding
    pad_b = bw_b * padding
    x1 = max(0, int(min(bbox_a[0], bbox_b[0]) - pad_a))
    y1 = max(0, int(min(bbox_a[1], bbox_b[1]) - pad_a))
    x2 = min(w, int(max(bbox_a[2], bbox_b[2]) + pad_b))
    y2 = min(h, int(max(bbox_a[3], bbox_b[3]) + pad_b))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# ── VLM 客户端 ────────────────────────────────────────────────

class VLMInteractionClient:
    """通过 Ollama API 调用 Qwen3-VL 进行交互分析."""

    def __init__(self, model: str = "yemifo/qwen25-vl-3b-instruct4k:latest", timeout: float = 30.0):
        import requests as _requests
        self.model = model
        self.timeout = timeout
        self.base_url = "http://127.0.0.1:11434"

    def analyze_pair(
        self,
        encoded_image: str,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """送一张两人同框截图 + 几何特征，让 VLM 判断交互类型."""
        import requests as _requests

        feature_text = (
            f"距离均值: {features.get('distance_mean', '?')}px, "
            f"距离标准差: {features.get('distance_std', '?')}px, "
            f"轨迹方向余弦: {features.get('trajectory_cosine', '?')}, "
            f"朝向差: {features.get('orientation_diff', '?')}°, "
            f"共存: {features.get('co_present_seconds', '?')}s"
        )

        prompt = (
            "你是视频监控交互分析助手。看这张两人同框截图，结合以下时空特征，判断交互类型：\n\n"
            f"时空特征：{feature_text}\n\n"
            "选项：结伴同行 (同向移动距离稳定) / 面对面交谈 (面向对方距离近) / "
            "短暂接触 (擦肩或短暂靠近) / 无交互 (只是同屏)\n\n"
            "只输出 JSON：\n"
            '{"interaction_type": "...", "confidence": 0.xx, "reasoning": "..."}'
        )

        try:
            resp = _requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [encoded_image],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200},
                },
                timeout=self.timeout,
            )
            data = resp.json()
            raw = data.get("response", "")
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
            return {"interaction_type": "未知", "confidence": 0.3, "reasoning": raw[:200]}
        except Exception as e:
            return {"interaction_type": "未知", "confidence": 0.0, "reasoning": str(e)[:200]}
