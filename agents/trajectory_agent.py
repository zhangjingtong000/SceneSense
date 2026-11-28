"""
轨迹分析 Agent。

分析个体目标的运动模式。支持统计自适应速度归一化 + 规则快速路径 + LLM 边界补充。
"""

import math
import time
from collections import deque

from .base_agent import BaseAgent
from .llm_client import LLMClient


SPEED_LEVEL_LABELS = {
    "stationary": "静止", "slow": "缓慢", "normal": "正常",
    "fast": "快速", "running": "奔跑", "unknown": "未知",
}


def normalize_image_speed(tracklet: dict, speed_px_s: float) -> float:
    bbox = tracklet.get("bbox") or []
    if len(bbox) != 4:
        return 0.0
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    return max(0.0, float(speed_px_s)) / height


def classify_normalized_speed(normalized_speed: float, normalizer: "VelocityNormalizer") -> str:
    if normalized_speed <= normalizer.stationary_threshold:
        return "stationary"
    if normalized_speed <= normalizer.loitering_threshold:
        return "slow"
    if normalized_speed + 1e-9 >= normalizer.running_threshold * 1.5:
        return "running"
    if normalized_speed + 1e-9 >= normalizer.running_threshold:
        return "fast"
    return "normal"


class VelocityNormalizer:
    """统计自适应速度归一化器。

    维护滑动窗口收集场景中所有目标的归一化速度观察值，
    用百分位动态计算低速/中速/高速阈值，自动适配不同相机距离和分辨率。
    """

    def __init__(
        self,
        window_size: int = 500,
        stationary_percentile: float = 0.15,
        loitering_percentile: float = 0.35,
        running_percentile: float = 0.85,
        min_stationary_speed: float = 0.03,
        min_loitering_speed: float = 0.12,
        min_running_speed: float = 0.75,
    ):
        self._window: deque[float] = deque(maxlen=window_size)
        self.stationary_p = stationary_percentile
        self.loitering_p = loitering_percentile
        self.running_p = running_percentile
        self.min_stationary = min_stationary_speed
        self.min_loitering = min_loitering_speed
        self.min_running = min_running_speed
        self._sorted: list[float] = []
        self._dirty = True

    def observe(self, speed: float):
        """记录一个人物框高度归一化后的速度观察值。"""
        self._window.append(speed)
        self._dirty = True

    def observe_batch(self, speeds: list[float]):
        """批量记录速度观察值。"""
        for s in speeds:
            self._window.append(s)
        self._dirty = True

    def _rebuild(self):
        if not self._dirty:
            return
        self._sorted = sorted(self._window)
        self._dirty = False

    @property
    def stationary_threshold(self) -> float:
        self._rebuild()
        if len(self._sorted) < 10:
            return self.min_stationary
        idx = int(len(self._sorted) * self.stationary_p)
        return max(self._sorted[idx], self.min_stationary)

    @property
    def loitering_threshold(self) -> float:
        self._rebuild()
        if len(self._sorted) < 10:
            return self.min_loitering
        idx = int(len(self._sorted) * self.loitering_p)
        return max(self._sorted[idx], self.min_loitering)

    @property
    def running_threshold(self) -> float:
        self._rebuild()
        if len(self._sorted) < 10:
            return self.min_running
        idx = int(len(self._sorted) * self.running_p)
        return max(self._sorted[idx], self.min_running)

    @property
    def running_percentile(self) -> float:
        return self.running_p * 100

    @property
    def median_speed(self) -> float:
        self._rebuild()
        if len(self._sorted) < 10:
            return 5.0
        return self._sorted[len(self._sorted) // 2]

    @property
    def sample_count(self) -> int:
        return len(self._window)


class TrajectoryAgent(BaseAgent):
    """分析个体目标的运动模式。

    输入：SpatioTemporalEvent 中的 tracklets（轨迹+速度+方向）
    输出：每个目标的 motion_pattern 分类结果
    """

    def __init__(self, llm_client: LLMClient) -> None:
        # 纯规则 Agent：不调模型，因此没有 system prompt（原来的 prompts/trajectory.md 已删）
        super().__init__("trajectory_agent", llm_client, "")
        self.normalizer = VelocityNormalizer()

    def think(self) -> dict:
        """对每个活跃目标生成结构化运动数据。"""
        tracklets = self.context.get("tracklets", [])

        all_speeds: list[float] = []
        for t in tracklets:
            spd = self._normalized_speed(t, float(t.get("velocity_ms", 0) or 0))
            if spd > 0:
                all_speeds.append(spd)
        if all_speeds:
            self.normalizer.observe_batch(all_speeds)

        findings: list[dict] = []
        warnings: list[str] = []
        confidences: list[float] = []

        for tracklet in tracklets:
            track_id = tracklet.get("track_id", 0)
            entity_id = str(tracklet.get("entity_id") or f"Entity-{track_id}")

            data = self._describe_motion(tracklet)
            pose_label = self._describe_pose(tracklet.get("pose_keypoints", []))
            data["pose"] = pose_label

            # 旧版 text 字段给 summary_agent 用
            desc_text = data.get("description", data.get("label", "未知"))
            if pose_label:
                desc_text = desc_text + "。" + pose_label

            if len(tracklet.get("trajectory", [])) < 3:
                warnings.append(f"{entity_id} 轨迹点不足")

            finding = {
                "entity_id": entity_id,
                "finding_type": "motion_description",
                "value": data,
                "confidence": round(data.get("confidence", 0.5), 3),
                "evidence": data.get("evidence", []),
                "kg_operations": [{
                    "op": "upsert_node", "node_type": "Entity", "node_id": entity_id,
                    "properties": {
                        "motion_description": desc_text,
                        "motion_label": data.get("label", "unknown"),
                        # 下面几项是 RuleEngine 的匹配依据。不写的话规则引擎即使被调用，
                        # 也只会看到 None 然后一条都匹配不上（此前就是这样空转的）。
                        "motion_profile": data.get("motion_profile", "unknown"),
                        "velocity": data.get("normalized_speed", 0.0),
                        "direction_change_count": data.get("direction_change", 0),
                        "first_seen": tracklet.get("first_seen_ts", time.time()),
                        "class": tracklet.get("class", tracklet.get("category", "person")),
                        "active": True,
                        "last_analyzed": time.time(),
                    },
                }],
            }
            findings.append(finding)
            confidences.append(data.get("confidence", 0.5))

        agent_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        return {"agent_confidence": round(agent_confidence, 3), "warnings": warnings, "findings": findings}

    _LABEL_MAP = {
        "stationary": "静止", "slow": "缓慢", "loitering": "徘徊",
        "walking": "行走", "walking_fast": "快速", "running": "奔跑",
        "erratic": "无规则", "unknown": "未知",
    }
    _SPEED_LEVEL_MAP = SPEED_LEVEL_LABELS

    @staticmethod
    def _normalized_speed(tracklet: dict, speed_px_s: float) -> float:
        return normalize_image_speed(tracklet, speed_px_s)

    def _describe_motion(self, tracklet: dict) -> dict:
        """返回结构化运动数据。"""
        trajectory = tracklet.get("trajectory", [])
        velocity_ms = tracklet.get("velocity_ms", 0.0)
        evidence: list[str] = []
        n = self.normalizer

        if len(trajectory) < 2:
            return {"label": "未知", "confidence": 0.3, "evidence": ["轨迹点<2"],
                    "speed_px": 0, "normalized_speed": 0, "speed_pct": 50,
                    "speed_level": "未知", "direction_change": 0, "pose": "",
                    "duration": 0, "motion_profile": "unknown"}

        speeds: list[float] = []
        dirs: list[float] = []
        for i in range(1, len(trajectory)):
            x0, y0 = trajectory[i-1][0], trajectory[i-1][1]
            x1, y1 = trajectory[i][0], trajectory[i][1]
            dx, dy = x1 - x0, y1 - y0
            if len(trajectory[i - 1]) >= 3 and len(trajectory[i]) >= 3:
                dt = float(trajectory[i][2]) - float(trajectory[i - 1][2])
            else:
                dt = 0.0
            speeds.append(math.sqrt(dx*dx + dy*dy) / dt if dt > 0 else math.sqrt(dx*dx + dy*dy))
            dirs.append(math.degrees(math.atan2(dy, dx)) % 360)

        time_span = max(float(trajectory[-1][2]) - float(trajectory[0][2])
                        if len(trajectory[-1]) >= 3 and len(trajectory[0]) >= 3
                        else len(trajectory) * 0.04, 0.1)
        mean_speed = sum(speeds) / len(speeds) if speeds else 0
        effective_speed = velocity_ms if velocity_ms > 0 else mean_speed
        normalized_speed = self._normalized_speed(tracklet, effective_speed)

        if len(dirs) >= 3:
            rads = [math.radians(d) for d in dirs]
            ms, mc = sum(math.sin(r) for r in rads)/len(rads), sum(math.cos(r) for r in rads)/len(rads)
            straightness = math.sqrt(ms*ms + mc*mc)
            dir_spread = 0.0 if straightness >= 0.99 else math.degrees(math.sqrt(-2*math.log(max(straightness, 1e-6))))
        else:
            straightness, dir_spread = 1.0, 0.0
        # 偏离次数（> 45° 的帧间跳变）
        changes = sum(1 for i in range(1, len(dirs))
                      if min(abs(dirs[i]-dirs[i-1]), 360-abs(dirs[i]-dirs[i-1])) > 45)

        total_disp = math.sqrt((trajectory[-1][0]-trajectory[0][0])**2 +
                                (trajectory[-1][1]-trajectory[0][1])**2)
        v_stat, v_loiter, v_run = n.stationary_threshold, n.loitering_threshold, n.running_threshold

        # 百分位：用阈值插值估算
        if n.sample_count < 10:
            pct = 50
        elif normalized_speed <= v_stat:
            pct = max(0, int(n.stationary_p * 100 * (normalized_speed / max(v_stat, 0.001))))
        elif normalized_speed >= v_run:
            pct = min(100, int(100 - (100 - n.running_p * 100) * (1 - (normalized_speed - v_run) / max(normalized_speed - v_run + 0.5, 0.01))))
        else:
            ratio = (normalized_speed - v_loiter) / max(v_run - v_loiter, 0.001)
            pct = int(n.loitering_p * 100 + ratio * (n.running_p * 100 - n.loitering_p * 100))

        speed_level = classify_normalized_speed(normalized_speed, n)

        label, conf = "unknown", 0.5
        if speed_level == "running" and len(trajectory) >= 5:
            label, conf = "running", 0.88
        elif speed_level == "fast" and len(trajectory) >= 5:
            label, conf = "walking_fast", 0.85
        elif total_disp < 10 and normalized_speed <= v_stat and len(trajectory) >= 10:
            label, conf = "stationary", 0.85
        elif normalized_speed <= v_stat and len(trajectory) >= 5:
            label, conf = "stationary", 0.85
        elif normalized_speed <= v_loiter and dir_spread > 60 and len(trajectory) >= 10:
            label, conf = "loitering", 0.85
        elif dir_spread > 80 and total_disp > 50:
            label, conf = "erratic", 0.80
        elif normalized_speed <= v_loiter:
            label, conf = "slow", (0.70 if len(trajectory) >= 10 else 0.55)
        else:
            label, conf = "walking", 0.85

        evidence.append(
            f"speed={effective_speed:.1f}px/s normalized={normalized_speed:.3f}/s "
            f"median={n.median_speed:.3f}/s spread={dir_spread:.0f}°"
        )
        return {
            "label": self._LABEL_MAP.get(label, label),
            # 英文键留给规则引擎用：`RuleEngine` 的 conditions["motion_profile"]
            # 比对的是英文标签（stationary/loitering/walking/...），不是中文显示名。
            "motion_profile": label,
            "confidence": conf,
            "speed_px": round(effective_speed, 1),
            "normalized_speed": round(normalized_speed, 3),
            "speed_pct": pct,
            "speed_level": self._SPEED_LEVEL_MAP.get(speed_level, speed_level),
            "direction_change": changes,
            "pose": "",
            "duration": round(time_span, 1),
            "evidence": evidence,
        }

    def _describe_pose(self, keypoints: list[list[float]]) -> str:
        """从 COCO 17 关键点生成姿态描述。"""
        if not keypoints or len(keypoints) < 17:
            return ""

        # COCO 索引: 5=左肩, 6=右肩, 7=左肘, 8=右肘, 9=左腕, 10=右腕
        # 11=左臀, 12=右臀, 13=左膝, 14=右膝, 15=左踝, 16=右踝
        def conf(idx): return keypoints[idx][2] if len(keypoints[idx]) > 2 else 0
        def below(a, b): return conf(a) > 0.5 and conf(b) > 0.5 and keypoints[a][1] > keypoints[b][1]

        parts: list[str] = []

        # 手腕位置
        lw = conf(9); rw = conf(10)
        if lw > 0.5 and rw > 0.5:
            lw_y, rw_y = keypoints[9][1], keypoints[10][1]
            le_y, re_y = keypoints[7][1], keypoints[8][1]
            ls_y, rs_y = keypoints[5][1], keypoints[6][1]
            if lw_y > le_y and rw_y > re_y:
                parts.append("双手下垂")
            elif lw_y < ls_y and rw_y < rs_y:
                parts.append("双臂高举")
            elif lw_y < le_y and rw_y > re_y:
                parts.append("左臂抬起、右手下垂")
            elif lw_y > le_y and rw_y < re_y:
                parts.append("左手下垂、右臂抬起")

        # 蹲姿/坐姿（髋低于膝）
        if conf(11) > 0.5 and conf(13) > 0.5 and keypoints[11][1] > keypoints[13][1]:
            parts.append("呈蹲姿或坐姿" if conf(12) > 0.5 and keypoints[12][1] > keypoints[14][1] else "单膝弯曲")
        elif conf(11) > 0.5 and conf(13) > 0.5 and conf(12) > 0.5 and conf(14) > 0.5:
            if below(13, 15) and below(14, 16) and keypoints[13][1] > keypoints[15][1] + 30:
                parts.append("双腿直立")

        if not parts:
            # 有关键点但无明显姿态特征
            valid = sum(1 for kp in keypoints if len(kp) > 2 and kp[2] > 0.5)
            if valid >= 10:
                parts.append("站姿端正")
            elif valid >= 5:
                parts.append("姿态可辨识")

        return "，".join(parts) if parts else ""
