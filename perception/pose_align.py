"""姿态关键点与轨迹的对齐工具。

姿态估计每 N 帧才算一次、中间帧复用结果；而 tracked 列表的顺序会因
lost / 新增轨迹而变化，按下标回填会把骨架画到别人身上。这里统一以
`track_id` 为键，中间帧按 ID 查回填。
"""

from __future__ import annotations


def build_pose_map(tracked: list[dict], keypoints_list: list) -> dict:
    """把一次姿态估计结果按 track_id 建索引。"""
    mapping: dict = {}
    for tracklet, keypoints in zip(tracked, keypoints_list):
        if keypoints:
            mapping[tracklet.get("track_id")] = keypoints
    return mapping


def apply_pose_keypoints(tracked: list[dict], pose_map: dict) -> list[dict]:
    """按 track_id 给当前帧的每条轨迹回填姿态关键点（缺失则空列表）。"""
    for tracklet in tracked:
        tracklet["pose_keypoints"] = pose_map.get(tracklet.get("track_id"), [])
    return tracked


def prune_pose_map(pose_map: dict, tracked: list[dict]) -> dict:
    """丢掉当前帧不可见的轨迹缓存，避免旧关键点被回填到重新出现的轨迹上。"""
    visible = {tracklet.get("track_id") for tracklet in tracked}
    return {track_id: keypoints for track_id, keypoints in pose_map.items() if track_id in visible}
