"""姿态关键点必须按 track_id 对齐，不能按下标。"""

from perception.pose_align import apply_pose_keypoints, build_pose_map, prune_pose_map


def keypoints(seed: float):
    return [[seed, seed + 1.0, 0.9] for _ in range(17)]


def tracklet(track_id):
    return {"track_id": track_id, "bbox": [0, 0, 10, 10]}


def test_pose_follows_track_id_when_order_changes():
    # 第 0 帧：顺序是 3、7，姿态只给这两条轨迹算过
    tracked_frame0 = [tracklet(3), tracklet(7)]
    pose_map = build_pose_map(tracked_frame0, [keypoints(10.0), keypoints(70.0)])

    # 第 1 帧：tracked 顺序反了（3 号丢失后回来、7 号排到前面）
    tracked_frame1 = [tracklet(7), tracklet(3)]
    apply_pose_keypoints(tracked_frame1, pose_map)

    assert tracked_frame1[0]["pose_keypoints"] == keypoints(70.0)
    assert tracked_frame1[1]["pose_keypoints"] == keypoints(10.0)


def test_pose_map_skips_empty_keypoints_and_missing_track():
    pose_map = build_pose_map([tracklet(1), tracklet(2)], [keypoints(1.0), []])

    tracked = [tracklet(1), tracklet(2), tracklet(3)]
    apply_pose_keypoints(tracked, pose_map)

    assert tracked[0]["pose_keypoints"] == keypoints(1.0)
    assert tracked[1]["pose_keypoints"] == []
    assert tracked[2]["pose_keypoints"] == []


def test_prune_pose_map_drops_invisible_tracks():
    pose_map = {1: keypoints(1.0), 2: keypoints(2.0)}

    pruned = prune_pose_map(pose_map, [tracklet(2)])

    assert pruned == {2: keypoints(2.0)}
