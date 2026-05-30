# 设计 4/5：检测与跟踪优化边界

第一阶段不换 tracker，不上高级 ReID。目标是减少小人物漏检、遮挡断 ID 对连续理解的破坏。

## 0 — Tracker 参数层

统一 web/CLI 参数入口：

| 参数 | 作用 | 默认调整方向 |
|------|------|------------|
| `conf` | 检测置信度阈值 | 适当降低，减少小目标漏检 |
| `imgsz` | 输入图像尺寸 | 提高，增强小目标分辨率 |
| `track_activation_threshold` | 新 track 激活阈值 | 配合 conf 调整 |
| `new_track_thresh` | 新 tracker 初始化条件 | 保守，避免碎片过多 |
| `lost_track_buffer` | 遮挡容忍帧数 | 延长，降低 ID switch |
| `ReID 开关` | supervision 内置 ReID | 按场景评估开启 |
| `GMC 开关` | 全局运动补偿 | 动态相机场景开启 |

## 1 — Entity 融合门控

遮挡后新 track 出现时，保守评分决定是否并入已有 Person：

- **Frame gap**：帧间隔越短越安全
- **Predicted position**：Kalman 预测位置 vs 新 track 起始位置
- **BBox shape**：高度/宽度连续性
- **Appearance**：现有 ResNet18（supervision 内置）外观特征

高置信 → 自动并入同一 Person；低置信 → 新建 Person，记录 tentative candidate，后续证据再确认。

## 2 — 质量标注

每个 Person/segment 记录 `tracking_quality`：

| 标注值 | 含义 |
|--------|------|
| `stable` | 连续跟踪，无遮挡 |
| `occluded` | 经遮挡后 ReID 恢复 |
| `fragmented` | 跟踪断裂，片段化 |
| `tentative` | 低置信新建，待确认 |

Agent 和 UI 读此字段，避免把检测断裂误判为行为异常。

## 明确不做

- 不替换成完整 BoT-SORT / Deep-OC-SORT
- 不训练或引入高级 ReID 模型
- 不做多摄像头身份重识别
- 现有 tracker 输出保持兼容，EntityProfile 在上层吸收断裂

## 验证

- MOT17 / 现有上传样本验证
- ID switch 数量、fragmented Person 数、平均 track segment 数
- 生命线缺口是否正确标注
- UI 必须能看出"遮挡后重新接上"的虚线
