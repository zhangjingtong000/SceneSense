# 连续实体理解与前端知识图谱展示优化设计

日期：2026-05-29

## 背景

当前系统按固定时间窗口生成 `SpatioTemporalEvent`，再由多 Agent 对每个窗口独立分析。这个结构能产出分段摘要，但会打断同一实体在整段视频中的连续轨迹和交互历史。前端也以事件片段为主，知识图谱只显示计数徽章，用户无法直观看到 KG 如何支撑推理。

第一阶段目标是在不推翻现有协议和 tracker 的前提下，建立稳定的全视频实体锚点，并让前端围绕实体生命线和证据子图展示分析结果。

## 不变约束

- `SpatioTemporalEvent` 格式不变。
- `AgentMessage` / `AgentResponse` 格式不变。
- KG Schema 不变，仍使用 Entity / Event / Behavior / Scene 节点和 belongs_to / interacts_with / matches / next / violates 边。
- 不替换 tracker 方案，不切到完整 BoT-SORT / Deep-OC-SORT。
- 不引入外挂高级 ReID 网络；只使用当前 tracker 内已有 appearance feature。
- 不做通用 KG 查询语言，只做预置过滤器和受控问答入口。
- 时间轴统一以 frame number 为唯一时间基准，秒数仅用于显示。
- 页面全局中文；布局和关键控件需要支持自适应缩放与响应式显示。

## 总体方案

按分层推进，但每层先做最小可用闭环：

- Layer 0：tracker 参数与一致性修正。
- Layer 1：EntityProfile 融合层，维护 `track_id -> Person-*` 映射。
- Layer 2：KG Entity 节点自动维护与跨段关系写入。
- Layer 3：双层时间轴，事件热力为导航辅助，实体生命线为主层。
- Layer 4：KG 证据子图面板，点击实体或事件展示 1-2 跳证据与推理路径。

Layer 0 和 Layer 1 可以并行；Layer 2 依赖 Layer 1；Layer 3 和 Layer 4 依赖 Layer 2。

## Layer 1：EntityProfile 连续实体层

新增 `EntityProfile` 作为 Event 之上的运行时连续实体档案。UI、KG 和跨段推理主用稳定的 `Person-*`，原始 `track_id` 作为证据保留。

建议结构：

```text
EntityProfile
- entity_id: "Person-1"
- track_ids: [3, 17]
- segments: [{track_id, frame_start, frame_end, confidence}]
- global_trajectory: sampled points across segments
- occlusion_gaps: [{from_frame, to_frame, reason, reid_score}]
- merge_evidence: [{track_a, track_b, score, factors}]
- tentative_merge_candidates: ["Person-3"]
- interaction_history: summarized pairs over frame ranges
- tracking_quality: stable | occluded | fragmented | tentative
```

合并策略：

- 高置信：自动合并到已有 `Person-*`。
- 低置信：新建 `Person-*`，并记录 `tentative_merge_candidates`。
- 合并证据包括 frame gap、预测位置、bbox 尺寸连续性、方向一致性、现有 tracker appearance feature。
- 任何合并都必须保留原始 track 段、帧范围、分数和判断因素。

## Layer 2：KG 连续写入

KG 不作为完整轨迹时序数据库使用。`Entity` 节点只存可查询摘要，完整轨迹点留在 EntityProfile 或运行结果中，避免 GraphML 膨胀。

Entity 节点建议属性：

- `entity_id`
- `track_ids`
- `first_seen_frame`
- `last_seen_frame`
- `trajectory_summary`：速度均值/方差、主要方向、活动区域 bounding box
- `tracking_quality`
- `merge_evidence_summary`

写入策略：

- 每个事件窗口写入 `Event` 节点，包含 frame range、duration、entity_count、summary、expires_at。
- 用 `next` 边串联相邻 Event。
- 用 `belongs_to` 连接 `Person-* -> Event-*`，记录 role、frame range、distance_to_center 等摘要证据。
- 用 `interacts_with` 连接 `Person-* -> Person-*`，在边属性中累计 `event_ids`、`frame_ranges`、`duration`、`confidence`、`evidence`。
- `matches` / `violates` 继续用于行为模板和异常证据，证据必须能追溯到 Person、Event、frame range 和原始 track。

第一版沿用当前 NetworkX 图类型，把交互历史聚合在同一条边属性里。后续如需多条同类边，再考虑 MultiDiGraph 或 Interaction 节点。

## Layer 3：双层时间轴与中文自适应 UI

前端主布局改成视频工作台：

- 左侧主区：视频画面、当前选中 Person 高亮、轨迹覆盖。
- 视频下方：双层时间轴。
- 右侧面板：实体详情、KG 证据子图、推理路径、预置过滤器。

双层时间轴：

- 上层事件热力：用于快速定位高密度、高交互、高异常窗口。
- 下层实体生命线：主叙事层，每个 `Person-*` 一行。
- 遮挡或断裂用虚线缺口表示。
- 事件热力和生命线使用同一 frame scale，Event 只是叠加着色区间。

交互：

- 点击事件热力：跳转视频到对应帧区间，并将生命线滚动到该时间范围。
- 点击生命线：选中 `Person-*`，视频高亮 bbox/轨迹，右侧显示证据子图。
- 点击 KG 子图节点：反向高亮时间轴和视频证据。

前端要求：

- 页面文字、按钮、状态、提示、过滤器、错误信息全局中文。
- 桌面端优先，但要支持宽度变化时的自适应缩放。
- 关键 UI 不依赖固定像素宽度；时间轴、视频、右侧面板使用响应式约束。
- 文本不得溢出控件；长内容折行或收进可滚动面板。

## Layer 4：KG 证据子图面板

KG 展示以证据子图为主，不以全局大图为主。

主流程：

- 点击 Person 或 Event 后，右侧显示相关 1-2 跳子图。
- 子图高亮当前实体、关键边、行为模板和异常路径。
- 每条边显示 frame range、confidence、source agent、evidence summary。
- 推理路径以中文步骤展示，例如：低速停留 → 重复接近 Person-2 → 匹配徘徊模板。

预置过滤器：

- 按实体 ID。
- 按交互类型。
- 按异常等级。
- 按 tracking_quality。
- 按是否存在遮挡断点。

问答入口限定在 KG 可回答问题，例如“这个人跟谁交互过”“这个事件涉及哪些实体”“这个异常证据来自哪些帧”。不开放任意自然语言查询范围。

## Layer 0：检测与跟踪优化边界

第一阶段只做参数和一致性修正：

- 统一 web/CLI 配置入口：`conf`、`imgsz`、`track_activation_threshold`、`new_track_thresh`、`lost_track_buffer`、ReID/GMC 开关。
- 针对小目标默认提高 `imgsz`、适度降低检测阈值、延长 lost buffer。
- CLI 路径需要像 web 路径一样向 tracker 传入 `frame`，确保已有 ReID/GMC 能启用。
- 每个 segment 标注 `tracking_quality`，避免把遮挡或断 ID 误判成行为异常。

明确不做：

- 不替换 tracker。
- 不训练新 ReID。
- 不引入外部身份系统。
- 不让低置信合并污染主 `Person-*` 链路。

## WebSocket / 前端数据

保留现有事件推送，同时增加轻量连续层数据：

- `entities`：当前全局 Person 摘要。
- `lifelines`：用于时间轴渲染的 frame segments。
- `kg_subgraph`：当前选中实体/事件的子图数据。
- `tracking_quality`：实体或 segment 的质量标注。

这些数据是展示增强，不改变底层 Agent 协议。

## 验收标准

- 同一人遮挡后重现时，UI 主显示仍为同一 `Person-*`；断点用虚线表示；原始 `track_id` 可追溯。
- 事件热力和实体生命线严格按 frame number 对齐，跳转 Event 不漂移。
- 点击 Person 能看到 1-2 跳 KG 证据子图，边上有帧范围、置信度、source agent、原始 track 证据。
- KG `Entity` 节点只存轨迹统计摘要，完整轨迹不写入 GraphML 节点属性。
- 页面全局中文，主要布局在不同宽度下可用，文本不明显溢出。
- 预置过滤器覆盖交互类型、异常等级、实体 ID、tracking_quality。
- MOT17 或现有样本中，ID 断裂可被标注或合并，生命线能表达遮挡和重现。
- tracker 参数调整需要有量化记录：在同一验证视频上统计检测漏检数、ID switch 数、平均每个 Person 的 track segment 数；第一阶段要求这些指标不劣于当前基线，并以 ID switch 或 segment fragmentation 相对下降作为调参目标。

## 非目标

- 不做全局大图作为主要工作区；全局 KG 可作为调试或导出视图。
- 不做任意自然语言 KG 查询。
- 不做多摄像头重识别。
- 不把完整轨迹序列塞进 KG 节点属性。
- 不重写前端为大型框架，除非后续复杂度明显超过单文件 HTML 可维护范围。
