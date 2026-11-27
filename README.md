# SceneSense
输入一段监控视频（文件 / 摄像头 / 图片序列），输出带 `track_id` 的逐帧轨迹、每 30 帧切出的时空事件、中文场景报告，以及一份可用 GraphML 打开、可问答的场景知识图谱。

![SceneSense 分层架构](docs/architecture.png)

> 上图是感知层 → 实体层 → 事件层 → Agent 层 → KG 层的分层与数据流。Mermaid 源仍在 [`docs/architecture.md`](docs/architecture.md)（三段：整体架构 / 时序 / KG Schema）；重绘这张 PNG 用 `python docs/gen_architecture_png.py docs/architecture.png`（matplotlib，不依赖 Node）。

## 它解决什么问题

监控视频的日常用法是"事后找那一段"：谁在几点进了哪个区域、和谁待了多久、有没有异常。逐帧检测器回答不了——它每帧独立，人一被遮挡就换 ID，事件之间也串不起来。SceneSense 把检测结果往上收成三层：稳定的 Entity（身份）、时间窗内的 Event（发生了什么）、可查询的图（怎么关联），回看视频变成查图。

## 怎么做的

### 1. 感知：检测 + 姿态

YOLOv8 出 bbox / 类别 / 置信度，YOLOv8-pose 单独出 17 个 COCO 关键点，作为可选信号注入轨迹。两阶段解耦：换检测模型不影响姿态，关掉姿态也不影响跟踪。

### 2. 跟踪 `perception/tracker.py`

BoT-SORT 式结构，四个部件各解决一件事：

- **GMC**：相邻帧稀疏光流（`goodFeaturesToTrack` → `calcOpticalFlowPyrLK` → RANSAC 求 2×3 仿射），先减掉背景整体位移再更新卡尔曼状态。没有这一步，手持或云台视频里静止的人会被算成高速移动。
- **卡尔曼**：8 状态 `(cx, cy, w, h, vx, vy, vw, vh)`，4 维观测；轨迹丢失期间靠它预测位置。
- **ReID**：OSNet x0.25（MSMT17 预训练）出 512 维 L2 归一化特征。BoxMOT 与 torchreid 两条 OSNet 链路，都不可用时回退 ImageNet ResNet18，此时 ID 连续性明显变差。
- **组合距离**：先算 `1 - IoU`，再用 IoU 门控外观——IoU 低于 0.5 的配对不看外观，外观余弦距离大于 0.25 直接置 1，两者取 `min` 作为匈牙利代价。

**为什么遮挡必须上 ReID**：人被挡住时 IoU 会掉到 0，纯 IoU 在这类帧上无解，只能新建 ID，遮挡结束就变成"两个人"。但只用外观也不行——同款工服、相似身高的人会互相串号。所以这里先用 IoU 圈出位置还说得过去的候选，再让外观在候选内部做区分，而不是让外观全局找人。

**低分框阶段故意不用外观**：遮挡后重现的人往往只剩低分检测，代码第二阶段只用 IoU（阈值 0.5）。跟踪器注释记录了实测结论——这一阶段放外观会引入错误关联，恢复率反而下降。lost 轨迹缓冲 120 帧后才丢弃。

### 3. 实体 `perception/entity_profile.py`

tracklet 会断，Entity 不该断。按 0.35 位置 + 0.30 时间 + 0.35 外观加权打分：≥0.72 自动合并，0.30–0.72 记为 tentative 候选并逐帧累积证据。外观余弦 >0.90 时允许突破 300 帧的 gap 上限，用于长时间被立柱、货架遮住再出现的人。位置分取"最后位置"和"按速度外推位置"中更近的那个；`pipeline.py` 启动时还会把位置门限抬到画面宽的 40%，否则 1920 画面里跨几秒遮挡的位移会被判成新实体。

### 4. 事件 `perception/event_builder.py`

每 30 帧（`--event-interval`）切一个 `SpatioTemporalEvent`：tracklet 按 `track_id` 去重、保留轨迹最全的一条，计算空间近邻关系后交给 Agent 层。事件既是 Agent 的输入粒度，也是 KG 里 Event 节点的来源。

### 5. 编排 `agents/orchestrator.py`：星型

中心节点按 `trajectory → interaction → summary` 串行路由，每一步把前序所有 Agent 的 findings 塞进下一步上下文；结束后由中心统一收集 `kg_operations` 写入 GraphStore。**为什么不是链式**：链式 A→B→C 里 C 只能看到 B 的输出，想加第四个分析维度就得改链路；星型中心自己持有全部 `responses`，Agent 只依赖中心传进来的 context，因此可以单独替换或追加（`anomaly_agent.py` 就是这样独立写出来的，不碰路由代码），`reasoning_trail` 也能完整记下每个 Agent 的输入摘要和产出。

**四个 Agent 怎么选**：默认路由仍是三个（`trajectory → interaction → summary`）；`python pipeline.py --with-anomaly …` 会把 `anomaly_agent` 按 `position=2` 插进 `Orchestrator.route_order`，形成 `trajectory → interaction → anomaly → summary`——异常评分先产出，总结阶段就能读到它。开关只动注册与路由顺序，四个 Agent 的内部逻辑一行没改。

### 6. 四个 Agent

| Agent | 职责 | 关键取舍 |
| --- | --- | --- |
| `trajectory_agent` | 个体运动模式 | 速度先除以人体框高度做尺度归一化，再按场景分位数分 stationary/slow/normal/fast/running；规则先出结论，LLM 只补规则覆盖不到的边界帧 |
| `interaction_agent` | 两两交互 | 轨迹按跳变点切段后做时间对齐，提取距离均值/方差、轨迹余弦、共现时长，分成结伴同行 / 面对面交谈 / 短暂接触 / 无交互 |
| `summary_agent` | 场景总述 | 只吃前两者的结构化 findings，不重新看原始轨迹，避免重复推理 |
| `anomaly_agent` | 异常行为 | 拿 Behavior 模板的关键词集合与轨迹/交互特征打分，只输出 `raw_score`，最终严重度留给协调层融合 |

### 7. 知识图谱 `kg/`

- `graph_store.py`：NetworkX DiGraph。属性统一 JSON 编解码，支持 upsert、子图、VF2 子图匹配、GraphML 存取；Event 节点带 `expires_at`，可按时间批量淘汰。
- `continuity.py`：写 Event / Entity 节点和 `belongs_to`、`interacts_with`、`next` 边。同一对实体的交互边按事件聚合，保留最近 20 条证据，`confidence = min(1, 事件数 / 5)`，`duration` 由证据累加。
- `rule_engine.py` + `kg_builder.py`：从 Behavior 节点生成规则（zone_type、速度区间、停留时长、方向变化），逐实体检查；`kg_builder` 让 LLM 产出行为模板与 Scene 节点。

## 快速开始

```bash
git clone <repo-url> scenesense && cd scenesense
python -m venv .venv
.venv\Scripts\activate            # Windows；Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**权重**：`yolov8n-pose.pt` 首次运行时 Ultralytics 自动下载；想要更好的 OSNet 特征，把 `osnet_x0_25_msmt17.pt` 放到 `models/`（或交给 BoxMOT 管理）。缺失时回退 ImageNet ResNet18，日志里会打印 `[ReID] BoxMOT 不可用（…），回退 ResNet18`——看到这行就说明 ID 连续性会变差。

**示例视频（不需要自备素材）**：`assets/demo/sample.mp4` 是仓库自带的**合成监控素材**（1280×720 / 25fps / 8s / 1.3 MB），4 个几何绘制的行人，包含交叉遮挡、一人在画面中部低速徘徊、一人走出画面再回来。**不是真实监控画面，也不含任何人像。**

```bash
python pipeline.py --source assets/demo/sample.mp4 --dry-run --annotate --annotate-path outputs/demo.mp4
```

**跑通感知层（不需要 API Key）**：

```bash
python pipeline.py --source your_video.mp4 --dry-run --annotate --annotate-path outputs/demo_annotated.mp4
```

`--source 0` 接摄像头，`--source path/to/frames/` 接图片序列目录；`--dry-run` 只跑检测/跟踪/事件/KG，跳过 LLM。

**这段示例视频跑出来的实际数字**（CPU，`--event-interval` 默认 30 帧）：8 秒切出 **6 个事件**；事件里出现过的 `track_id` 共 **7 个（2/3/5/7/8/9/11）**，`raw_tracks=12`，最终收敛成 **4 个 Entity**；每个事件内 1–3 个 tracklet、最多 1 条 `near` 空间关系；结束时 KG 有 **10 个节点（6 Event + 4 Entity）/ 19 条边**；`tracker metrics: drop_rate=0.0838 gap_rate=0.0479 id_switches=8 entities=4`。`id_switches=8` 偏高是这类合成素材的预期表现——纯色人形在交叉重叠和出画时外观特征几乎不可分，ReID 本来就该失败，这正是下面"遮挡必须上 ReID"那段要解决的问题；换真实监控素材会明显好转。

**带 Agent 的完整链路**：

```bash
set DEEPSEEK_API_KEY=sk-...        # Linux/macOS: export DEEPSEEK_API_KEY=...
python pipeline.py --source your_video.mp4 --model yolov8n-pose.pt \
  --output outputs/results.json --kg-path outputs/scene.graphml

# 额外接上异常检测 Agent（路由变成 trajectory → interaction → anomaly → summary）
python pipeline.py --source your_video.mp4 --with-anomaly \
  --output outputs/results.json --kg-path outputs/scene.graphml
```

**Web 演示**：先取消 `requirements.txt` 里 Web 段的注释并安装，然后 `python -m uvicorn web.backend.main:app --port 8000`，打开 `http://127.0.0.1:8000`。

## 测试

```bash
python -m pytest -q
```

实测 **84 passed**（Python 3.13 + pytest，本机完整依赖，两次运行 17.5s / 17.7s）。`tests/` 共 11 个文件，覆盖 tracker 配置、Entity 合并判定、事件序列化、KG 连续性写入、gate 指标、Web 前端契约。

## 技术栈

| 层 | 选型 |
| --- | --- |
| 检测 / 姿态 | Ultralytics YOLOv8（det 与 pose 两阶段） |
| 跟踪 | 自实现：GMC 稀疏光流 + 8 状态卡尔曼 + OSNet ReID + 匈牙利匹配 |
| 特征 | OSNet x0.25（MSMT17），BoxMOT / torchreid 双后端 |
| 图谱 | NetworkX DiGraph + GraphML |
| Agent | 自实现 BaseAgent / Orchestrator + DeepSeek（OpenAI SDK 兼容） |
| 服务 | FastAPI + WebSocket |

## 已知限制

- **没有标准 MOT 指标**。没有跑出 IDF1 / MOTA / HOTA。`evaluation/` 是自建 gate 体系：帧级召回/精度（各 0.90 门槛）、人物发现率、最长连续漏检（10 帧）、错误合并次数、碎片数；`recovery_rate` 依赖人工连续性标注，当前显式返回 `None`。要标准指标需自备 MOT17，并用 py-motmetrics 或 TrackEval 重跑。
- **没有 TURN、没有做实时流**。逐帧离线处理，Web 端是 WebSocket 推进度 + base64 帧，不是 WebRTC，前端拿不到低延迟视频。
- **单相机**。没有跨相机 ReID，多路视频只能各自建图。
- **只有 legacy 前端**。`web/frontend` 是可用版本；未完成的 `frontend-v2` 脚手架未包含在本次发布中。
- **OSNet 权重不入库**，缺失时回退 ResNet18，日志会打印 `[ReID] BoxMOT 不可用（…），回退 ResNet18`，ID 连续性会变差。
- **依赖在线 LLM**。Agent 层需要 DeepSeek API，`--dry-run` 不依赖。
- **实验脚本未整理**。`scripts/` 里多数是一次性入口，参数依赖本机 GPU 与绝对路径。
