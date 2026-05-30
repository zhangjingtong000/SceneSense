# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project overview

YOLO 时空感知 + 多Agent协作场景理解 + 知识图谱推理系统。

三层架构：感知层（YOLOv8+ByteTrack）→ 多Agent分析层（轨迹Agent+交互Agent+场景摘要Agent，星型拓扑）→ KG层（NetworkX图存储）。目标是用自然语言描述视频场景中发生的事——每个目标的运动模式、目标间的交互关系、全局场景摘要。

## Reference code

`references/` 目录包含 5 个参考项目，**不向这些目录写入代码**，只读：

| 目录 | 项目 | 参考价值 |
|------|------|---------|
| `references/Project-Ava/` | EKG构建+Agent检索 | KG构建管线、NetworkXStorage、tri-view retrieval |
| `references/VideoMultiAgents/` | 多Agent VQA | 星型拓扑（LangGraph实现）、Organizer路由决策 |
| `references/Vision-Agents/` | 实时视觉Agent | Processor管线模式（小模型感知→大模型推理） |
| `references/GraphAgent/` | Agent图推理 | 三阶段Agent（图生成→任务规划→动作执行） |
| `references/LightAgent/` | 极简Agent框架 | Agent循环参考（clone失败，部分文件不可读） |

## Architecture

完整架构和接口协议见 `docs/architecture.md`。

### 三级数据流

```
视频 → [感知层] YOLOv8+ByteTrack → SpatioTemporalEvent(JSON)
     → [事件队列]
     → [Agent层] 协调Agent ←星型→ 轨迹/交互/异常Agent
     → [KG层] NetworkX 图存储 ←→ 规则引擎
     → [输出] 行为报告 + 实时告警
```

### 已冻结的接口契约（位于 `docs/architecture.md` 第4节）

| 接口 | 方向 | 关键字段 |
|------|------|---------|
| `SpatioTemporalEvent` (4.1) | 感知层→Agent层 | `event_id, duration, frame_start/end, fps, tracklets[], spatial_relations[], zone` |
| `AgentMessage` (4.2) | 协调Agent→分析Agent | `target_agent, action, guidance, context` |
| `AgentResponse` (4.3) | 分析Agent→协调Agent | `agent_confidence, warnings[], findings[{entity_id, confidence, evidence, kg_operations}]` |
| `Organizer Output` (4.4) | 协调Agent→输出层 | `status, rounds_used, alerts[{fused_severity, contributing_agents, reasoning_path}]` |

### 关键设计约束

- Agent之间**不直接通信**，只通过协调Agent。这是VideoMultiAgents论文验证的最优结构
- KG是**唯一共享状态**，所有Agent的kg_operations由协调者统一执行
- 每个finding必须带`evidence`字段，实现审计可追溯
- 异常Agent输出`raw_score`（原始分），协调Agent做加权融合→`fused_severity`
- 协调Agent有终止条件：CONSENSUS / STALEMATE（2轮未共识→uncertain） / MAX_ROUNDS（≤3轮）
- Event节点有`expires_at`TTL，KG API提供`evict_before()`防止内存无限增长

### KG Schema

节点类型：Entity / Event / Behavior / Scene
边类型：belongs_to / interacts_with / matches / next(时序) / violates

## Tech stack

- **检测+跟踪**: `ultralytics` (YOLOv8) + `supervision` (ByteTrack)
- **Agent编排**: 自己写（`openai` SDK → DeepSeek API），不引入LangChain/CrewAI
- **KG存储**: `NetworkX` 内存图，持久化到 GraphML
- **LLM**: DeepSeek API（兼容 OpenAI SDK 格式）
- **运行环境**: Python 3.10+, Windows

## Commands

```bash
# 安装依赖（项目初始化时）
pip install ultralytics supervision openai networkx numpy

# 运行感知层测试
python -m perception.detector --source <video_path>

# 运行全链路（集成后）
python pipeline.py --source <video_path> --config config.yaml

# Agent层单元测试
python -m pytest tests/agents/ -v

# KG层单元测试
python -m pytest tests/kg/ -v
```

## 项目结构规划

```
d:/gp/
├── perception/           # 感知层
│   ├── detector.py       # YOLOv8封装
│   ├── tracker.py        # ByteTrack封装（基于supervision）
│   └── event_builder.py  # 轨迹→SpatioTemporalEvent转换
├── agents/               # 多Agent层
│   ├── llm_client.py     # DeepSeek API封装（openai SDK）
│   ├── base_agent.py     # Agent基类
│   ├── orchestrator.py   # 协调Agent
│   ├── trajectory_agent.py
│   ├── interaction_agent.py
│   ├── anomaly_agent.py
│   └── prompts/          # Prompt模板目录
│       ├── trajectory.md
│       ├── interaction.md
│       ├── anomaly.md
│       └── orchestrator.md
├── kg/                   # 知识图谱层
│   ├── graph_store.py    # NetworkX封装，实现KG API (docs/architecture.md §3.1)
│   ├── rule_engine.py    # 时空规则引擎
│   └── kg_builder.py     # LLM辅助KG构建
├── pipeline.py           # 主管线编排
├── config.yaml           # 全局配置
├── docs/
│   └── architecture.md   # 架构与接口协议（权威文档）
└── tests/
    ├── perception/
    ├── agents/
    └── kg/
```

## 开发流程

1. 三层并行开发，接口契约已冻结在`docs/architecture.md`
2. Agent层开发时KG接口可以先mock（`graph_store.py`定义后即可）
3. 集成阶段用真实视频跑全链路验证
4. 可视化调试可用playwright MCP打开浏览器看YOLO检测框
5. context7 MCP可实时查ultralytics/supervision/networkx最新API
