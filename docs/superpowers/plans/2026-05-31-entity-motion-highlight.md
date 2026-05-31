# Entity Continuity, Motion Speed, and Persistent Highlight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve conservative person identity continuation after occlusion, classify speed using perspective-reduced image-plane motion, and make entity selection visibly persistent and toggleable.

**Architecture:** Preserve raw tracker IDs and frozen Agent/Event contracts. Extend `EntityProfileManager` with conservative predicted-position scoring and tentative predecessor metadata, extend `TrajectoryAgent` with normalized speed while preserving raw `px/s`, and keep frontend highlight state in `selectedEntity`.

**Tech Stack:** Python 3.10+, pytest, YOLO tracker pipeline, FastAPI WebSocket frontend, vanilla JavaScript canvas.

---

### Task 1: Conservative Entity Continuity

**Files:**
- Modify: `perception/entity_profile.py`
- Test: `tests/perception/test_entity_profile.py`

- [ ] Add a failing test showing a walking person reappearing after an occlusion gap can match its predicted continuation point.
- [ ] Add a failing test showing appearance-only similarity at an implausible spatial distance does not auto-merge.
- [ ] Add a failing test showing enriched tentative output exposes `tentative_predecessor`.
- [ ] Run `python -m pytest tests/perception/test_entity_profile.py -q` and confirm the new tests fail for the expected missing behavior.
- [ ] Add conservative predicted-position scoring from recent entity trajectory points.
- [ ] Require plausible spatial continuity for automatic merge.
- [ ] Expose tentative predecessor metadata without changing raw `track_id`.
- [ ] Run `python -m pytest tests/perception/test_entity_profile.py -q`.

### Task 2: Perspective-Reduced Speed Classification and Person IDs

**Files:**
- Modify: `agents/trajectory_agent.py`
- Create: `tests/agents/test_trajectory_agent.py`

- [ ] Add a failing test showing equal relative motion at different bbox heights yields equal normalized speed.
- [ ] Add a failing test showing high normalized speed is not overwritten by a low net-displacement rule.
- [ ] Add a failing test showing Agent findings prefer existing `Person-N` IDs over synthesized `Entity-track_id`.
- [ ] Run `python -m pytest tests/agents/test_trajectory_agent.py -q` and confirm the new tests fail.
- [ ] Add normalized speed calculation and a normalized percentile window.
- [ ] Separate speed level from motion-pattern labeling while retaining existing output keys.
- [ ] Use `tracklet["entity_id"]` when present.
- [ ] Run `python -m pytest tests/agents/test_trajectory_agent.py -q`.

### Task 3: Toggleable Persistent Highlight and Tentative UI

**Files:**
- Modify: `web/backend/pipeline_runner.py`
- Modify: `web/frontend/index.html`
- Modify: `tests/web/test_frontend_contract.py`

- [ ] Add contract checks for tentative predecessor propagation and repeat-click clear logic.
- [ ] Run `python -m pytest tests/web/test_frontend_contract.py -q` and confirm the new checks fail.
- [ ] Include tentative predecessor metadata in stable display persons.
- [ ] Make `selectEntity()` clear selection when the same entity is clicked again.
- [ ] Draw a persistent thick orange outline, glow, and fixed selected `Person-N` label.
- [ ] Display `待确认 Person-N?` for tentative entities.
- [ ] Run `python -m pytest tests/web/test_frontend_contract.py -q`.

### Task 4: Verification

**Files:**
- Modify: `scripts/benchmark_video.py`
- Test: existing suite

- [ ] Add speed-level distribution to benchmark JSON and terminal summary.
- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m py_compile perception/entity_profile.py agents/trajectory_agent.py web/backend/pipeline_runner.py scripts/benchmark_video.py`.
- [ ] Run `git diff --check`.
- [ ] Benchmark `D:\gp\test_data\mot17_180.mp4`.
- [ ] Benchmark `D:\gp\test_data\mot17_04_raw.mp4`.
- [ ] Restart the FastAPI service and verify highlight enable, switch, clear, and persistence in the browser.
