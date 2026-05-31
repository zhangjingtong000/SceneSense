# Main Stabilize Salvage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean stabilization branch from `main` that fixes core detection playback usability without inheriting the experimental `trae` branch.

**Architecture:** Keep the `main` perception and EntityProfile pipeline as the source of truth. Apply narrowly scoped frontend/backend fixes for startup, playback, target hit-testing, and layout pressure, while leaving AnomalyAgent, calibration, dynamic KG rules, and the full `trae` UI rewrite out of the default path.

**Tech Stack:** Python 3.10, OpenCV, ultralytics, supervision-style tracker wrapper, FastAPI WebSocket backend, plain HTML/CSS/JavaScript frontend.

---

### Task 1: Preserve High-Quality Detection Baseline

**Files:**
- Modify: `web/backend/pipeline_runner.py`
- Test: `python -m pytest -q`

- [ ] Confirm the salvage branch starts from `main`, not `trae`.
- [ ] Keep `YOLODetector("yolov8l.pt", conf=0.15, imgsz=960)` and `lost_track_buffer=120`.
- [ ] If model warmup is added, read the first video frame through `cap.read()` and reset `CAP_PROP_POS_FRAMES` afterward.
- [ ] Do not add `detect_interval=2` unless tracker output is carried forward on skipped frames.

### Task 2: Fix Playback Controls

**Files:**
- Modify: `web/frontend/index.html`
- Test: manual browser smoke on `http://127.0.0.1:8000/`

- [ ] Find the play/pause state variables and frame slider handlers.
- [ ] Make play work from any slider position by deriving the next frame from the current slider value.
- [ ] Stop playback at the last known frame without requiring the slider to be at frame zero.
- [ ] Keep all visible UI text in Chinese.

### Task 3: Improve Target Click Hit-Testing

**Files:**
- Modify: `web/frontend/index.html`
- Test: manual browser smoke on `http://127.0.0.1:8000/`

- [ ] Normalize click coordinates from displayed image size to natural frame coordinates.
- [ ] Hit-test against expanded bbox padding, not just the exact rectangle.
- [ ] Prefer the smallest matching bbox when boxes overlap so clicking a small person does not select a larger neighbor.
- [ ] Preserve existing selected-entity linkage to timeline and KG panels.

### Task 4: Reduce Single-Page Crowding Without Full Redesign

**Files:**
- Modify: `web/frontend/index.html`
- Test: manual browser smoke on desktop width and narrow width.

- [ ] Replace the most crowded right-side panels with Chinese tab buttons for entities, events, KG, and logs.
- [ ] Keep video and timeline visible as the primary workspace.
- [ ] Add CSS `resize` support to the most important panel boundaries as a first step toward full draggable splitters.
- [ ] Do not copy the full `trae` frontend rewrite.

### Task 5: Verification

**Files:**
- Run only.

- [ ] Run `python -m pytest -q` and confirm all tests pass.
- [ ] Start the local backend if needed and open the app.
- [ ] Smoke-test `D:\gp\test_data\mot17_180.mp4` for startup, playback from middle, target clicking, and panel layout.
- [ ] Report which checks were automated and which were manual.
