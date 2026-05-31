# Entity Continuity, Motion Speed, and Persistent Highlight Design

Date: 2026-05-31

## Scope

This iteration addresses three related usability and perception issues:

1. A person can receive a new visible number after occlusion and reappearance.
2. Motion speed labels do not reliably expose fast movement in the provided test videos.
3. Selecting an entity in the right panel does not provide a sufficiently persistent visual anchor in the video.

The existing `SpatioTemporalEvent`, `AgentMessage`, `AgentResponse`, and KG schemas remain unchanged. Raw `track_id` values remain traceable evidence. No external ReID model is introduced.

## Design Decisions

### 1. Conservative Entity Continuity

`EntityProfileManager` remains the owner of `track_id -> Person-N` fusion.

When a new raw track appears:

- High-confidence evidence automatically maps the raw track to the existing `Person-N`.
- Medium-confidence evidence creates a tentative entity association. The UI exposes this as `待确认 Person-N?`.
- Low-confidence evidence creates a new entity without a proposed predecessor.

Tentative associations accumulate evidence over subsequent frames. Promotion requires sustained agreement from:

- spatial proximity,
- motion direction,
- appearance similarity when available,
- temporal continuity.

Appearance similarity alone must not promote a merge when the spatial gap is implausible. The implementation will preserve all raw tracks, merge evidence, and occlusion gaps for audit.

### 2. Perspective-Reduced Speed Classification

Tracker output currently represents image-plane velocity in pixels per second despite the historical field name `velocity_ms`. This iteration does not claim real-world meters per second.

For each tracklet, calculate:

```text
normalized_speed = speed_px_s / bbox_height_px
```

The trajectory Agent uses normalized speed as its primary speed-level signal and retains raw `px/s` for audit and display.

Speed level and motion pattern are separate concepts:

- Speed level: static, slow, normal, fast, running.
- Motion pattern: stationary, walking, loitering, erratic.

A high normalized speed must not be overwritten by a small net displacement rule. This prevents fast movement with turns or fragmented tracks from being mislabeled as slow.

Agent motion output adds:

- `speed_px`
- `normalized_speed`
- `speed_pct`
- `speed_level`
- classification evidence

Existing consumers remain compatible because existing keys are retained.

### 3. Persistent Entity Highlight

The frontend uses `selectedEntity` as the single visual selection state.

Interaction rules:

- Click an entity in the right panel to enable highlight.
- Click the same entity again to clear highlight.
- Click a different entity to switch highlight.
- Playback, scrubbing, and live frames preserve the selected entity.
- If the selected entity is absent in the current frame, no stale box is drawn.
- When the selected entity reappears, highlighting resumes automatically.

Highlight style:

- thick orange outline,
- restrained glow,
- fixed `Person-N` label near the box,
- normal rendering for other detections.

## Data Flow

```text
YOLO detections
  -> tracker raw tracks
  -> EntityProfileManager conservative fusion
  -> enriched tracklets with Person-N and tentative metadata
  -> SpatioTemporalEvent
  -> TrajectoryAgent normalized speed classification
  -> WebSocket display payload
  -> persistent frontend entity highlight
```

## Testing

### Unit Tests

- A plausible occlusion reappearance continues the original entity.
- Appearance-only evidence with an implausible spatial gap does not merge.
- A medium-confidence reappearance remains tentative until sustained evidence promotes it.
- Normalized speed reduces perspective sensitivity for similar person motion.
- High normalized speed is not classified as slow solely because net displacement is small.
- Clicking an already-selected entity clears frontend selection.

### Video Benchmarks

Run both:

- `D:\gp\test_data\mot17_180.mp4`
- `D:\gp\test_data\mot17_04_raw.mp4`

Record JSON and terminal summaries containing:

- processed FPS,
- tracker drop rate,
- detection gap rate,
- raw track count,
- entity count,
- ID switch count,
- entity merge records,
- speed-level distribution.

### Browser Verification

Verify:

- entity highlight enables, switches, and clears,
- highlight persists during playback and scrub,
- highlight disappears while absent and resumes after reappearance,
- tentative predecessor text is visible when applicable.

## Non-Goals

- No tracker replacement.
- No external ReID network.
- No real-world `m/s` claim without camera calibration.
- No generic KG query language.
- No aggressive entity merge policy.
