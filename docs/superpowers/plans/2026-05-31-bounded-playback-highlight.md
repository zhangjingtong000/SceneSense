# Bounded Playback And Entity Highlight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make entity and event clicks start bounded playback at the correct first frame, add a highlight-only entity action, and render a stable high-contrast entity highlight on an independent overlay canvas.

**Architecture:** Keep the existing single-file frontend and frozen WebSocket payloads. Add a small `playbackRange` state machine for bounded playback and a transparent `highlightCanvas` above the base canvas so selected-entity rendering no longer redraws or contaminates ordinary bbox frames.

**Tech Stack:** Vanilla HTML/CSS/JavaScript, Python `pytest`, Codex in-app browser

---

## File Structure

- Modify: `tests/web/test_frontend_contract.py`
  - Lock the requested playback and highlight interaction contracts with static frontend assertions.
- Modify: `web/frontend/index.html`
  - Add bounded playback helpers, entity-row highlight-only buttons, and the independent highlight overlay canvas.

Do not modify backend payloads, detection logic, entity fusion, Agent protocols, or KG code in this plan.

### Task 1: Lock Bounded Playback And Overlay Contracts

**Files:**
- Modify: `tests/web/test_frontend_contract.py:94`

- [ ] **Step 1: Replace the existing frontend interaction contract test with explicit bounded-playback assertions**

Use this test:

```python
def test_frontend_bounded_playback_and_overlay_highlight_contract():
    html = Path("web/frontend/index.html").read_text(encoding="utf-8")

    assert 'id="highlightCanvas"' in html
    assert "let playbackRange=null;" in html
    assert "startBoundedPlayback(start,end,'event',ev.event_id||'');" in html
    assert "startBoundedPlayback(start,end,'entity',entityId);" in html
    assert "Number(ent.first_seen_frame)" in html
    assert "Number(ent.last_seen_frame)" in html
    assert "highlightBtn.onclick=e=>toggleEntityHighlightOnly(e,ent.entity_id);" in html
    assert "e.stopPropagation();" in html
    assert "clearPlaybackRange();" in html
    assert "const endFrame=playbackRange?playbackRange.endFrame:totalFrames-1;" in html
    assert "drawHighlight(highlightCtx,bbox,selectedEntity||selectedPerson||'');" in html
    assert "ctx.fillText(String(label),labelX+5,labelY);" not in html
    assert "待确认 ${escapeHtml(tentative)}?" in html
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```powershell
python -m pytest tests/web/test_frontend_contract.py::test_frontend_bounded_playback_and_overlay_highlight_contract -v
```

Expected: FAIL because `highlightCanvas`, `playbackRange`, and bounded playback helpers do not exist yet.

### Task 2: Implement Bounded Playback State

**Files:**
- Modify: `web/frontend/index.html:229-235`
- Modify: `web/frontend/index.html:476-481`
- Modify: `web/frontend/index.html:566-573`
- Modify: `web/frontend/index.html:683-720`

- [ ] **Step 1: Add and reset the playback range state**

Add beside the existing playback state:

```js
let playbackRange=null;
```

Reset it when a new upload starts:

```js
selectedPerson=null;selectedEntity=null;framePrefix='';playingFrame=0;playbackRange=null;
```

- [ ] **Step 2: Add bounded playback helpers**

Add before `togglePlayback()`:

```js
function clampFrame(n){
  return Math.max(0,Math.min(totalFrames-1,Math.round(Number(n))||0));
}
function clearPlaybackRange(){
  playbackRange=null;
}
function startBoundedPlayback(start,end,sourceType,sourceId){
  if(!framePrefix||!totalFrames)return;
  const startFrame=clampFrame(start);
  const endFrame=Math.max(startFrame,clampFrame(end));
  playbackRange={startFrame,endFrame,sourceType,sourceId};
  playFrom(startFrame);
}
```

- [ ] **Step 3: Make ordinary playback clear the active range and enforce the bounded end frame**

Change the ordinary play toggle and timer:

```js
function togglePlayback(){
  if(playTimer){stopPlay();return;}
  clearPlaybackRange();
  const scrub=Number(document.getElementById('scrubBar').value);
  const start=Number.isFinite(scrub)?scrub:playingFrame;
  playFrom(start>=totalFrames-1?0:start);
}
function playFrom(n){
  stopPlay();
  playingFrame=clampFrame(n);
  showFrame(playingFrame);
  const playBtn=document.getElementById('playBtn');
  if(playBtn)playBtn.textContent='播放中';
  playTimer=setInterval(()=>{
    const endFrame=playbackRange?playbackRange.endFrame:totalFrames-1;
    if(playingFrame>=endFrame){stopPlay();return;}
    showFrame(Math.min(endFrame,playingFrame+1));
  },1000/Math.max(1,fps));
}
```

- [ ] **Step 4: Make event heat blocks and right-side event rows start bounded playback**

Both existing entry points already call `selectEvent(ev)`. Update that function:

```js
function selectEvent(ev){
  showEventDetail(ev);
  setRightTab('kg');
  const start=Number(ev.frame_start)||Math.round((ev.duration?.[0]||0)*fps)||0;
  const end=Number(ev.frame_end)||Math.max(start,Math.round((ev.duration?.[1]||0)*fps));
  startBoundedPlayback(start,end,'event',ev.event_id||'');
}
```

- [ ] **Step 5: Split entity selection into row playback and highlight-only actions**

Replace `selectEntity()` with:

```js
function toggleEntityHighlight(entityId){
  selectedEntity=selectedEntity===entityId?null:entityId;
  selectedPerson=null;
  renderEntities();renderLifelines();renderEvidence();
  redrawSelectedHighlight();
}
function playEntityRange(entityId){
  const ent=entities.find(e=>e.entity_id===entityId);
  toggleEntityHighlight(entityId);
  if(!ent)return;
  const start=Number(ent.first_seen_frame)||0;
  const end=Number(ent.last_seen_frame)||start;
  startBoundedPlayback(start,end,'entity',entityId);
}
function toggleEntityHighlightOnly(e,entityId){
  e.stopPropagation();
  toggleEntityHighlight(entityId);
}
function selectEntity(entityId,jump=true){
  if(jump){playEntityRange(entityId);return;}
  toggleEntityHighlight(entityId);
}
```

This keeps existing bbox click and lifeline call sites compatible while making entity row clicks start the requested bounded playback.

- [ ] **Step 6: Add the entity-row highlight-only button**

Extend the entity item template with a compact action button:

```js
item.innerHTML=`<div class="entity-main"><span class="entity-name" style="color:${entityColor(ent.entity_id)}">${escapeHtml(ent.entity_id)}</span><span class="quality">${qualityLabel(ent.tracking_quality)}</span><button class="highlight-only-btn" type="button" title="仅切换高亮，不跳转">仅高亮</button></div>
  <div class="entity-meta">
    <span>帧 ${ent.first_seen_frame}-${ent.last_seen_frame}</span><span>轨迹点 ${points}</span>
    <span>track ${escapeHtml(tracks)}</span><span>${tentativeText}</span>
  </div>`;
const highlightBtn=item.querySelector('.highlight-only-btn');
highlightBtn.onclick=e=>toggleEntityHighlightOnly(e,ent.entity_id);
```

Add the restrained button CSS:

```css
.highlight-only-btn{margin-left:auto;border:1px solid var(--border);background:var(--panel);color:var(--soft);border-radius:5px;padding:2px 7px;font-size:10px;cursor:pointer;white-space:nowrap}
.highlight-only-btn:hover{border-color:var(--accent);color:var(--accent)}
```

- [ ] **Step 7: Run the targeted contract test**

Run:

```powershell
python -m pytest tests/web/test_frontend_contract.py::test_frontend_bounded_playback_and_overlay_highlight_contract -v
```

Expected: the test still fails only on overlay-canvas assertions, because playback is implemented but the overlay is not.

### Task 3: Move Entity Highlight To An Independent Overlay Canvas

**Files:**
- Modify: `web/frontend/index.html:27`
- Modify: `web/frontend/index.html:161`
- Modify: `web/frontend/index.html:231`
- Modify: `web/frontend/index.html:692`
- Modify: `web/frontend/index.html:735-764`

- [ ] **Step 1: Add the transparent overlay canvas**

Add directly after the base canvas:

```html
<canvas id="highlightCanvas" style="display:none;position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none"></canvas>
```

Add the JavaScript context:

```js
let highlightCanvas=document.getElementById('highlightCanvas'),highlightCtx=highlightCanvas.getContext('2d');
```

Show it wherever the base canvas is shown:

```js
canvas.style.display='';highlightCanvas.style.display='';
```

- [ ] **Step 2: Keep both canvas bitmap sizes synchronized**

Add:

```js
function ensureCanvasSize(width,height){
  if(canvasSized)return;
  canvas.width=width;canvas.height=height;
  highlightCanvas.width=width;highlightCanvas.height=height;
  canvasSized=true;
}
```

Replace both current `if(!canvasSized)` canvas initialization branches with:

```js
ensureCanvasSize(img.naturalWidth,img.naturalHeight);
```

and:

```js
ensureCanvasSize(imgR.naturalWidth,imgR.naturalHeight);
```

- [ ] **Step 3: Draw selected highlights only on the overlay**

Replace the selection rendering helpers with:

```js
function redrawSelectedHighlight(){
  highlightCtx.clearRect(0,0,highlightCanvas.width,highlightCanvas.height);
  const snap=latestCanvasSnap||{};
  const bbox=selectedEntity?snap[selectedEntity]:(selectedPerson?snap[selectedPerson]:null);
  if(!bbox)return;
  drawHighlight(highlightCtx,bbox,selectedEntity||selectedPerson||'');
}
function renderCanvasFrame(image,snap,frame){
  latestCanvasImage=image;latestCanvasSnap=snap;latestCanvasFrame=frame;
  ctx.drawImage(image,0,0,canvas.width,canvas.height);
  drawSkeletonOverlay(frame);
  redrawSelectedHighlight();
}
function redrawCurrentFrame(){
  if(!latestCanvasImage)return;
  ctx.drawImage(latestCanvasImage,0,0,canvas.width,canvas.height);
  drawSkeletonOverlay(latestCanvasFrame);
  redrawSelectedHighlight();
}
function drawHighlight(targetCtx,bbox,label=''){
  const x=bbox[0],y=bbox[1],w=bbox[2]-bbox[0],h=bbox[3]-bbox[1];
  targetCtx.save();
  targetCtx.fillStyle='rgba(240,160,64,.18)';
  targetCtx.fillRect(x,y,w,h);
  targetCtx.strokeStyle='#fff';targetCtx.lineWidth=8;
  targetCtx.strokeRect(x,y,w,h);
  targetCtx.strokeStyle='#ff9f1c';targetCtx.lineWidth=4;
  targetCtx.strokeRect(x,y,w,h);
  if(label){
    targetCtx.font='800 14px Microsoft YaHei, sans-serif';
    const labelX=Math.max(2,x),labelY=Math.max(18,y-7);
    const width=targetCtx.measureText(String(label)).width+12;
    targetCtx.fillStyle='rgba(255,159,28,.98)';targetCtx.fillRect(labelX,labelY-16,width,21);
    targetCtx.fillStyle='#111318';targetCtx.fillText(String(label),labelX+6,labelY);
  }
  targetCtx.restore();
}
```

- [ ] **Step 4: Run targeted and web contract tests**

Run:

```powershell
python -m pytest tests/web/test_frontend_contract.py -v
```

Expected: PASS.

- [ ] **Step 5: Run the full automated test suite**

Run:

```powershell
python -m pytest -q
```

Expected: PASS.

### Task 4: Browser Verification

**Files:**
- Verify: `web/frontend/index.html`

- [ ] **Step 1: Start or reuse the local web service**

Run:

```powershell
python -m uvicorn web.backend.app:app --host 127.0.0.1 --port 8000
```

Expected: SceneSense is available at `http://127.0.0.1:8000/`. If port `8000` is already serving the current workspace, reuse it.

- [ ] **Step 2: Load the page in the Codex in-app browser**

Open:

```text
http://127.0.0.1:8000/
```

Expected: page loads without console errors and the overlay canvas is present above the base canvas.

- [ ] **Step 3: Upload `D:\gp\test_data\mot17_180.mp4` and verify entity behavior**

Verify:

- clicking an entity row jumps to `first_seen_frame`;
- the entity row click starts playback automatically;
- playback stops at `last_seen_frame`;
- the selected entity uses a stable high-contrast overlay;
- clicking “仅高亮” changes selection without jumping or changing playback state.

- [ ] **Step 4: Verify event behavior**

Verify:

- clicking an event heat block starts playback at `frame_start`;
- playback stops at `frame_end`;
- clicking the matching right-side event row follows the same behavior;
- pressing the ordinary play button clears the bounded range and continues full-video playback.

- [ ] **Step 5: Inspect the diff**

Run:

```powershell
git diff --check
git diff -- tests/web/test_frontend_contract.py web/frontend/index.html
```

Expected: no whitespace errors; diff only contains the intended frontend interaction work in these already-modified files.

Do not commit automatically. The working tree contains earlier uncommitted work, so review the combined frontend diff with the user before staging.
