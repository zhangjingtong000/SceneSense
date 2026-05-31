// State
let videoId=null,ws=null,fps=25,totalFrames=0,frameIdx=0,events=[],framePrefix='';
let entities=[],lifelines=[],kgGraph={nodes:[],edges:[]},kgFilter='all';
let canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
let highlightCanvas=document.getElementById('highlightCanvas'),highlightCtx=highlightCanvas.getContext('2d');
let personMap={},selectedPerson=null,selectedEntity=null,frameBboxes={},framePersons={};
let playingFrame=0,playTimer=null,canvasSized=false,latestLiveFrame=-1,frameRequestToken=0;
let playbackRange=null;
let latestCanvasImage=null,latestCanvasSnap={},latestCanvasFrame=0;
let poseCache={},skeletonEnabled=false,skeletonDataReady=false;
let appSettings=loadSettings();
const POSE_MAX_AGE=12;
const ENTITY_COLORS=['#5b9bd5','#4caf91','#f0a040','#e05a68','#a27bd8','#5ac6c8','#d97842','#8bbf5c','#c96b9e','#6ba8d9'];

// Router
function route(){
  const hash=location.hash.replace('#','')||'home';
  document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id==='page-'+hash));
  document.querySelectorAll('.nav-link').forEach(a=>a.classList.toggle('active',a.dataset.page===hash));
  if(hash==='workspace'){requestAnimationFrame(()=>{renderTimeline();renderEvidence();});}
}
window.addEventListener('hashchange',route);
route();

// Settings
function loadSettings(){
  try{return JSON.parse(localStorage.getItem('scenesense:settings'))||{};}catch(_){return {};}
}
function saveSettings(){
  appSettings={
    conf:document.getElementById('setConf')?.value||'0.15',
    imgsz:document.getElementById('setImgsz')?.value||'960',
    classes:document.getElementById('setClasses')?.value||'0',
    lost_buffer:document.getElementById('setLostBuffer')?.value||'120',
    matching_threshold:document.getElementById('setMatchThresh')?.value||'0.65',
    with_reid:document.getElementById('setReid')?.checked||false,
    with_gmc:document.getElementById('setGmc')?.checked||false,
    entity_auto:document.getElementById('setAutoMerge')?.value||'0.72',
    entity_tentative:document.getElementById('setTentative')?.value||'0.30',
  };
  localStorage.setItem('scenesense:settings',JSON.stringify(appSettings));
  addChatMsg('agent','设置已保存。重新上传视频后生效。');
}
function syncSettingsToUI(){
  const s=appSettings;
  document.getElementById('setConf').value=s.conf||'0.15';
  document.getElementById('setConfVal').textContent=s.conf||'0.15';
  document.getElementById('setImgsz').value=s.imgsz||'960';
  document.getElementById('setClasses').value=s.classes||'0';
  document.getElementById('setLostBuffer').value=s.lost_buffer||'120';
  document.getElementById('setLostBufferVal').textContent=s.lost_buffer||'120';
  document.getElementById('setMatchThresh').value=s.matching_threshold||'0.65';
  document.getElementById('setMatchThreshVal').textContent=s.matching_threshold||'0.65';
  document.getElementById('setReid').checked=s.with_reid===true;
  document.getElementById('setGmc').checked=s.with_gmc!==false;
  document.getElementById('setAutoMerge').value=s.entity_auto||'0.72';
  document.getElementById('setAutoMergeVal').textContent=s.entity_auto||'0.72';
  document.getElementById('setTentative').value=s.entity_tentative||'0.30';
  document.getElementById('setTentativeVal').textContent=s.entity_tentative||'0.30';
}
document.getElementById('setConf').oninput=function(){document.getElementById('setConfVal').textContent=this.value;};
document.getElementById('setLostBuffer').oninput=function(){document.getElementById('setLostBufferVal').textContent=this.value;};
document.getElementById('setMatchThresh').oninput=function(){document.getElementById('setMatchThreshVal').textContent=this.value;};
document.getElementById('setAutoMerge').oninput=function(){document.getElementById('setAutoMergeVal').textContent=this.value;};
document.getElementById('setTentative').oninput=function(){document.getElementById('setTentativeVal').textContent=this.value;};

// Upload (shared between home and nav)
function startUpload(file){
  const fd=new FormData();fd.append('file',file);
  setStatus('loading','上传中...');
  fetch('/upload',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{
    videoId=d.video_id;
    setStatus('loading','分析中...');
    location.hash='workspace';
    setupWorkspace();
  }).catch(e=>setStatus('error','上传失败'));
}
function setupWorkspace(){
  document.getElementById('overlay').style.display='none';
  canvas.style.display='';highlightCanvas.style.display='';
  canvasSized=false;
  document.getElementById('progressWrap').style.display='';
  events=[];entities=[];lifelines=[];kgGraph={nodes:[],edges:[]};personMap={};frameBboxes={};framePersons={};poseCache={};
  skeletonEnabled=false;skeletonDataReady=false;updateSkeletonButton();
  selectedPerson=null;selectedEntity=null;framePrefix='';playingFrame=0;playbackRange=null;
  redrawSelectedHighlight();
  renderTimeline();renderLifelines();renderEntities();renderEvents();renderEvidence();
  document.getElementById('chatMsgs').innerHTML='';
  resetAgentView();resetAgentDots();
  connectWS();
}
document.getElementById('uploadBtn').onclick=()=>document.getElementById('fileInput').click();
const homeBtn=document.getElementById('homeUploadBtn');
if(homeBtn)homeBtn.onclick=()=>document.getElementById('fileInput').click();
document.getElementById('fileInput').onchange=function(){
  const f=this.files[0];if(!f)return;
  startUpload(f);
};
syncSettingsToUI();
resetAgentView();

// WebSocket
function connectWS(){
  if(ws)ws.close();
  const proto=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(`${proto}//${location.host}/ws/${videoId}`);
  ws.onmessage=async e=>{
    const msg=JSON.parse(e.data);
    switch(msg.type){
      case 'meta':
        fps=Number(msg.fps)||fps;
        totalFrames=Number(msg.total_frames||msg.totalFrames)||totalFrames;
        updateFrameScale();
        break;
      case 'frame':
        handleFrameMessage(msg);
        break;
      case 'pose_update':
        handlePoseUpdate(msg);
        break;
      case 'progress':
        document.getElementById('pctText').textContent=msg.pct+'%';
        document.getElementById('pctFill').style.width=msg.pct+'%';
        setStatus('loading',`分析中... ${msg.pct}%`);
        break;
      case 'agent_status':
        updateAgentStatus(msg.agent,msg.status,msg.message);
        break;
      case 'event':
        events.push(msg);
        mergeDisplayPayload(msg);
        renderTimeline();renderLifelines();renderEntities();renderEvents();renderEvidence();
        addChatMsg('agent',formatEventMsg(msg));
        break;
      case 'done':
        setStatus('ready','分析完成');
        document.getElementById('progressWrap').style.display='none';
        framePrefix=msg.frame_dir||framePrefix;
        totalFrames=Number(msg.total_frames)||totalFrames;
        fps=Number(msg.fps)||fps;
        if(msg.events&&msg.events.length)mergeDisplayPayload(msg.events[msg.events.length-1]);
        canvas.style.display='';highlightCanvas.style.display='';
        document.getElementById('videoPlayer').style.display='none';
        document.getElementById('playerControls').style.display='flex';
        document.getElementById('scrubBar').max=Math.max(0,totalFrames-1);
        showFrame(0);
        renderTimeline();renderLifelines();renderEntities();renderEvents();renderEvidence();updateFrameScale();
        addChatMsg('agent','分析完成。点击事件热力定位片段，点击 Person 行查看完整生命线和 KG 证据。');
        break;
      case 'error':
        setStatus('error',msg.message);
        addChatMsg('agent','错误：'+escapeHtml(msg.message||'未知错误'));
        break;
    }
  };
  ws.onerror=()=>setStatus('error','连接错误');
}
function handleFrameMessage(msg){
  frameIdx=Number(msg.frame_idx)||0;
  latestLiveFrame=Math.max(latestLiveFrame,frameIdx);
  if(msg.entities){entities=msg.entities;renderEntities();}
  const now=Date.now();
  (msg.persons||[]).forEach(p=>{
    p._ts=now;
    p.entity_id=p.entity_id||`Track-${p.track_id}`;
    personMap[p.track_id]=p;
  });
  for(const [tid,p] of Object.entries(personMap)){if(now-p._ts>5000)delete personMap[tid];}
  const snap={};
  const people=[];
  (msg.persons||[]).forEach(p=>{
    if(!p.bbox)return;
    people.push({...p});
    snap[p.track_id]=p.bbox;
    if(p.entity_id)snap[p.entity_id]=p.bbox;
  });
  frameBboxes[frameIdx]=snap;
  framePersons[frameIdx]=people;
  trimFrameSnapshots();
  const imgR=new Image();
  const liveFrame=frameIdx;
  imgR.onload=()=>{
    if(liveFrame<latestLiveFrame)return;
    ensureCanvasSize(imgR.naturalWidth,imgR.naturalHeight);
    renderCanvasFrame(imgR,snap,liveFrame);
  };
  imgR.src='data:image/jpeg;base64,'+msg.image;
}
function handlePoseUpdate(msg){
  const f=Number(msg.frame_idx);
  if(!Number.isFinite(f))return;
  poseCache[f]=msg.poses||[];
  trimPoseCache();
  skeletonDataReady=Object.keys(poseCache).length>0;
  updateSkeletonButton();
}
function mergeDisplayPayload(payload){
  if(payload.entities)entities=payload.entities;
  if(payload.lifelines)lifelines=payload.lifelines;
  if(payload.kg_subgraph)kgGraph=payload.kg_subgraph;
}
function trimFrameSnapshots(){
  const fids=Object.keys(frameBboxes);
  if(fids.length<=3000)return;
  const sorted=fids.map(Number).sort((a,b)=>a-b);
  for(let i=0;i<sorted.length-3000;i++){
    delete frameBboxes[sorted[i]];
    delete framePersons[sorted[i]];
  }
}
function trimPoseCache(){
  const fids=Object.keys(poseCache).map(Number).sort((a,b)=>a-b);
  if(fids.length<=240)return;
  for(let i=0;i<fids.length-240;i++)delete poseCache[fids[i]];
}

// Agent status
function resetAgentDots(){
  ['Traj','Inter','Summ'].forEach(a=>{
    const dot=document.getElementById('dot'+a),stat=document.getElementById('stat'+a);
    if(dot)dot.className='dot';
    if(stat)stat.textContent='等待中';
  });
}
function updateAgentStatus(agent,status,message){
  const map={trajectory_agent:'Traj',interaction_agent:'Inter',summary_agent:'Summ'};
  if(agent==='system'){
    setStatus(status==='loading'?'loading':'ready',message||'就绪');
    return;
  }
  const k=map[agent]; if(!k)return;
  const dot=document.getElementById('dot'+k),stat=document.getElementById('stat'+k);
  if(!dot||!stat)return;
  dot.className='dot '+(status==='active'?'active':'done');
  stat.textContent=status==='active'?'分析中':'完成';
}
function resetAgentView(){
  const ap=document.getElementById('agentPanel');
  ap.innerHTML=`<div class="agent-row"><div class="dot" id="dotTraj"></div><span class="label">轨迹分析</span><span class="stat" id="statTraj">等待中</span></div>
    <div class="agent-row"><div class="dot" id="dotInter"></div><span class="label">交互分析</span><span class="stat" id="statInter">等待中</span></div>
    <div class="agent-row"><div class="dot" id="dotSumm"></div><span class="label">场景总结</span><span class="stat" id="statSumm">等待中</span></div>`;
}

// Timeline and lifelines
function renderTimeline(){
  const bar=document.getElementById('timelineBar');
  bar.innerHTML='';
  if(!events.length){
    bar.innerHTML='<div class="panel-empty" style="padding:9px 10px">分析中，事件片段将在此显示</div>';
    return;
  }
  const total=getTimelineTotal();
  events.forEach(ev=>{
    const start=Number(ev.frame_start)||Math.round((ev.duration?.[0]||0)*fps);
    const end=Number(ev.frame_end)||Math.max(start+1,Math.round((ev.duration?.[1]||0)*fps));
    const left=Math.max(0,Math.min(100,start/total*100));
    const width=Math.max(0.5,(end-start+1)/total*100);
    const seg=document.createElement('div');
    seg.className='seg';
    seg.style.left=left+'%';
    seg.style.width=Math.min(100-left,width)+'%';
    const intensity=Math.min(1,Number(ev.entity_count||0)/25+Number((ev.interactions||[]).length||0)/18);
    const r=Math.round(80+intensity*160),g=Math.round(135+intensity*65),b=Math.round(170-intensity*100);
    seg.style.background=`rgb(${r},${g},${b})`;
    seg.innerHTML=`<span class="tooltip">${formatEventTooltip(ev)}</span>`;
    seg.onclick=()=>selectEvent(ev);
    bar.appendChild(seg);
  });
  if(playingFrame>=0){
    const marker=document.createElement('div');
    marker.className='time-marker';
    marker.style.left=Math.max(0,Math.min(100,playingFrame/getTimelineTotal()*100))+'%';
    bar.appendChild(marker);
  }
}
function renderLifelines(){
  const pane=document.getElementById('lifelinePane');
  pane.innerHTML='';
  if(!lifelines.length){
    pane.innerHTML='<div class="panel-empty">检测到人物后将显示连续轨迹段，断开处为虚线缺口</div>';
    return;
  }
  const total=getTimelineTotal();
  lifelines.slice().sort((a,b)=>entityIndex(a.entity_id)-entityIndex(b.entity_id)).forEach(line=>{
    const row=document.createElement('div');
    row.className='life-row';
    const label=document.createElement('div');
    label.className='life-label'+(line.entity_id===selectedEntity?' active':'');
    label.textContent=line.entity_id;
    label.title='点击查看实体证据';
    label.onclick=()=>selectEntity(line.entity_id);
    const track=document.createElement('div');
    track.className='life-track';
    (line.segments||[]).forEach(seg=>{
      const part=document.createElement('div');
      part.className='life-seg';
      part.style.left=(Number(seg.frame_start||0)/total*100)+'%';
      part.style.width=Math.max(0.5,((Number(seg.frame_end||0)-Number(seg.frame_start||0)+1)/total*100))+'%';
      part.style.background=entityColor(line.entity_id);
      part.title=`track #${seg.track_id}  帧 ${seg.frame_start}-${seg.frame_end}  ${qualityLabel(seg.quality)}`;
      track.appendChild(part);
    });
    (line.occlusion_gaps||[]).forEach(gap=>{
      const g=document.createElement('div');
      g.className='life-gap';
      const from=Number(gap.from_frame||0),to=Number(gap.to_frame||from);
      g.style.left=(from/total*100)+'%';
      g.style.width=Math.max(0.4,((to-from+1)/total*100))+'%';
      g.title=`遮挡/断轨缺口：帧 ${from}-${to}`;
      track.appendChild(g);
    });
    row.appendChild(label);row.appendChild(track);pane.appendChild(row);
  });
}
function selectEvent(ev){
  showEventDetail(ev);
  setRightTab('kg');
  const startValue=Number(ev.frame_start),endValue=Number(ev.frame_end);
  const start=Number.isFinite(startValue)?startValue:Math.round((ev.duration?.[0]||0)*fps)||0;
  const end=Number.isFinite(endValue)?endValue:Math.max(start,Math.round((ev.duration?.[1]||0)*fps));
  startBoundedPlayback(start,end,'event',ev.event_id||'');
}
function showEventDetail(ev){
  const box=document.getElementById('kgEvidence');
  const interactions=ev.interactions||[];
  let h=`<div class="node-card"><b>事件 ${escapeHtml(ev.event_id||'')}</b><br>
    <span class="muted">帧 ${ev.frame_start}-${ev.frame_end}，${ev.entity_count||0} 个目标</span><br>
    ${escapeHtml(ev.summary||'暂无摘要')}</div>`;
  if(interactions.length){
    h+='<div class="muted">本事件交互</div>';
    interactions.forEach(item=>{
      h+=`<div class="kg-edge interacts_with">${escapeHtml((item.entities||[]).join(' ↔ '))}<br><span class="muted">${escapeHtml(item.desc||'')}</span></div>`;
    });
  }
  if(ev.trail&&ev.trail.length){
    h+='<div class="muted">Agent 推理链</div>';
    ev.trail.forEach(t=>{
      h+=`<div class="kg-edge"><b>${escapeHtml(t.agent||'Agent')}</b><br><span class="muted">输入：${escapeHtml(t.input||'')}</span><br>${escapeHtml(t.output||'')}</div>`;
    });
  }
  box.innerHTML=h;
}
function updateFrameScale(){
  document.getElementById('frameScaleLabel').textContent=totalFrames?`0-${Math.max(0,totalFrames-1)} 帧统一基准`:'统一帧号基准';
}
function getTimelineTotal(){
  let maxFrame=totalFrames||0;
  events.forEach(ev=>{maxFrame=Math.max(maxFrame,Number(ev.frame_end)||0);});
  lifelines.forEach(line=>(line.segments||[]).forEach(seg=>{maxFrame=Math.max(maxFrame,Number(seg.frame_end)||0);}));
  return Math.max(1,maxFrame);
}

function renderEvents(){
  const list=document.getElementById('eventList');
  const count=document.getElementById('eventCountText');
  if(!list||!count)return;
  count.textContent=`${events.length} 个事件`;
  if(!events.length){
    list.innerHTML='<div class="panel-empty">分析开始后按时间顺序显示事件片段</div>';
    return;
  }
  list.innerHTML='';
  events.slice().reverse().forEach(ev=>{
    const item=document.createElement('div');
    item.className='event-item';
    item.onclick=()=>selectEvent(ev);
    const start=Number(ev.frame_start)||0,end=Number(ev.frame_end)||start;
    item.innerHTML=`<div class="event-title"><span>${escapeHtml(ev.event_id||'事件')}</span><span class="muted">${start}-${end} 帧</span></div>
      <div class="event-summary">${ev.entity_count||0} 个目标 · ${escapeHtml(String(ev.summary||'暂无摘要').slice(0,120))}</div>`;
    list.appendChild(item);
  });
}

function setRightTab(tab){
  document.querySelectorAll('#rightTabs button').forEach(btn=>btn.classList.toggle('active',btn.dataset.tab===tab));
  document.querySelectorAll('[data-tab-view]').forEach(view=>view.classList.toggle('active',view.dataset.tabView===tab));
  if(tab==='kg')requestAnimationFrame(renderEvidence);
}

// Entities and KG evidence
function renderEntities(){
  const list=document.getElementById('entityList');
  const searchBox=document.getElementById('entitySearch');
  document.getElementById('entityCountText').textContent=`${entities.length} 个实体`;
  if(!entities.length){
    list.innerHTML='<div class="panel-empty">上传视频后显示 Person 档案。点击行查看轨迹和 KG 证据。</div>';
    if(searchBox)searchBox.style.display='none';
    return;
  }
  if(searchBox)searchBox.style.display='';
  list.innerHTML='';
  entities.slice().sort((a,b)=>entityIndex(a.entity_id)-entityIndex(b.entity_id)).forEach(ent=>{
    const item=document.createElement('div');
    item.className='entity-item'+(ent.entity_id===selectedEntity?' active':'');
    item.onclick=()=>selectEntity(ent.entity_id);
    const tracks=(ent.track_ids||[]).join(', ')||'-';
    const summary=decodeMaybeJson(ent.trajectory_summary)||{};
    const points=summary.point_count||0;
    const candidates=(ent.tentative_merge_candidates||[]).join(', ');
    const tentative=(ent.tentative_merge_candidates||[])[0]||'';
    const tentativeText=tentative?`待确认 ${escapeHtml(tentative)}?`:'无候选合并';
    item.innerHTML=`<div class="entity-main"><span class="entity-name" style="color:${entityColor(ent.entity_id)}">${escapeHtml(ent.entity_id)}</span><span class="quality">${qualityLabel(ent.tracking_quality)}</span><button class="highlight-only-btn" type="button" title="仅切换高亮，不跳转">仅高亮</button></div>
      <div class="entity-meta">
        <span>帧 ${ent.first_seen_frame}-${ent.last_seen_frame}</span><span>轨迹点 ${points}</span>
        <span>track ${escapeHtml(tracks)}</span><span>${tentativeText}</span>
      </div>`;
    const highlightBtn=item.querySelector('.highlight-only-btn');
    highlightBtn.onclick=e=>toggleEntityHighlightOnly(e,ent.entity_id);
    list.appendChild(item);
  });
}
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
function renderEvidence(){
  const box=document.getElementById('kgEvidence');
  const nodes=kgGraph.nodes||[],edges=kgGraph.edges||[];
  document.getElementById('kgCountText').textContent=`${nodes.length} 节点 / ${edges.length} 边`;
  const graphEdges=selectedEntity?edges.filter(edge=>{
    const near=edge.src===selectedEntity||edge.tgt===selectedEntity||isNeighborOfSelected(edge);
    return near&&(kgFilter==='all'||edge.edge_type===kgFilter);
  }):edges.filter(edge=>kgFilter==='all'||edge.edge_type===kgFilter).slice(0,28);
  renderKGGraph(nodes, graphEdges);
  if(!selectedEntity){
    box.innerHTML=nodes.length?'<div class="panel-empty">点击左侧实体行，查看关联事件、交互记录和推理路径</div>':'<div class="panel-empty">尚未生成 KG 数据。分析完成后自动构建。</div>';
    return;
  }
  const related=graphEdges;
  const ent=entities.find(e=>e.entity_id===selectedEntity)||{};
  const summary=decodeMaybeJson(ent.trajectory_summary)||{};
  let h=`<div class="node-card"><b style="color:${entityColor(selectedEntity)}">${escapeHtml(selectedEntity)}</b><br>
    <span class="muted">track ${(ent.track_ids||[]).join(', ')||'-'} · ${qualityLabel(ent.tracking_quality)}</span><br>
    <span class="muted">轨迹摘要：${formatTrajectorySummary(summary)}</span></div>`;
  if(!related.length){
    h+='<div class="panel-empty">当前过滤条件下暂无关联边。</div>';
  }else{
    related.forEach(edge=>{
      h+=formatEdge(edge);
    });
  }
  box.innerHTML=h;
}
function renderKGGraph(nodes, edges){
  const canvasEl=document.getElementById('kgCanvas');
  if(!canvasEl)return;
  const rect=canvasEl.parentElement.getBoundingClientRect();
  const dpr=window.devicePixelRatio||1;
  const w=Math.max(260,Math.floor(rect.width)),h=Math.max(160,Math.floor(rect.height));
  canvasEl.width=Math.floor(w*dpr);canvasEl.height=Math.floor(h*dpr);
  const g=canvasEl.getContext('2d');
  g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,w,h);
  const nodeMap=new Map((nodes||[]).map(n=>[n.node_id,n]));
  const ids=new Set();
  (edges||[]).forEach(e=>{ids.add(e.src);ids.add(e.tgt);});
  if(selectedEntity)ids.add(selectedEntity);
  if(!ids.size){
    (nodes||[]).slice(0,18).forEach(n=>ids.add(n.node_id));
  }
  const graphNodes=[...ids].filter(Boolean).slice(0,24).map(id=>nodeMap.get(id)||{node_id:id,node_type:'unknown'});
  const graphNodeIds=new Set(graphNodes.map(n=>n.node_id));
  const graphEdges=(edges||[]).filter(e=>graphNodeIds.has(e.src)&&graphNodeIds.has(e.tgt)).slice(0,36);
  if(!graphNodes.length){
    g.fillStyle='#9aa3af';g.font='12px Microsoft YaHei, sans-serif';g.fillText('暂无图数据',16,28);
    return;
  }
  const cx=w/2,cy=h/2;
  const radius=Math.max(48,Math.min(w,h)/2-34);
  const pos=new Map();
  graphNodes.forEach((node,i)=>{
    if(node.node_id===selectedEntity){
      pos.set(node.node_id,{x:cx,y:cy});
    }else{
      const count=selectedEntity?Math.max(1,graphNodes.length-1):graphNodes.length;
      const offset=selectedEntity?i-1:i;
      const angle=-Math.PI/2+(offset/count)*Math.PI*2;
      pos.set(node.node_id,{x:cx+Math.cos(angle)*radius,y:cy+Math.sin(angle)*radius});
    }
  });
  g.lineWidth=1;
  graphEdges.forEach(edge=>{
    const a=pos.get(edge.src),b=pos.get(edge.tgt);
    if(!a||!b)return;
    g.strokeStyle=edgeColor(edge.edge_type);
    g.globalAlpha=.72;
    g.beginPath();g.moveTo(a.x,a.y);g.lineTo(b.x,b.y);g.stroke();
  });
  g.globalAlpha=1;
  graphNodes.forEach(node=>{
    const p=pos.get(node.node_id);
    const selected=node.node_id===selectedEntity;
    g.beginPath();
    g.fillStyle=selected?'#f0a040':nodeColor(node.node_type);
    g.arc(p.x,p.y,selected?10:8,0,Math.PI*2);g.fill();
    g.strokeStyle=selected?'#fff':'#20242c';g.lineWidth=2;g.stroke();
    g.fillStyle='#e8ebf0';g.font='11px Microsoft YaHei, sans-serif';g.textAlign='center';
    g.fillText(shortNodeLabel(node.node_id),p.x,Math.min(h-8,p.y+22));
  });
}
function isNeighborOfSelected(edge){
  const edges=kgGraph.edges||[];
  const neighbors=new Set();
  edges.forEach(e=>{
    if(e.src===selectedEntity)neighbors.add(e.tgt);
    if(e.tgt===selectedEntity)neighbors.add(e.src);
  });
  return neighbors.has(edge.src)||neighbors.has(edge.tgt);
}
function setKGFilter(filter){
  kgFilter=filter;
  document.querySelectorAll('#kgFilters button').forEach(btn=>btn.classList.toggle('active',btn.dataset.filter===filter));
  renderEvidence();
}
function formatEdge(edge){
  const type=edge.edge_type||'relation';
  const evidence=decodeMaybeJson(edge.evidence_events)||decodeMaybeJson(edge.evidence)||[];
  const count=edge.count||edge.interaction_count||'';
  const label=edgeLabel(type);
  const evText=Array.isArray(evidence)&&evidence.length?`证据事件：${evidence.slice(-3).join(', ')}`:'证据随 KG 边聚合';
  return `<div class="kg-edge ${escapeHtml(type)}"><b>${escapeHtml(edge.src)} → ${escapeHtml(edge.tgt)}</b><br>
    ${escapeHtml(label)} ${count?`· ${count} 次`:''}<br><span class="muted">${escapeHtml(evText)}</span></div>`;
}

// Playback
function ensureCanvasSize(width,height){
  if(canvasSized)return;
  canvas.width=width;canvas.height=height;
  highlightCanvas.width=width;highlightCanvas.height=height;
  canvasSized=true;
}
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
function showFrame(n){
  if(!framePrefix||!totalFrames)return;
  const requestedFrame=clampFrame(n);
  const token=++frameRequestToken;
  playingFrame=requestedFrame;
  const img=new Image();
  img.onload=()=>{
    if(token!==frameRequestToken)return;
    ensureCanvasSize(img.naturalWidth,img.naturalHeight);
    renderCanvasFrame(img,frameBboxes[requestedFrame]||{},requestedFrame);
    renderTimeline();
  };
  img.src=framePrefix+String(requestedFrame).padStart(6,'0')+'.jpg';
  document.getElementById('scrubBar').value=requestedFrame;
}
function togglePlayback(){
  clearPlaybackRange();
  if(playTimer)return;
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
function stopPlay(){
  if(playTimer){clearInterval(playTimer);playTimer=null;}
  const playBtn=document.getElementById('playBtn');
  if(playBtn)playBtn.textContent='播放';
}
function toggleSkeleton(){
  if(!skeletonDataReady)return;
  skeletonEnabled=!skeletonEnabled;
  updateSkeletonButton();
  if(framePrefix)showFrame(playingFrame);
  else redrawCurrentFrame();
}
function updateSkeletonButton(){
  const btn=document.getElementById('skeletonBtn');
  if(!btn)return;
  btn.disabled=!skeletonDataReady;
  btn.classList.toggle('toggle-on',skeletonEnabled);
  btn.textContent=skeletonEnabled?'骨架开':'骨架';
}
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
function drawSkeletonOverlay(frame){
  if(!skeletonEnabled)return;
  const poseFrame=findNearestPoseFrame(frame);
  if(poseFrame==null)return;
  const poses=poseCache[poseFrame]||[];
  poses.forEach(pose=>drawSkeleton(pose,frame));
}
function findNearestPoseFrame(frame){
  let best=null,bestDist=Infinity;
  Object.keys(poseCache).forEach(k=>{
    const f=Number(k),dist=Math.abs(f-frame);
    if(dist<=POSE_MAX_AGE&&dist<bestDist){
      best=f;bestDist=dist;
    }
  });
  return best;
}
function drawSkeleton(pose,frame){
  const kps=pose.keypoints||[];
  if(kps.length<17)return;
  const current=(findPersonsForFrame(frame)||[]).find(person=>
    (pose.entity_id&&person.entity_id===pose.entity_id)||String(person.track_id)===String(pose.track_id)
  );
  const sourceBox=pose.bbox||[],currentBox=current?.bbox||sourceBox;
  if(sourceBox.length!==4||currentBox.length!==4)return;
  const sourceW=Math.max(1,sourceBox[2]-sourceBox[0]),sourceH=Math.max(1,sourceBox[3]-sourceBox[1]);
  const scaleX=(currentBox[2]-currentBox[0])/sourceW,scaleY=(currentBox[3]-currentBox[1])/sourceH;
  const mapPoint=kp=>[
    currentBox[0]+(kp[0]-sourceBox[0])*scaleX,
    currentBox[1]+(kp[1]-sourceBox[1])*scaleY,
  ];
  const color=pose.candidate?'#5ac6c8':'#f0a040';
  ctx.save();
  ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;
  const bones=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
  bones.forEach(([a,b])=>{
    if(kps[a]&&kps[b]&&kps[a][2]>.3&&kps[b][2]>.3){
      const pa=mapPoint(kps[a]),pb=mapPoint(kps[b]);
      ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();
    }
  });
  kps.forEach(kp=>{
    if(kp&&kp[2]>.3){
      const point=mapPoint(kp);
      ctx.beginPath();ctx.arc(point[0],point[1],2,0,Math.PI*2);ctx.fill();
    }
  });
  ctx.restore();
}

// Chat
function addChatMsg(role,html){
  const div=document.createElement('div');div.className='chat-msg '+role;
  const bub=document.createElement('div');bub.className='bubble';bub.innerHTML=html;
  div.appendChild(bub);
  const msgs=document.getElementById('chatMsgs');msgs.appendChild(div);
  msgs.scrollTop=msgs.scrollHeight;
}
function formatEventMsg(ev){
  let h=`<b>[${Number(ev.duration?.[0]||0).toFixed(0)}-${Number(ev.duration?.[1]||0).toFixed(0)}s]</b> ${ev.entity_count||0} 个目标`;
  if(ev.summary)h+=`<br>${escapeHtml(String(ev.summary).substring(0,200))}`;
  if(ev.trail&&ev.trail.length){
    h+=`<br><span class="trail-toggle" onclick="toggleTrail(this)">展开推理链</span><div class="trail-box">`;
    ev.trail.forEach(t=>{h+=`<div class="trail-step"><span class="t-agent">${escapeHtml(t.agent||'Agent')}</span><br>输入: ${escapeHtml(t.input||'')}<br>输出: ${escapeHtml(String(t.output||'').substring(0,120))}</div>`;});
    h+='</div>';
  }
  return h;
}
function toggleTrail(el){
  const box=el.nextElementSibling;
  box.classList.toggle('open');
  el.textContent=box.classList.contains('open')?'收起推理链':'展开推理链';
}
async function sendChat(){
  const input=document.getElementById('chatInput'),q=input.value.trim();
  if(!q||!videoId)return;
  input.value='';addChatMsg('user',escapeHtml(q));
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_id:videoId,question:q})});
  const d=await r.json();addChatMsg('agent',escapeHtml(d.answer||'无回答'));
}

// Canvas / video click
canvas.onclick=handleVideoClick;
const videoEl=document.getElementById('videoPlayer');
videoEl.onclick=e=>handlePointerPick(e,videoEl,videoEl.videoWidth,videoEl.videoHeight);
function handleVideoClick(e){handlePointerPick(e,canvas,canvas.width,canvas.height);}
function handlePointerPick(e,el,naturalW,naturalH){
  const point=mapPointerToFrame(e,el,naturalW,naturalH);
  if(!point){hidePersonCard();return;}
  const people=findPersonsForFrame(playingFrame);
  const hit=pickPersonAt(point.x,point.y,people);
  if(hit){
    const hitPersonId=String(hit.track_id);
    selectedPerson=hitPersonId;
    if(hit.candidate){
      selectedEntity=null;
      renderEntities();renderLifelines();renderEvidence();
      redrawSelectedHighlight();
    }else{
      selectEntity(hit.entity_id||selectedEntity,false);
    }
    showPersonCard(hitPersonId,hit);
    return;
  }
  hidePersonCard();
}
function mapPointerToFrame(e,el,naturalW,naturalH){
  if(!naturalW||!naturalH)return null;
  const rect=el.getBoundingClientRect();
  const scale=Math.min(rect.width/naturalW,rect.height/naturalH);
  const drawW=naturalW*scale,drawH=naturalH*scale;
  const offX=(rect.width-drawW)/2,offY=(rect.height-drawH)/2;
  const localX=e.clientX-rect.left-offX,localY=e.clientY-rect.top-offY;
  if(localX<0||localY<0||localX>drawW||localY>drawH)return null;
  return {x:localX/scale,y:localY/scale};
}
function findPersonsForFrame(frame){
  if(framePersons[frame]&&framePersons[frame].length)return framePersons[frame];
  let bestFrame=null,bestDist=Infinity;
  Object.keys(framePersons).forEach(k=>{
    const dist=Math.abs(Number(k)-frame);
    if(dist<bestDist&&dist<=3){bestDist=dist;bestFrame=Number(k);}
  });
  if(bestFrame!=null)return framePersons[bestFrame];
  return Object.values(personMap);
}
function pickPersonAt(x,y,people){
  const hits=[];
  (people||[]).forEach(p=>{
    const b=p.bbox; if(!b||b.length!==4)return;
    const w=Math.max(1,b[2]-b[0]),h=Math.max(1,b[3]-b[1]);
    const pad=Math.max(8,Math.min(24,Math.max(w,h)*0.12));
    if(x>=b[0]-pad&&x<=b[2]+pad&&y>=b[1]-pad&&y<=b[3]+pad){
      hits.push({person:p,area:w*h});
    }
  });
  hits.sort((a,b)=>a.area-b.area);
  return hits[0]?.person||null;
}
function showPersonCard(tid,p){
  const eid=p.entity_id||`Track-${tid}`;
  const card=document.getElementById('personCard');
  card.style.display='';
  if(p.candidate){
    const title=p.tentative_entity?escapeHtml(eid):'候选检测';
    const subtitle=p.tentative_entity?'展示层 Tentative Entity，暂不进入 Agent/KG':'未稳定成 Entity';
    document.getElementById('personContent').innerHTML=`<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      <span class="ent-id" style="color:#5ac6c8;font-size:15px">${title}</span>
      <span style="font-size:10px;color:var(--muted);background:var(--bg);padding:1px 6px;border-radius:3px">${subtitle}</span>
    </div>
    <div class="motion">${p.tentative_entity?'该候选已连续出现，已开启展示层骨架；正式分析仍等待 tracker 确认。':'YOLO 已检测到该目标，但 tracker 尚未把它确认成稳定 Person。它只用于画面辅助，不进入 Agent/KG。'}</div>
    <div style="margin-top:8px;color:var(--muted);font-size:10px">连续命中：${p.candidate_hits||1} · 置信度：${Number(p.confidence||0).toFixed(2)} · bbox: [${(p.bbox||[]).map(v=>Math.round(v)).join(', ')}]</div>`;
    return;
  }
  let md=null;
  for(let i=events.length-1;i>=0;i--){
    const found=(events[i].motions||[]).find(m=>m.entity_id===eid);
    if(found&&found.data){md=found.data;break;}
  }
  const ent=entities.find(e=>e.entity_id===eid)||{};
  const color=entityColor(eid);
  let h=`<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
    <span class="ent-id" style="color:${color};font-size:15px">${escapeHtml(eid)}</span>
    <span style="font-size:10px;color:var(--muted);background:var(--bg);padding:1px 6px;border-radius:3px">track #${escapeHtml(String(tid))}</span>
  </div>`;
  h+=`<div style="color:var(--muted);font-size:11px;margin-bottom:8px">质量：${qualityLabel(ent.tracking_quality)} · 帧 ${ent.first_seen_frame??'-'}-${ent.last_seen_frame??'-'}</div>`;
  if(md){
    const rows=[
      ['运动标签',`<b style="color:${color}">${escapeHtml(md.label||'?')}</b>`],
      ['速度',`${escapeHtml(String(md.speed_px||0))} px/s`],
      ['速度百分位',`P${md.speed_pct!=null?escapeHtml(String(md.speed_pct)):'-'}`],
      ['方向变化',`${escapeHtml(String(md.direction_change||0))} 次`],
      ['姿态',escapeHtml(md.pose||'-')],
      ['持续',`${escapeHtml(String(md.duration||0))}s`],
      ['置信度',`${(((md.confidence||0)*100)).toFixed(0)}%`],
    ];
    h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:11px">';
    rows.forEach(([k,v])=>{h+=`<div style="color:var(--muted)">${k}</div><div style="text-align:right">${v}</div>`;});
    h+='</div>';
  }else{
    h+='<div class="motion">暂无 Agent 运动描述，已保留检测框和 Entity 证据。</div>';
  }
  h+=`<div style="margin-top:8px;color:var(--muted);font-size:10px">bbox: [${(p.bbox||[]).map(v=>Math.round(v)).join(', ')}]</div>`;
  document.getElementById('personContent').innerHTML=h;
}
function hidePersonCard(){document.getElementById('personCard').style.display='none';selectedPerson=null;}

// Helpers
function setStatus(st,text){
  const dot=document.getElementById('statusDot');
  dot.className='status-dot '+st;
  document.getElementById('statusText').textContent=text;
}
function entityIndex(id){const m=String(id||'').match(/\d+/);return m?parseInt(m[0],10):9999;}
function entityColor(id){return ENTITY_COLORS[entityIndex(id)%ENTITY_COLORS.length];}
function qualityLabel(q){
  const labels={stable:'稳定',occluded:'遮挡重连',fragmented:'疑似断裂',tentative:'候选合并'};
  return labels[q]||'未知';
}
function edgeLabel(t){
  const labels={belongs_to:'属于事件',interacts_with:'存在交互',matches:'匹配证据',next:'时序相邻',violates:'触发规则'};
  return labels[t]||t;
}
function edgeColor(t){
  const colors={belongs_to:'#f0a040',interacts_with:'#4caf91',matches:'#a27bd8',next:'#5ac6c8',violates:'#e05a68'};
  return colors[t]||'#5b9bd5';
}
function nodeColor(t){
  const colors={Entity:'#5b9bd5',Event:'#f0a040',Behavior:'#a27bd8',Scene:'#4caf91'};
  return colors[t]||'#9aa3af';
}
function shortNodeLabel(id){
  const text=String(id||'');
  return text.length>16?text.slice(0,13)+'...':text;
}
function decodeMaybeJson(value){
  if(value==null)return value;
  if(typeof value!=='string')return value;
  const text=value.trim();
  if(!text||!['{','['].includes(text[0]))return value;
  try{return JSON.parse(text);}catch(_){return value;}
}
function formatTrajectorySummary(summary){
  if(!summary||typeof summary!=='object')return '暂无';
  const parts=[];
  if(summary.point_count!=null)parts.push(`${summary.point_count} 点`);
  if(Array.isArray(summary.activity_bbox))parts.push(`活动框 [${summary.activity_bbox.map(v=>Math.round(v)).join(', ')}]`);
  return parts.join('，')||'暂无';
}
function formatEventTooltip(ev){
  return `帧 ${ev.frame_start}-${ev.frame_end}<br>${ev.entity_count||0} 个目标<br>${escapeHtml(ev.summary||'点击查看事件详情')}`;
}
function escapeHtml(s){
  return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// Resize
(function(){
  const rp=document.getElementById('rightPanel');
  const savedW=localStorage.getItem('scenesense:panelW');
  if(savedW)rp.style.width=savedW+'px';
  document.getElementById('resizeMain').addEventListener('mousedown',e=>{
    e.preventDefault();document.getElementById('resizeMain').classList.add('active');
    const sx=e.clientX,sw=rp.offsetWidth;
    const onMove=ev=>{const w=Math.max(300,sw+sx-ev.clientX);rp.style.width=w+'px';};
    const onUp=()=>{document.getElementById('resizeMain').classList.remove('active');document.removeEventListener('mousemove',onMove);document.removeEventListener('mouseup',onUp);localStorage.setItem('scenesense:panelW',rp.offsetWidth);};
    document.addEventListener('mousemove',onMove);document.addEventListener('mouseup',onUp);
  });

  const ep=document.getElementById('entityPanel');
  const savedH=localStorage.getItem('scenesense:entityH');
  if(savedH)ep.style.flexBasis=savedH+'px';
  document.getElementById('resizeAgent').addEventListener('mousedown',e=>{
    e.preventDefault();document.getElementById('resizeAgent').classList.add('active');
    const sy=e.clientY,sh=ep.offsetHeight;
    const onMove=ev=>{const h=Math.max(150,sh+ev.clientY-sy);ep.style.flexBasis=h+'px';};
    const onUp=()=>{document.getElementById('resizeAgent').classList.remove('active');document.removeEventListener('mousemove',onMove);document.removeEventListener('mouseup',onUp);localStorage.setItem('scenesense:entityH',ep.offsetHeight);};
    document.addEventListener('mousemove',onMove);document.addEventListener('mouseup',onUp);
  });
})();

// Keyboard shortcuts
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
  switch(e.key){
    case 'ArrowLeft': if(totalFrames)showFrame(Math.max(0,playingFrame-1));e.preventDefault();break;
    case 'ArrowRight': if(totalFrames)showFrame(Math.min(totalFrames-1,playingFrame+1));e.preventDefault();break;
    case ' ': togglePlayback();e.preventDefault();break;
    case 'Escape': selectedEntity=null;selectedPerson=null;renderEntities();renderLifelines();renderEvidence();redrawSelectedHighlight();hidePersonCard();break;
  }
});

// Entity search filter
function filterEntities(q){
  const s=q.trim().toLowerCase();
  const list=document.getElementById('entityList');
  if(!list)return;
  const items=list.querySelectorAll('.entity-item');
  items.forEach(item=>{
    const name=(item.querySelector('.entity-name')?.textContent||'').toLowerCase();
    const meta=(item.querySelector('.entity-meta')?.textContent||'').toLowerCase();
    item.style.display=(!s||name.includes(s)||meta.includes(s))?'':'none';
  });
}
