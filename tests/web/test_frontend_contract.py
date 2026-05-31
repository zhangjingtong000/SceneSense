from kg.continuity import write_continuity_event
from kg.graph_store import GraphStore
from perception.entity_profile import EntityProfileManager
from pipeline import strip_tracker_debug_fields as cli_strip_tracker_debug_fields
from web.backend.pipeline_runner import strip_tracker_debug_fields as web_strip_tracker_debug_fields
from pathlib import Path

import numpy as np


def track(track_id, frame, feature=None):
    return {
        "track_id": track_id,
        "bbox": [10, 10, 30, 60],
        "centroid": [20 + frame, 35],
        "trajectory": [[20 + frame, 35, frame / 30.0]],
        "appearance_feature": feature or [1.0, 0.0],
    }


def test_entity_payload_contains_chinese_ui_ready_fields():
    mgr = EntityProfileManager()
    enriched = mgr.update([track(1, 1)], frame_idx=1, timestamp=1 / 30.0)

    assert enriched[0]["entity_id"] == "Person-1"
    assert mgr.to_entities()[0]["tracking_quality"] == "stable"
    assert mgr.to_lifelines()[0]["segments"][0]["frame_start"] == 1


def test_pipeline_safe_tracklets_keep_entity_and_strip_debug_feature():
    tracked = [track(1, 1)]
    tracked[0]["entity_id"] = "Person-1"

    cli_safe = cli_strip_tracker_debug_fields(tracked)
    web_safe = web_strip_tracker_debug_fields(tracked)

    assert cli_safe[0]["entity_id"] == "Person-1"
    assert web_safe[0]["entity_id"] == "Person-1"
    assert "appearance_feature" not in cli_safe[0]
    assert "appearance_feature" not in web_safe[0]
    assert "appearance_feature" in tracked[0]


def test_continuity_payload_uses_person_ids_for_frontend_graph():
    mgr = EntityProfileManager()
    enriched = mgr.update(
        [track(1, 1, [1.0, 0.0]), track(2, 1, [0.0, 1.0])],
        frame_idx=1,
        timestamp=1 / 30.0,
    )
    event = {
        "event_id": "Event-frontend",
        "timestamp": 1.0,
        "duration": [0.0, 2.0],
        "frame_start": 0,
        "frame_end": 60,
        "tracklets": enriched,
        "spatial_relations": [
            {
                "track_1": 1,
                "track_2": 2,
                "distance_m": 20.0,
                "relation_type": "near",
                "duration_s": 1.5,
            }
        ],
        "zone_id": "zone_default",
        "zone_type": "full_frame",
    }
    graphstore = GraphStore()

    write_continuity_event(graphstore, event, mgr.to_entities())
    graph = graphstore.to_dict()

    assert any(node["node_id"] == "Person-1" for node in graph["nodes"])
    assert any(edge["src"] == "Person-1" and edge["tgt"] == "Event-frontend" for edge in graph["edges"])


def test_web_display_person_propagates_tentative_predecessor():
    from web.backend.pipeline_runner import make_annotated_frame

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracked = [{
        **track(2, 2),
        "entity_id": "Person-2",
        "tentative_predecessor": "Person-1",
    }]

    _, persons = make_annotated_frame(frame, tracked)

    assert persons[0]["tentative_predecessor"] == "Person-1"


def test_frontend_bounded_playback_and_overlay_highlight_contract():
    html = Path("web/frontend/index.html").read_text(encoding="utf-8")
    js = Path("web/frontend/app.js").read_text(encoding="utf-8")
    source = html + "\n" + js

    toggle_playback = js[
        js.index("function togglePlayback(){"):js.index("function playFrom(n){")
    ]
    select_event = js[
        js.index("function selectEvent(ev){"):js.index("function showEventDetail(ev){")
    ]
    highlight_only = js[
        js.index("function toggleEntityHighlightOnly(e,entityId){"):js.index("function selectEntity(entityId,jump=true){")
    ]
    play_from = js[
        js.index("function playFrom(n){"):js.index("function stopPlay(){")
    ]

    assert 'id="highlightCanvas"' in html
    assert "let playbackRange=null;" in source
    assert "startBoundedPlayback(start,end,'event',ev.event_id||'');" in source
    assert "startBoundedPlayback(start,end,'entity',entityId);" in source
    assert "Number(ent.first_seen_frame)" in source
    assert "Number(ent.last_seen_frame)" in source
    assert "highlightBtn.onclick=e=>toggleEntityHighlightOnly(e,ent.entity_id);" in source
    assert "e.stopPropagation();" in source
    assert "clearPlaybackRange();" in source
    assert "const endFrame=playbackRange?playbackRange.endFrame:totalFrames-1;" in source
    assert "drawHighlight(highlightCtx,bbox,selectedEntity||selectedPerson||'');" in source
    assert "ctx.fillText(String(label),labelX+5,labelY);" not in source
    assert "待确认合并:" in source
    assert toggle_playback.index("clearPlaybackRange();") < toggle_playback.index("if(playTimer)return;")
    assert "if(playTimer){stopPlay();return;}" not in toggle_playback
    assert "const startValue=Number(ev.frame_start),endValue=Number(ev.frame_end);" in select_event
    assert "Number.isFinite(startValue)?startValue" in select_event
    assert "Number.isFinite(endValue)?endValue" in select_event
    assert "".join(highlight_only.split()) == (
        "functiontoggleEntityHighlightOnly(e,entityId){"
        "e.stopPropagation();"
        "toggleEntityHighlight(entityId);"
        "}"
    )
    assert "showFrame(" not in highlight_only
    assert "playFrom(" not in highlight_only
    assert "startBoundedPlayback(" not in highlight_only
    assert "showFrame(Math.min(endFrame,playingFrame+1));" in play_from
    assert "if(playingFrame>=endFrame){stopPlay();return;}" in play_from
    assert "seg.onclick=()=>selectEvent(ev);" in source
    assert "item.onclick=()=>selectEvent(ev);" in source
