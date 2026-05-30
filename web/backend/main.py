"""FastAPI 后端：上传 + WebSocket 推送 + QA 对话。"""

import asyncio
import os
import uuid
from pathlib import Path

import networkx as nx
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .pipeline_runner import run_pipeline

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
KG_DIR = Path(__file__).resolve().parent.parent.parent / "kg_outputs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
KG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="YOLO + Multi-Agent Scene Understanding")

# 静态文件（前端）
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# 活跃的 pipeline 任务
active_tasks: dict[str, asyncio.Task] = {}
# 最近的 KG graph（用于 QA）
video_kgs: dict[str, str] = {}  # video_id → graphml_path


@app.get("/")
async def index():
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    video_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    path = UPLOAD_DIR / f"{video_id}{ext}"
    content = await file.read()
    path.write_bytes(content)
    return {"video_id": video_id, "filename": file.filename, "size": len(content)}


@app.websocket("/ws/{video_id}")
async def ws_pipeline(ws: WebSocket, video_id: str):
    await ws.accept()

    # 找上传的文件
    files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not files:
        await ws.send_json({"type": "error", "message": "视频未找到"})
        await ws.close()
        return

    video_path = str(files[0])

    try:
        await run_pipeline(video_path, video_id, ws)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        await ws.close()


@app.post("/chat")
async def chat(req: dict):
    """QA 对话接口。加载对应视频的 KG 回答问题。"""
    video_id = req.get("video_id", "")
    question = req.get("question", "")

    if not video_id or not question:
        return {"answer": "请提供 video_id 和 question。"}

    graphml = KG_DIR / f"{video_id}.graphml"
    if not graphml.exists():
        return {"answer": "视频尚未完成分析，KG 未生成。请等待 pipeline 完成后再提问。"}

    from scripts.qa_repl import KGQA
    G = nx.read_graphml(str(graphml))

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    llm = None
    if api_key:
        from agents.llm_client import LLMClient
        llm = LLMClient(api_key=api_key)

    qa = KGQA(G, llm)
    answer = qa.answer(question)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
