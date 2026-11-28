"""Web 后端接口：静态页、上传、/chat 兜底、WebSocket 错误分支。

不触碰真实视频与模型：上传只写文件，/chat 在 KG 缺失时直接返回提示。
"""

from fastapi.testclient import TestClient

from web.backend.main import KG_DIR, UPLOAD_DIR, app

client = TestClient(app)

FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128


def test_index_serves_frontend_shell():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SceneSense" in response.text
    assert "/static/app.js" in response.text


def test_upload_returns_video_id_and_persists_file():
    response = client.post(
        "/upload",
        files={"file": ("demo.mp4", FAKE_MP4, "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "demo.mp4"
    assert payload["size"] == len(FAKE_MP4)
    assert len(payload["video_id"]) == 12

    saved = UPLOAD_DIR / f"{payload['video_id']}.mp4"
    try:
        assert saved.exists()
        assert saved.read_bytes() == FAKE_MP4
    finally:
        saved.unlink(missing_ok=True)


def test_chat_without_video_id_asks_for_input():
    response = client.post("/chat", json={"question": "谁在跑动"})

    assert response.status_code == 200
    assert "请提供" in response.json()["answer"]


def test_chat_without_generated_kg_returns_waiting_message():
    (KG_DIR / "missing_graph.graphml").unlink(missing_ok=True)

    response = client.post(
        "/chat", json={"video_id": "missing_graph", "question": "场景里有多少人"}
    )

    assert response.status_code == 200
    assert "尚未完成分析" in response.json()["answer"]


def test_websocket_reports_missing_upload():
    with client.websocket_connect("/ws/notuploaded") as ws:
        message = ws.receive_json()

    assert message["type"] == "error"
    assert "视频未找到" in message["message"]
