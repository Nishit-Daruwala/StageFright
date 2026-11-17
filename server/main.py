"""StageFright AI server: serves the web front end and the live audio WebSocket."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from . import config
from .models import Models
from .session import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpx2", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("stagefright")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
models = Models()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loader = asyncio.create_task(models.load_all())     # the page is usable while models load
    yield
    loader.cancel()


app = FastAPI(title="StageFright AI", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": models.status, "llm": bool(config.GROQ_API_KEY)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws, models)

    async def report_models() -> None:
        while True:
            await session.send({"type": "models", "status": models.status, "ready": models.ready,
                                "llm": bool(config.GROQ_API_KEY)})
            if models.settled:
                return
            await asyncio.sleep(1.0)

    session.spawn(report_models())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await session.on_audio(msg["bytes"])
            elif msg.get("text"):
                await session.on_event(json.loads(msg["text"]))
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()


# Mounted last so the routes above take priority.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
