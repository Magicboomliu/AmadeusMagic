"""
FastAPI 路由：提供 Amadeus 的 HTTP 接口。

- POST /chat：纯文字对话（可带历史），返回文字回复
- POST /chat/voice：上传语音 -> STT -> RAG+LLM -> TTS，返回语音 MP3
- POST /memory：添加一条记忆到 RAG
- GET /health：健康检查
"""

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .config import get_settings
from .llm import chat as llm_chat
from .rag import EmbeddingUnavailableError, add_memory
from .stt import speech_to_text
from .tts import text_to_speech

router = APIRouter(prefix="/api", tags=["amadeus"])


@router.get("/health")
def health() -> dict[str, str]:
    """健康检查，用于部署与负载均衡。"""
    return {"status": "ok", "service": "amadeus-mvp"}


@router.post("/chat")
def chat_text(
    message: str = Form(..., description="用户当前消息"),
    history: str | None = Form(None, description="JSON 数组字符串，可选历史 [{\"role\":\"user\"|\"assistant\",\"content\":\"...\"}]"),
) -> dict[str, Any]:
    """
    文字对话：传入当前消息与可选历史，返回 Amadeus 的文字回复。
    """
    hist: list[dict] = []
    if history:
        try:
            import json
            hist = json.loads(history)
        except Exception:
            raise HTTPException(status_code=400, detail="history 必须是合法 JSON 数组")
    reply = llm_chat(message, history=hist or None)
    return {"reply": reply, "role": "assistant"}


@router.post("/chat/voice")
async def chat_voice(
    audio: UploadFile = File(..., description="用户语音文件（如 webm/mp3/wav）"),
) -> Response:
    """
    语音对话：上传一段语音 -> STT 转文字 -> RAG+LLM 生成回复 -> TTS 转语音，返回 MP3。
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="请上传有效的音频文件")

    # 根据上传文件名推断格式，供 Whisper 参考
    filename = audio.filename or "audio.webm"
    if not filename.lower().endswith((".webm", ".mp3", ".wav", ".m4a", ".ogg", ".mp4")):
        filename = filename + ".webm"

    text_input = speech_to_text(raw, filename=filename)
    if not text_input.strip():
        raise HTTPException(status_code=400, detail="无法识别语音内容，请重试或换一种格式")

    reply_text = llm_chat(text_input, history=None)
    if not reply_text.strip():
        reply_text = "抱歉，我这边没有生成出回复，请再试一次。"

    mp3_bytes = text_to_speech(reply_text)
    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=amadeus_reply.mp3"},
    )


@router.post("/memory")
def memory_add(
    content: str = Form(..., description="要存入 Amadeus 记忆的文本"),
) -> dict[str, Any]:
    """添加一条记忆到 RAG 知识库，后续对话可被检索到。"""
    try:
        mem_id = add_memory(content)
        return {"id": mem_id, "message": "记忆已添加"}
    except EmbeddingUnavailableError as e:
        # 当 Embedding 不可用时，明确告知用户需要配置 embedding，避免 500
        raise HTTPException(status_code=503, detail=str(e))
