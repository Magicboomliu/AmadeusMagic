"""
TTS（文本转语音）模块：Amadeus 的「声音」输出。

支持两种后端：
1. OpenAI TTS API（需 API Key，音质好）
2. edge-tts（免费，微软 Edge 在线语音）
"""

import io
from pathlib import Path

from .config import get_settings


def tts_openai(text: str) -> bytes:
    """使用 OpenAI 兼容 TTS 将文本转为语音，返回 MP3 字节。"""
    from openai import OpenAI

    s = get_settings()
    client = OpenAI(
        api_key=s.tts_key(),
        base_url=s.tts_url(),
    )
    resp = client.audio.speech.create(
        model=s.tts_model,
        voice=s.tts_voice,
        input=text,
    )
    return resp.content


def tts_edge(text: str) -> bytes:
    """使用 edge-tts 将文本转为语音，返回 MP3 字节。"""
    import edge_tts

    s = get_settings()
    voice = s.edge_tts_voice
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    communicate.save(buf)
    return buf.getvalue()


def text_to_speech(text: str) -> bytes:
    """
    根据配置选择 TTS 后端，将文本转为语音。
    返回 MP3 格式的字节流，可直接作为 HTTP 响应体。
    """
    s = get_settings()
    if s.use_edge_tts:
        return tts_edge(text)
    return tts_openai(text)
