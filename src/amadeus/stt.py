"""
STT（语音转文字）模块：将用户上传的语音转为文本，供 LLM 与 RAG 使用。

使用 OpenAI Whisper API（或兼容接口）进行识别。
"""

import io

from openai import OpenAI

from .config import get_settings


def speech_to_text(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """
    将语音二进制数据转为文字。

    :param audio_bytes: 音频文件内容（支持常见格式如 mp3, wav, webm, m4a 等）
    :param filename: 用于提示 Whisper 格式的扩展名，不影响解析结果
    :return: 识别出的文本
    """
    s = get_settings()
    client = OpenAI(
        api_key=s.stt_key(),
        base_url=s.stt_url(),
    )

    # Whisper API 要求以 multipart 形式传文件，这里用 io.BytesIO 模拟文件
    file_like = io.BytesIO(audio_bytes)
    file_like.name = filename

    resp = client.audio.transcriptions.create(
        model=s.stt_model,
        file=file_like,
        response_format="text",
    )
    return (resp if isinstance(resp, str) else getattr(resp, "text", "")) or ""
