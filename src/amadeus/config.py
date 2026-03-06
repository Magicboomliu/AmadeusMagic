"""
配置管理：从环境变量读取 API Key、模型名等。
使用 pydantic-settings，支持 .env 文件。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，未设置时从环境变量或 .env 读取。

    为了支持「聊天走 Groq 免费，Embedding/STT/TTS 继续用 OpenAI」的场景，
    这里将各能力的 API Key / Base URL / 模型名拆开配置，并保留向后兼容：
    - 若对应子模块未单独配置，则回退到 openai_* 通用配置。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 通用 OpenAI 兼容配置（旧版配置仍然可用）
    openai_api_key: str = ""
    openai_base_url: str | None = None

    # LLM（可指向 Groq、OpenRouter 等 OpenAI 兼容服务）
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"

    # Embedding（通常建议继续用 OpenAI，费用很低）
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"

    # 本地 Embedding（免费）：开启后将完全不调用 Embedding API（用于解决 OpenAI 配额不足）
    use_local_embeddings: bool = False
    # 中文/多语都不错的默认模型；首次运行会自动下载到本机缓存目录
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # TTS（默认用 OpenAI TTS；也可以指向兼容服务）
    tts_api_key: str | None = None
    tts_base_url: str | None = None
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer

    # STT / Whisper
    stt_api_key: str | None = None
    stt_base_url: str | None = None
    stt_model: str = "whisper-1"

    # RAG：Chroma 持久化目录
    chroma_persist_dir: str = "./data/chroma"

    # 是否使用免费 TTS（edge-tts）代替 OpenAI TTS
    use_edge_tts: bool = False
    edge_tts_voice: str = "zh-CN-XiaoxiaoNeural"  # 中文女声

    # 服务
    host: str = "0.0.0.0"
    port: int = 8000

    # -------- 工具方法：带回退逻辑的取值 --------

    def llm_key(self) -> str | None:
        """LLM 使用的 API Key，优先 llm_api_key，其次 openai_api_key。"""
        return self.llm_api_key or (self.openai_api_key or None)

    def llm_url(self) -> str | None:
        """LLM 使用的 base URL，优先 llm_base_url，其次 openai_base_url。"""
        return self.llm_base_url or self.openai_base_url

    def embedding_key(self) -> str | None:
        """Embedding 使用的 API Key。"""
        return self.embedding_api_key or (self.openai_api_key or None)

    def embedding_url(self) -> str | None:
        """Embedding 使用的 base URL。"""
        return self.embedding_base_url or self.openai_base_url

    def tts_key(self) -> str | None:
        """TTS 使用的 API Key。"""
        return self.tts_api_key or (self.openai_api_key or None)

    def tts_url(self) -> str | None:
        """TTS 使用的 base URL。"""
        return self.tts_base_url or self.openai_base_url

    def stt_key(self) -> str | None:
        """STT 使用的 API Key。"""
        return self.stt_api_key or (self.openai_api_key or None)

    def stt_url(self) -> str | None:
        """STT 使用的 base URL。"""
        return self.stt_base_url or self.openai_base_url


def get_settings() -> Settings:
    """获取单例配置（懒加载）。"""
    return Settings()
