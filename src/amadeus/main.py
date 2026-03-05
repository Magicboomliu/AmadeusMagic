"""
Amadeus MVP 应用入口：挂载 API 路由并启动 FastAPI 服务。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预加载 RAG 集合（可选）。"""
    # 首次请求时会懒加载 Chroma 与默认记忆，这里仅作占位
    yield
    # 关闭时若有资源可在此释放
    pass


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    app = FastAPI(
        title="Amadeus MVP API",
        description="Steins;Gate 风格的 AI 助手：RAG 记忆 + LLM 对话 + TTS/语音",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()


def run():
    """通过 `uv run python -m amadeus.main` 启动服务。"""
    import uvicorn
    s = get_settings()
    uvicorn.run(
        "amadeus.main:app",
        host=s.host,
        port=s.port,
        reload=True,
    )


if __name__ == "__main__":
    run()
