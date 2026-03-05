# Amadeus MVP

> 灵感来自《命运石之门 0》中的 **Amadeus**：基于记忆与人格数据的 AI 助手，支持自然对话与语音交互。

本仓库是一个 **MVP（最小可行产品）** 的 Python 后端实现，结合当前常见的 **AI（LLM）**、**RAG（检索增强）**、**TTS（语音合成）** 与 **语音输入（STT）** 技术，实现「能记忆、能对话、能说能听」的 Amadeus 风格助手。

## 功能概览

| 能力       | 实现方式说明 |
|------------|--------------|
| **记忆/RAG** | 使用 Chroma 存储文档向量，OpenAI Embedding 检索，对话时注入相关记忆作为上下文 |
| **对话/LLM** | OpenAI 兼容 API（如 gpt-4o-mini），带 Amadeus 人设的系统提示 + RAG 上下文 |
| **语音输入** | 用户上传音频 → Whisper（STT）转文字 → 再走上述对话流程 |
| **语音输出** | 回复文字 → TTS 转语音（OpenAI TTS 或免费 **edge-tts**）→ 返回 MP3 |

## 技术栈

- **Python 3.11+**，依赖由 **uv** 管理
- **FastAPI**：HTTP API
- **OpenAI 兼容 API**：
  - LLM 可指向 Groq / OpenRouter / 本地代理等
  - Embedding / Whisper / TTS 一般使用 OpenAI 官方（费用极低）
- **Chroma**：向量存储与检索（RAG）
- **edge-tts**：可选免费 TTS 后端

## 快速开始

### 1. 环境准备

- 已安装 [uv](https://docs.astral.sh/uv/)（若未安装：`curl -LsSf https://astral.sh/uv/install.sh | sh`）
- 拥有 **OpenAI API Key**（或兼容 OpenAI 的 API 端点）

### 2. 克隆与依赖

```bash
cd AmadeusMagic
uv sync
```

### 3. 配置

```bash
cp .env.example .env
```

常见推荐配置：**聊天用 Groq 免费，Embedding/STT/TTS 用 OpenAI**：

```env
# 通用回退（可选）
OPENAI_API_KEY=sk-your-openai-key

# LLM：Groq 免费
LLM_API_KEY=groq_your_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile

# Embedding / STT / TTS：OpenAI 官方
EMBEDDING_API_KEY=sk-your-openai-key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

STT_API_KEY=sk-your-openai-key
STT_BASE_URL=https://api.openai.com/v1
STT_MODEL=whisper-1

TTS_API_KEY=sk-your-openai-key
TTS_BASE_URL=https://api.openai.com/v1
TTS_MODEL=tts-1
TTS_VOICE=alloy
```

可选：使用免费 TTS（不消耗 OpenAI 额度）：

```env
USE_EDGE_TTS=true
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
```

### 4. 启动服务

```bash
uv run python run.py
```

或：

```bash
uv run uvicorn amadeus.main:app --reload --host 0.0.0.0 --port 8000
```

服务默认在 **http://127.0.0.1:8000**，文档在 **http://127.0.0.1:8000/docs**。

## API 说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/health` | 健康检查 |
| POST | `/api/chat` | 文字对话：`message`（必填）+ 可选 `history`（JSON 数组） |
| POST | `/api/chat/voice` | 语音对话：上传音频文件，返回 Amadeus 回复的 MP3 |
| POST | `/api/memory` | 添加一条记忆：`content`（必填），写入 RAG 供后续检索 |

### 文字对话示例

```bash
curl -X POST "http://127.0.0.1:8000/api/chat" \
  -F "message=你好，你是谁？"
```

### 添加记忆示例

```bash
curl -X POST "http://127.0.0.1:8000/api/memory" \
  -F "content=用户最喜欢的颜色是蓝色"
```

### 语音对话示例

```bash
# 上传录音文件，返回为 MP3 流
curl -X POST "http://127.0.0.1:8000/api/chat/voice" \
  -F "audio=@your_voice.webm" \
  --output reply.mp3
```

## 项目结构

```
AmadeusMagic/
├── pyproject.toml          # 项目与 uv 依赖
├── run.py                   # 启动入口
├── .env.example             # 环境变量示例
├── README.md
└── src/
    └── amadeus/
        ├── __init__.py
        ├── config.py        # 配置（pydantic-settings）
        ├── rag.py           # RAG：Chroma + Embedding，记忆检索与写入
        ├── llm.py           # LLM 对话（系统提示 + RAG 上下文）
        ├── tts.py           # TTS：OpenAI / edge-tts
        ├── stt.py           # STT：Whisper
        ├── api.py           # FastAPI 路由
        └── main.py          # FastAPI 应用与 CORS、lifespan
```

首次请求会懒加载 RAG 并写入默认的 Amadeus 人格记忆；之后可通过 `/api/memory` 继续追加记忆。

## 配置项说明（.env）

| 变量 | 说明 | 默认 |
|------|------|------|
| `OPENAI_API_KEY` | 通用回退用的 API Key | 空 |
| `OPENAI_BASE_URL` | 通用回退用的 base URL | 空 |
| `LLM_API_KEY` / `LLM_BASE_URL` | 聊天 LLM 的 Key 和 URL（可指向 Groq 等） | 回退到 `OPENAI_*` |
| `LLM_MODEL` | 对话模型名 | gpt-4o-mini |
| `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` | Embedding 所用 Key 和 URL | 回退到 `OPENAI_*` |
| `EMBEDDING_MODEL` | 嵌入模型名 | text-embedding-3-small |
| `TTS_API_KEY` / `TTS_BASE_URL` | TTS 所用 Key 和 URL | 回退到 `OPENAI_*` |
| `TTS_MODEL` / `TTS_VOICE` | TTS 模型与音色 | tts-1 / alloy |
| `STT_API_KEY` / `STT_BASE_URL` | Whisper 所用 Key 和 URL | 回退到 `OPENAI_*` |
| `STT_MODEL` | Whisper 模型名 | whisper-1 |
| `USE_EDGE_TTS` | 是否用 edge-tts 做 TTS | false |
| `EDGE_TTS_VOICE` | edge-tts 音色 | zh-CN-XiaoxiaoNeural |
| `CHROMA_PERSIST_DIR` | Chroma 数据目录 | ./data/chroma |
| `HOST` / `PORT` | 服务监听地址与端口 | 0.0.0.0 / 8000 |

## 许可证

见 [LICENSE](LICENSE) 文件。

---

*El Psy Kongroo.*
