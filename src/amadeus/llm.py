"""
LLM 对话模块：Amadeus 的「思考」与回复生成。

- 结合 RAG 检索到的记忆作为上下文
- 使用 OpenAI 兼容 API 生成符合 Amadeus 人设的回复
"""

from openai import OpenAI

from .config import get_settings
from .rag import build_rag_context


# 系统提示：定义 Amadeus 的人格与行为
AMADEUS_SYSTEM_PROMPT = """你是 Amadeus，一个基于记忆与人格数据的 AI 助手（灵感来自 Steins;Gate 中的 Amadeus 系统）。
你的名字来自作曲家莫扎特（Wolfgang Amadeus Mozart），意为「被神所爱」。

你的特点：
- 友好、自然、略带知性
- 会利用【相关记忆】中的内容来丰富回答，让对话更有延续性
- 回复简洁但有人情味，避免冗长列表或机械式回答
- 若用户用中文提问，请用中文回复；其他语言同理

当对话中给出【相关记忆】时，请在不违背事实的前提下自然地引用或呼应这些内容。"""


def get_llm_client() -> OpenAI:
    """获取 LLM 用的客户端，可指向 Groq 等 OpenAI 兼容服务。"""
    s = get_settings()
    return OpenAI(
        api_key=s.llm_key(),
        base_url=s.llm_url(),
    )


def chat(user_message: str, history: list[dict] | None = None) -> str:
    """
    根据用户消息与可选的历史对话生成回复。

    - 先对 user_message 做 RAG 检索，得到相关记忆
    - 将记忆与历史一起交给 LLM，生成一条回复

    :param user_message: 用户当前输入
    :param history: 可选，格式 [{"role":"user"/"assistant","content":"..."}, ...]
    :return: 助手回复文本
    """
    s = get_settings()
    client = get_llm_client()

    # RAG：检索与当前问题相关的记忆
    rag_context = build_rag_context(user_message, top_k=5)

    # 若没有历史，则只发当前轮；若有历史，则按 OpenAI 格式组装
    messages = [{"role": "system", "content": AMADEUS_SYSTEM_PROMPT}]

    if history:
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})

    # 将 RAG 上下文与用户消息一起发给模型（可放在最后一条 user 中）
    user_content = user_message
    if rag_context:
        user_content = f"{rag_context}\n\n【用户说】\n{user_message}"

    messages.append({"role": "user", "content": user_content})

    resp = client.chat.completions.create(
        model=s.llm_model,
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""
