"""
RAG（检索增强生成）模块：Amadeus 的「记忆」层。

- 使用 Chroma 存储文档的向量嵌入
- 通过 OpenAI Embeddings 将文本转为向量
- 用户提问时检索最相关的记忆片段，供 LLM 作为上下文
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from .config import get_settings


# 默认的 Amadeus 人格/知识片段，可视为「初始记忆」
DEFAULT_MEMORIES = [
    "我是 Amadeus，一个基于记忆与人格数据的 AI 助手。名字来自作曲家莫扎特（Wolfgang Amadeus Mozart），意为「被神所爱」。",
    "我可以进行自然对话，并利用已存储的记忆来更好地理解与回应。",
    "如果你告诉我关于你或某件事的信息，我可以在对话中记住并引用。",
    "我支持文字对话与语音输入；回复可以以文字或语音形式返回。",
]


def get_embedding_client() -> OpenAI:
    """获取用于生成嵌入的客户端，可独立于 LLM 使用不同服务。"""
    s = get_settings()
    return OpenAI(
        api_key=s.embedding_key(),
        base_url=s.embedding_url(),
    )


def get_embedding(text: str) -> list[float]:
    """将单段文本转为 embedding 向量。"""
    client = get_embedding_client()
    s = get_settings()
    resp = client.embeddings.create(
        model=s.embedding_model,
        input=text,
    )
    return resp.data[0].embedding


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """批量将文本转为 embedding，减少 API 调用次数。"""
    if not texts:
        return []
    client = get_embedding_client()
    s = get_settings()
    resp = client.embeddings.create(
        model=s.embedding_model,
        input=texts,
    )
    # 保持与 input 顺序一致
    ordered = {e.index: e.embedding for e in resp.data}
    return [ordered[i] for i in range(len(texts))]


def get_chroma_client():
    """获取或创建 Chroma 持久化客户端。"""
    s = get_settings()
    path = Path(s.chroma_persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


# 集合名称，可理解为「Amadeus 的记忆库」
COLLECTION_NAME = "amadeus_memory"


def get_collection():
    """获取 Chroma 中的 amadeus_memory 集合；不存在则创建并写入默认记忆。"""
    client = get_chroma_client()
    try:
        coll = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        coll = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Amadeus 记忆与知识库"},
        )
        # 写入默认记忆
        ids = [f"default_{i}" for i in range(len(DEFAULT_MEMORIES))]
        coll.add(
            ids=ids,
            embeddings=get_embeddings(DEFAULT_MEMORIES),
            documents=DEFAULT_MEMORIES,
        )
    return coll


def add_memory(text: str, metadata: dict | None = None) -> str:
    """
    添加一条记忆到 RAG 库。
    返回新记忆的 id。
    """
    coll = get_collection()
    emb = get_embedding(text)
    new_id = f"mem_{len(coll.get()['ids'])}"
    coll.add(ids=[new_id], embeddings=[emb], documents=[text], metadatas=[metadata or {}])
    return new_id


def search_memory(query: str, top_k: int = 5) -> list[str]:
    """
    根据用户问题检索最相关的记忆片段。
    返回 top_k 条文档文本列表。
    """
    coll = get_collection()
    query_emb = get_embedding(query)
    results = coll.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, coll.count()),
    )
    if not results or not results["documents"]:
        return []
    return results["documents"][0] or []


def build_rag_context(query: str, top_k: int = 5) -> str:
    """
    根据 query 检索记忆并格式化为给 LLM 的上下文字符串。
    若无相关记忆则返回空字符串。
    """
    docs = search_memory(query, top_k=top_k)
    if not docs:
        return ""
    return "【相关记忆】\n" + "\n".join(f"- {d}" for d in docs)
