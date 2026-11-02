"""构建知识库向量索引：读取 data/ 文档 -> 切分 -> 向量化 -> 存入 ChromaDB"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

EMBEDDING_MODEL = "embedding-3"
CHUNK_SIZE = 500
OVERLAP = 50


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """把长文本切成带重叠的小块"""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def get_embedding(text: str):
    """调用 embedding 模型，把一段文字转成向量"""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def main():
    data_dir = BASE_DIR / "data"
    files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md"))
    if not files:
        print("data 目录下没有文档，请先放入 .txt 或 .md 文件")
        return

    chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
    collection = chroma_client.get_or_create_collection(name="campus_docs")

    total = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = split_text(text)
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            embedding = get_embedding(chunk)
            collection.upsert(
                ids=[f"{f.stem}_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": f.name}],
            )
            total += 1
        print(f"已索引《{f.name}》：{len(chunks)} 块")

    print(f"\n索引构建完成，共 {total} 块，向量库存在 ./chroma_db")


if __name__ == "__main__":
    main()
