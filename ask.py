"""知识库问答：问题 -> 向量检索 -> 拼接上下文 -> 大模型生成回答"""
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
CHAT_MODEL = "glm-4-flash"
TOP_K = 3


def get_embedding(text: str):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def main():
    chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
    collection = chroma_client.get_or_create_collection(name="campus_docs")

    print("校园知识问答助手（输入 exit 退出）\n")
    while True:
        question = input("请输入问题：").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        # 1. 问题向量化
        q_embedding = get_embedding(question)
        # 2. 检索最相关的 TOP_K 个片段
        results = collection.query(query_embeddings=[q_embedding], n_results=TOP_K)
        docs = results["documents"][0]
        context = "\n\n".join(docs)

        # 3. 拼接提示词，让模型严格基于资料回答
        prompt = (
            "你是一个校园知识问答助手。请严格根据下面的资料回答用户问题，"
            "如果资料中没有答案，就回答「资料中没有相关信息」。\n\n"
            f"【资料】\n{context}\n\n"
            f"【用户问题】{question}\n\n"
            "【回答】"
        )

        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        answer = resp.choices[0].message.content
        print(f"\n回答：{answer}\n")


if __name__ == "__main__":
    main()
