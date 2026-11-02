"""临时测试脚本：直接测试 RAG 问答效果（不需要交互输入）"""
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


def get_embedding(text):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def ask(question):
    chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
    collection = chroma_client.get_or_create_collection(name="campus_docs")

    q_embedding = get_embedding(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=TOP_K)
    docs = results["documents"][0]
    context = "\n\n".join(docs)

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
    return resp.choices[0].message.content


if __name__ == "__main__":
    questions = [
        "学生考试作弊会受到什么处分？",
        "学生请假需要办理什么手续？",
        "学生可以申请休学吗？",
    ]
    for q in questions:
        print(f"\n问题：{q}")
        print(f"回答：{ask(q)}")
