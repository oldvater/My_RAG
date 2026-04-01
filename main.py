from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "project": settings.PROJECT_NAME}

# app/main.py (在之前的基础上追加)

from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid

# 导入你刚刚写的两个神仙工具！
from app.rag.parser import structural_chunker
from app.rag.vector_store import VectorStoreBase
from app.rag.generator import generate_rag_response
from app.rag.retriever import HybridRetriever
from app.agents.react_agent import run_react_agent

# 初始化你的 Vector Store
vector_store = HybridRetriever()

class DocumentInput(BaseModel):
    text: str

class QueryInput(BaseModel):
    query: str

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = []

@app.post("/rag/upload")
def upload_document(doc: DocumentInput):
    """
    第一步：将被上传的整篇长文档，分块，并存入向量数据库
    """
    # 1. 使用 structural_chunker 结构化切片替代无脑的滑动窗口
    chunks = structural_chunker(doc.text)
    # 【Bug风险点提示】UUID 可以防止下一次传文档时前面的 chunk_0 被覆盖
    ids = [f"chunk_{uuid.uuid4().hex[:8]}" for _ in range(len(chunks))]
    # 3. 使用 vector_store.add_texts(chunks, ids) 存入数据库
    if chunks:  # 防止传入空文档报错
        vector_store.add_documents(chunks, ids)
    # 4. return 一下成功录入的块数
    return {"status": "success", "chunks_inserted": len(chunks)}

@app.post("/rag/query")
def query_database(q: QueryRequest):
    """
    第二步：让大模型（或者人）知道，系统里有没有这些相关知识
    """
    # 【Bug拦截】：pydantic 的 BaseModel 直接有一个 `.query` 属性存放字符串，不需要用 str(q) 强转
    query_text = q.query
    print("收到的历史记录：", q.chat_history)
    # result = vector_store.search(query_text, top_k=3)

  # ---------- 你需要在这里补齐大模型的调用！ -----------
    # 提示 1：你得引入刚才写的 generate_rag_response 函数
    # 提示 2：你得把用户的 query_text 和检索出来的 result 传给它
    # 提示 3：拿到答案后，把它一起包装进 return 的字典里返回
    # answer = generate_rag_response(query_text, result)
    answer, context = run_react_agent(query_text, vector_store, q.chat_history)
    # 2. 返回找到的内容结果
    return {"status": "success","answer":answer, "context": context}


from fastapi import File, UploadFile
import fitz

@app.post("/rag/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    bytes_data = await file.read()
    doc = fitz.open(stream=bytes_data, filetype="pdf")
    pages_text = []
    for page in doc:
        text = str(page.get_text(flags=fitz.TEXT_PRESERVE_WHITESPACE))
        pages_text.append(text)
    doc.close()

    full_text = "\n".join(pages_text)
    chunks = structural_chunker(full_text)
    ids = [f"chunk_{uuid.uuid4().hex[:8]}" for _ in range(len(chunks))]
    if chunks:
        vector_store.add_documents(chunks, ids)
    
    return {"status":"success", "chunks_inserted":len(chunks)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

