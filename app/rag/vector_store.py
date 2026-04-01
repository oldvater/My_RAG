# app/rag/vector_store.py

import chromadb
from sentence_transformers import SentenceTransformer

class VectorStoreBase:
    def __init__(self, collection_name: str = "rag_collection"):
        # 1. 实例化一个基于本地持久化文件存放的 chroma 客户端
        # 注意：chromadb支持PersistentClient, 存放在当前目录的 "./chroma_db" 文件夹中
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # 2. 从 client 中创建或获取对应的 collection
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # 3. 初始化通用的 Embedding 模型，推荐用 'all-MiniLM-L6-v2' (轻量极快)
        self.embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    def get_embedding(self, text: str) -> list[float]:
        """
        调用 embedding_model 把这段话转成对应维度的 float 列表。
        """
        # encode 返回的是 numpy 数组，转换成 python 的原生 list 便于 Chroma 接受
        return self.embedding_model.encode(text).tolist()

    def add_texts(self, texts: list[str], ids: list[str]):
        """
        把文本数组进行 embedding，然后存入 ChromaDB collection。
        """
        # 【性能优化】ML 模型天生适合“批处理 (Batch)”，直接把整个 texts 列表传进去，比 for 循环快得多！
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # ChromaDB 规定的插入语法是 .add() 并使用特定的参数名
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids
        )
        
    def search(self, query: str, top_k: int = 3):
        """
        根据 query 查询距离最近的 top_k 个文本段落
        """
        query_embedding = self.get_embedding(query)
        
        # 向量数据库的 query 自带一次性召回 top_k 个结果的功能，不需要 for 循环去查 top_k 次
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        # 获取返回的结果：id，文本内容，以及它们和查询词的『空间距离(distance)』
        ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results["distances"][0] # ChromaDB 默认会计算出的相似度距离

        formatted_results = []
        for i in range(len(ids)):
            # 引入阈值拦截：当距离分太离谱时（比如大于1.0），直接抛弃！宁少勿滥！
            if distances[i] > 1.0:
                print(f"⚠️ [拦截提示] Chunk '{ids[i]}' 距离({distances[i]:.2f})过高，已丢弃。")
                continue
                
            formatted_results.append({
                "id": ids[i],
                "distance": distances[i], # 距离分数也会一并返回
                "text": documents[i]
            })
        
        return formatted_results