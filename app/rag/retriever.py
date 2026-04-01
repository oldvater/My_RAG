# app/rag/retriever.py
import jieba
import pickle
import os
from rank_bm25 import BM25Okapi
from app.rag.vector_store import VectorStoreBase
from FlagEmbedding import FlagReranker

class HybridRetriever:
    def __init__(self):
        # 1. 继承咱们之前辛辛苦苦写的向量库
        self.vector_store = VectorStoreBase()

        self.bm25_cache_path = "bm25_cache.pkl"

        if os.path.exists(self.bm25_cache_path):
            with open(self.bm25_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.bm25_corpus = cache_data['corpus']
                self.raw_documents = cache_data['raw_docs']
                self.ids = cache_data['ids']
                self.bm25_model = cache_data['model']
            print("✅ 成功从本地加载 BM25 缓存库！包含数量:", len(self.raw_documents))
        else:
            # 没有缓存文件，说明是第一次运行，走原来的初始化流程
            self.bm25_corpus = []     # 存分词后的列表（给算法用） 例: [['我', '爱', '中国'], ['你', '好']]
            self.raw_documents = []   # 存原始字符串（为了最后返回给用户）
            self.ids = []             # 存对应的 ID
            self.bm25_model = None
            print("⚠️ 未发现 BM25 缓存，初始化为空库。")
        

        #初始化rerank模型
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

    def add_documents(self, chunks: list[str], ids: list[str]):
        """
        一份数据，两份存储！同时存入 向量库 和 BM25关键词库。
        """
        # 第一份：存入向量数据库 (Dense)
        self.vector_store.add_texts(chunks, ids)

        # 第二份：存入 BM25 关键词系统 (Sparse)
        for i, chunk in enumerate(chunks):
            # 💡 TODO 1: 使用 jieba.lcut() 对 chunk 也就是一段中文文本进行分词，得到一个字符串列表。
            tokenized_chunk = jieba.lcut(chunk)
            
            self.bm25_corpus.append(tokenized_chunk)
            self.raw_documents.append(chunk)
            self.ids.append(ids[i])
        
        # 💡 TODO 2: 将刚才准备好的 self.bm25_corpus 传入 BM25Okapi(...) 中进行初始化，赋给 self.bm25_model
        self.bm25_model = BM25Okapi(self.bm25_corpus)

        with open(self.bm25_cache_path, 'wb') as f:
            pickle.dump({
                'corpus': self.bm25_corpus,
                'raw_docs': self.raw_documents,
                'ids': self.ids,
                'model': self.bm25_model
            }, f)

        print("💾 BM25 数据已成功持久化保存到本地硬盘！")

    def search(self, query: str, top_k: int = 3):
        """
        双轨并行检索！分别拿到向量和字面匹配的结果。
        输出：{'text':...,'id':...,'rrf_score':...,'rerank_scores':...}
        """
        # ===== 1. 稠密检索 (Dense Retrieval - 靠深层语义) =====
        # 💡 TODO 3: 调用自己身上的 vector_store 去执行它原来的 search 方法
        vector_results = self.vector_store.search(query, top_k=10)

        # ===== 2. 稀疏检索 (Sparse Retrieval - 靠精确字面) =====
        if self.bm25_model is None:
            # 容错：要是没上传过数据，直接返回向量结果拉倒
            return vector_results 
        
        # 将用户查询也进行分词
        tokenized_query = jieba.lcut(query)
        # 获取 BM25 每一条语料在这个 query 下的得分
        bm25_scores = self.bm25_model.get_scores(tokenized_query)
        
        # 我们把分数、ID和原文本打包在一起，排个序
        bm25_combined = list(zip(self.ids, bm25_scores, self.raw_documents))
        # 💡 TODO 4: 把 bm25_combined 列表按照 分数（元素索引为 1） 从大到小降序排序，并切片只保留前 top_k 个
        bm25_combined_sorted = sorted(bm25_combined, key=lambda x: x[1], reverse= True)
        
        # # ===== 我们先不忙着合并，先把它打印出来，用肉眼观察两个算法找出来的东西有什么不同！ =====
        # print("\n=== 🧠 向量引擎 搜索结果 ===")
        # print(f"{vector_results}")
        
        # print("\n=== 🔍 传统 BM25 搜索结果 (ID, 得分, 前20字)===")
        # for item in bm25_combined_sorted:
        #     print(f"ID: {item[0]}, Score: {item[1]:.2f}, Text: {item[2][:20]}...")
        
        #实现RRF算法合并vector_results和bm25
        rrf_scores = self.rrf_score(vector_results, bm25_combined_sorted, top_k=10)
            
        #构造需要喂给reranker的数据对
        data_pair = []
        for doc in rrf_scores:
            text = doc["text"]
            data_pair.append([query, text])
        
        #调用ReRanker给这些对子算分
        rank_scores = self.reranker.compute_score(data_pair)

        # 因为偶尔 FlagReranker 在只有一个候选集时，可能直接返回一个 float，容错处理一下：
        if isinstance(rank_scores, float):
            rank_scores = [rank_scores]
        
        # 把精排的分数写回给原来的数据包裹
        for i, doc in enumerate(rrf_scores):
            doc['rerank_scores'] = rank_scores[i]

        # 按照这个新的 "rerank_score" 进行降序排序，并切片前 top_k (也就是 3)
        # 💡 TODO: 请你自己写这行最终的 sorted 逻辑！
        results = sorted(rrf_scores, key=lambda x: x['rerank_scores'], reverse=True)[:top_k]

        # print(f"最终传给大模型的片段数量：{len(results)}")
        return results
    
    def rrf_score(self, vector_results, bm25_combined_sorted, top_k):
        """
        RRF算法实现。
        """
        #用RRF合并稠密检索和稀疏检索的结果
        K = 60
        rrf_scores = {}
        doc_map = {}

        # 1. 处理向量检索结果 (假设是有序列表)
        for i, result in enumerate(vector_results):
            text = result["text"]
            rank_v = i + 1
            score_v = 1.0 / (K + rank_v)
            rrf_scores[text] = score_v
            doc_map[text] = {"text":text, "id":result["id"]}
        
        # 2. 处理 BM25 检索结果 (假设也是有序列表)
        for i, item in enumerate(bm25_combined_sorted):
            rank_s = i + 1
            score_s = 1.0/(K + rank_s)
            text = item[2]
            id = item[0]
            if text in rrf_scores:
                rrf_scores[text] = rrf_scores[text] + score_s
            else:
                rrf_scores[text] = score_s
                doc_map[text] = {"text":text, "id":id}
        
        # 3. 排序与截断
        result_final = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ans = []
        for tuple1 in result_final:
            text = tuple1[0]
            score = tuple1[1]

            doc_info = doc_map[text]
            doc_info['rrf_score'] = score
            ans.append(doc_info)
        return ans