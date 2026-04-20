# evaluate_ragas.py

# import os
# from app.core.config import settings
# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import faithfulness # 注意这里不要带下划线，直接引 faithfulness
# from langchain_openai import ChatOpenAI # 这里引入大模型包装器

# # 1. 直接配置 Langchain 的 OpenAI 包装器，明确指向 DeepSeek
# # 这次我们不用环境变量骗了，直接硬编码传给它，并强制使用 json 格式响应
# deepseek_judge = ChatOpenAI(
#     api_key=settings.DEEPSEEK_API_KEY, 
#     base_url="https://api.deepseek.com", # DeepSeek的base_url，不需要加/v1，或者加/v1试下
#     model="deepseek-chat",
#     temperature=0,
#     model_kwargs={"response_format": {"type": "json_object"}}
# )

# # 2. 我们必须告诉 RAGAS 这位特殊的裁判：你要用我刚刚配好的 deepseek_judge ！
# # faithfulness 内部也是一个大模型，我们需要把它的模型配一下
# faithfulness.llm = deepseek_judge

# data_samples = {
#     "question": [
#         "LightRAG 的核心设计目标是什么？", 
#         "LightRAG 是哪个公司在几年开发的？"
#     ],
#     "answer": [
#         "LightRAG 的核心设计目标是实现“简单”与“快速”，主要为了解决现有复杂RAG系统（如GraphRAG）存在的效率瓶颈和计算成本高昂的问题。", 
#         "根据资料，LightRAG 是由 OpenAI 公司在 2024 年投入巨资研发的闭源项目。" # ⚠️注意：这是我故意写的幻觉答案，用来测试裁判的！
#     ],
#     "contexts": [
#         ["LightRAG 是一种新型的检索增强生成（RAG）框架，其设计目标是实现“简单”与“快速”，旨在解决现有一些复杂RAG系统（如GraphRAG）存在的效率瓶颈和高昂成本问题。"],
#         ["它指的是香港大学数据科学实验室（HKUDS）在GitHub上开源的一个项目，其核心是一篇被EMNLP 2025会议收录的论文。"]
#     ],
#     "ground_truth": [
#         "实现“简单”与“快速”，解决复杂RAG系统的效率和成本问题。", 
#         "是由香港大学数据科学实验室（HKUDS）开源的项目，相关论文被EMNLP 2025会议收录。"
#     ]
# }

# dataset = Dataset.from_dict(data_samples)

# # 拼装提示：调用 evaluate(你的数据集, metrics=[你导入的评测维度列表])
# result = evaluate(dataset, metrics=[faithfulness])
# print(result)

from main import vector_store

# 直接手测你的核心检索器
question = "DROP论文中提到了Incremental（增量）和 Decremental（减量）两种搜索策略各自的主要优点和缺点是什么？"
docs = vector_store.search(query=question, top_k=5)

print(f"一共召回 {len(docs)} 个片段：")
for i, d in enumerate(docs):
    print(f"\n--- Chunk {i+1} ---")
    print(d['text'])