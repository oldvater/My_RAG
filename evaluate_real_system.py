# evaluate_real_system.py

import os
from app.core.config import settings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness # 你还可以引入 answer_relevancy，但这里我们先测 faithfulness
from langchain_openai import ChatOpenAI
from main import run_react_agent, vector_store  # 从您的系统主入口直接导入 Agent 和真实的向量库实例

deepseek_judge = ChatOpenAI(
    api_key=settings.DEEPSEEK_API_KEY, 
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)
faithfulness.llm = deepseek_judge

# ====================================================================
# 💡 请在这里填写你想对真实数据库发起的 "灵魂拷问" 测试题！
# 尽量选一些你在存入向量库的PDF/TXT中提到过的具体知识点。
# ====================================================================
test_questions = [
    "SBERT讲了什么？",
    "介绍一下DROP这篇论文里提到的核心架构是什么？",
    "有没有提到具体的评估指标，表现如何？"
]

data_samples = {
    "question": [],
    "answer": [],
    "contexts": []
}

print("🚀 开始请求您的 Agent-RAG 系统（调用真实 ChromaDB 数据库）...\n")

for i, q in enumerate(test_questions):
    print(f"[{i+1}/{len(test_questions)}] 👉 正在向 Agent 提问: {q}")
    
    # 🌟 调用你亲手写的核心流水线：携带历史上下文（给空避免串门）、经过智能体路由和 HybridRetriever 混合检索
    raw_answer, used_contexts = run_react_agent(q, vector_store, chat_history=[])
    
    # ⚠️ Agent 返回的用过的上下文是一个字典列表(含分数等)，我们需要把它提取成纯文本列表给 RAGAS 评委
    # 处理结构: used_contexts -> [{'text': '段落1', ...}, {'text': '段落2', ...}]
    str_contexts = [doc.get('text', str(doc)) if isinstance(doc, dict) else str(doc) for doc in used_contexts]
    
    # 防止完全没搜到东西报错，放入空字符串兜底
    if not str_contexts:
        str_contexts = ["未找到任何参考上下文"]

    data_samples["question"].append(q)
    data_samples["answer"].append(raw_answer)
    data_samples["contexts"].append(str_contexts)
    
    print(f"   🤖 Agent 的最终回答: {raw_answer[:60]}...")
    print(f"   📚 实际引用的上下文块数量: {len(str_contexts)}\n")

print("\n⚖️ 数据采集完毕，下面移交 DeepSeek 裁判（RAGAS）评估生成的忠实度（Faithfulness）...")

# 构造 HuggingFace Dataset 供 RAGAS 评测
dataset = Dataset.from_dict(data_samples)

# 启动魔法打分！
result = evaluate(dataset, metrics=[faithfulness])

print("\n==================================")
print("📊 真实数据库系统评测最终成绩单:")
print(result)
print("==================================\n")

for i in range(len(test_questions)):
    print(f"📝 样本 {i+1}:")
    print(f"  问题: {data_samples['question'][i]}")
    print(f"  💡 回答提取: {data_samples['answer'][i][:50]}...")
    print(f"  📚 引用文献数: {len(data_samples['contexts'][i])}")
    print("-" * 50)
