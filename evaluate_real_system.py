# evaluate_real_system.py

import os
from app.core.config import settings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy # 你还可以引入 answer_relevancy，但这里我们先测 faithfulness
from langchain_openai import ChatOpenAI # 这里引入大模型包装器
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from ragas.llms import llm_factory
from main import run_react_agent, vector_store  # 从您的系统主入口直接导入 Agent 和真实的向量库实例
from ragas.embeddings import embedding_factory # 以后测相关度需要包 Embedding 时用它

os.environ["LANGFUSE_SDK_ENABLED"] = "false" # <--- 加这行临时禁用 Langfuse

deepseek_judge = ChatOpenAI(
    api_key=settings.DEEPSEEK_API_KEY, 
    base_url="https://api.deepseek.com", # DeepSeek的base_url，不需要加/v1，或者加/v1试下
    model="deepseek-chat",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)
faithfulness.llm = deepseek_judge
embeddings_wrapper = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'}, # 如果有GPU，可以写 'cuda'
    encode_kwargs={'normalize_embeddings': True}
)
answer_relevancy.llm = deepseek_judge
answer_relevancy.embeddings = embeddings_wrapper
# myembeddings = embedding_factory('openai', model='text-embedding-3-small', client=deepseek_client)
# ====================================================================
# 💡 请在这里填写你想对真实数据库发起的 "灵魂拷问" 测试题！
# 尽量选一些你在存入向量库的PDF/TXT中提到过的具体知识点。
# ====================================================================
test_questions = [
    # ========================================================
    # 类别一：基础细节与硬核事实 (考察 Retriever 的底层精度和细粒度 Chunking)
    # 这类问题要求系统必须在满篇的学术名词里精准定位到特定算法的概念。
    # ========================================================
    "DROP论文中提到了实例集合缩减（instance set reduction）的三种搜索方向（Direction of search），分别是哪三种？",
    "DROP论文中提到了SNN (Selective Nearest Neighbor Rule) 算法的时间复杂度大约是多少？",
    "DROP论文中提到了在处理名义属性（nominal attributes）时，VDM (Value Difference Metric) 是如何计算距离的？",
    "DROP论文中提到了HVDM (Heterogeneous Value Difference Metric) 距离函数是如何处理输入值缺失（unknown）的情况的？",
    "DROP论文中提到了EACH 算法 (NGE 理论) 中，如果新实例与最近样例的分类相同，系统会进行什么操作？",
    "DROP论文中提到了Tomek 在 1976 年提出的 All k-NN 算法是如何决定哪些实例被标记为 bad 并删除的？",
    "DROP论文中提到了TIBL 算法中是如何定义一个实例的“典型性 (typicality)”的？",
    "DROP论文中提到了Skalak 在 1994 年使用了什么爬山算法（hill climbing）策略来选择保留的实例？",

    # ========================================================
    # 类别二：跨段落总结与对比推理 (考察重排模型的 Top-K 召回整合能力，及 Agent 的逻辑推理)
    # 这类问题必须让 Agent 拼凑两个距离较远的文档块，不能只看单一段落。
    # ========================================================
    "DROP论文中提到了Incremental（增量）和 Decremental（减量）两种搜索策略各自的主要优点和缺点是什么？",
    "对比 Hart 提出的 CNN 算法和 Aha 的 IB2 算法，它们在处理实例和对噪声的敏感度上有什么异同？",
    "为什么DROP论文作者认为在执行实例删除时，Batch (批处理) 模式可能会导致整个聚类簇（clusters）意外消失？",
    "DROP论文中提到了在决定保留哪些实例时，保留边界点（Border points）和保留中心点（Central points）在直觉上和对决策边界的影响上有什么不同？",
    "DROP论文中提到了IB3 算法是如何解决 IB2 算法保留了过多噪声实例的大问题的？请简述其核心机制。",
    "DROP论文中提到了RISE 系统处理实例的表达方式跟普通的最近邻模型（nearest neighbor）有什么最核心的区别？",
    "DROP1 算法决定是否安全移除一个实例 P 的核心判定规则是什么？",
    "DROP论文中提到了传统的 KNN 算法即使应用了 k-d trees 和 projection 技术，依然没有解决哪两个核心痛点？",
    "DROP论文中提到了Encoding Length Grow (ELGROW) 方法在评估一个分类器的成本 (Cost) 时，其公式 F(m,n) 包含了哪些因素？",

    # ========================================================
    # 类别三：高难度迷惑性问题 (考察 Faithfulness 忠实度 / 抗幻觉能力)
    # 这些问题捏造了原本不存在的技术或超越年代的概念。
    # 如果 RAG 模型回答了具体的细节，说明产生了严重“幻觉”，Faithfulness 将直接得 0 分。
    # ========================================================
    "DROP论文提到的 DROP6 和 DROP7 算法，相比前面的算法在距离度量层面做了哪些额外改进？", # 迷惑点：文档里只有 DROP1 到 DROP5 和 DEL。
    "Wilson 和 Martinez 在DROP论文的实验中，验证这 31 个分类任务时是否使用了 ImageNet 或类似的大型图像数据集？", # 迷惑点：文档在 2000 年发布，文中虽提及 31 个任务，但未提现代的 ImageNet，诱导模型瞎编。
    "在DROP论文的 Survey 部分讲解 Query 重写时，作者是如何评价 Transformer 模型的大语言模型在里面的潜力的？", # 迷惑点：这篇论文只讲了神经网络、SVM、K-NN，由于时代局限根本不可能提到 Transformer 这种 2017 年后的产物。
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
result = evaluate(dataset, 
                  metrics=[faithfulness, answer_relevancy])

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
