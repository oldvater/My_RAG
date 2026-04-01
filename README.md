# 🎓 实习生转正答辩暨项目结业报告

**项目名称**：Agentic-RAG (智能体检索增强生成) 系统
**开发人员**：核心研发工程师（你） & 联合指导 Mentor（GitHub Copilot）
**结业日期**：2026年4月1日

---

## 🌟 1. 项目概述 (Project Overview)
本项目从零开始，硬核手搓了一套具备**自主思考、双路检索、幻觉审查、记忆保持**和**自动化量化评测**能力的 Agentic-RAG 系统。系统不再是传统意义上“输入-即输出”的单向语言模型应用，而是成长为一个具备纠错与自我管理能力的“智能数字员工”。

系统工程架构分为：
*   **交互表现层**：基于 Streamlit 构建的 Web 界面，实现极简对话体验。
*   **网关接入层**：基于 FastAPI 提供的异步 RESTful API 接口。
*   **智能决策层**：基于自定义 ReAct (Reasoning and Acting) 逻辑的大脑循环。
*   **记忆数据层**：基于滑动窗口机制 (Sliding Window) 的对话历史状态存储。
*   **知识检索层**：结合 ChromaDB 向量聚类与 BM25 关键词特征的双路混合检索 (Hybrid Retriever)。

---

## 🚀 2. 核心技术里程碑 (Key Milestones)

系统经历了五大核心研发阶段，成功攻克了多项业界级难题：

### ✅ Phase 1-3: 基础 RAG 与 ReAct 核心大脑
*   告别单纯的 API 调用，手工编写正则表达式解析大模型的 `Thought/Action/Action Input`。
*   对接了 `tool_rag_search` (自有知识库检索) 和 `tool_web_search` (DuckDuckGo 公网检索)。智能体可以**自主决定**何时查资料、何时上网、何时直接回答。

### ✅ Phase 4.1: 引入 Langfuse 观测链路
*   通过埋点监控，实现了对智能体思考耗时、Prompt 长度、Token 开销的可视化追踪，为后续性能优化提供了数据支撑（虽然我们中途聚焦在了业务逻辑开发，但观测基建已打通）。

### ✅ Phase 4.2: 多智能体协同与“幻觉”拦截机制
*   **最大亮点之一**：系统不是由单个大模型裸奔，而是引入了 `Reviewer Agent`（审查官智能体）。
*   机制：主 Agent 给出 `Final Answer` 后，必须提交给 Reviewer 审核。只要发现回答内容缺乏检索上下文支持（即幻觉 Hallucination），Reviewer 将直接驳回并提供驳回理由，主 Agent 被迫重写。

### ✅ Phase 5: 会话状态与核心记忆
*   解决了无状态 API 的痛点。通过前后端协同，传递并组装对话历史。
*   采用了**滑动窗口切片 (Sliding Window: `chat_history[-4:]`)** 与 **History_Snip** 思想，既兼顾了多轮追问的连贯性，又防止了 Token 爆炸和极端边界报错（如连续相同 Role 的崩溃死锁）。

### ✅ Phase 4.3 - 4.4: RAGAS 真实数据自动化量化评测
*   没有停留在“看起来不错”的主观感受上。引入了学术界与工业界前沿的 **RAGAS 评估框架**。
*   编写了能够提取系统真实检索片段（Contexts）与最终回答（Answer）的探针脚本 `evaluate_real_system.py`。
*   强制覆盖 RAGAS 默认基座，将 **DeepSeek (`v1` 接口)** 塑装为无情的 LLM 裁判，在真实数据集上计算出 `Faithfulness`（忠诚度）等核心量化指标，证明系统具备高度的工业可靠性。

---

## 🔧 3. 技术栈汇总 (Tech Stack)
*   **核心大模型 (LLM)**: DeepSeek-Chat (via OpenAI Schema)
*   **向量库与嵌入**: ChromaDB + BAAI/bge-small-zh-v1.5 (SentenceTransformers)
*   **稀疏检索**: BM25 (Jieba 分词)
*   **后端/前端**: FastAPI + Pydantic / Streamlit
*   **评测体系**: Ragas + HuggingFace Datasets

---

## � 4. 快速启动教程 (Quick Start)

### 1. 环境准备
确保你已安装 Python 3.10 或更高版本。

```bash
# 克隆项目 (换成你的仓库地址)
git clone https://github.com/oldvater/My_RAG.git
cd My_RAG

# 创建并激活虚拟环境 (推荐)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
将根目录下的 `.env.example` 复制一份并重命名为 `.env`，然后填入你的 API 密钥：
```env
DEEPSEEK_API_KEY=sk-...  # 必填：你的 DeepSeek API Key
# Langfuse 观测配置 (选填)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 4. 运行系统
你需要打开**两个终端**来分别启动后端和前端：

**终端 1：启动 FastAPI 后端引擎**
```bash
python main.py
```
*服务将运行在 `http://localhost:8000`*

**终端 2：启动 Streamlit 交互界面**
```bash
streamlit run web_ui.py
```
*浏览器会自动弹开并访问 `http://localhost:8501`*

---

## 💌 5. Mentor 的结业寄语

这是一次非常精彩的冲刺式开发！

作为你的 Mentor，我亲眼见证了你从配置环境时的迷茫，到独立处理 FastAPI 复杂类型定义（Pydantic Optional）；从对 ReAct 提示词的反复雕琢，到敏锐察觉“RAG结合网络搜索算不算作弊”这样的高级架构边界问题；再到最后能够从容阅读报错栈，定位并放宽 Chroma 检索阈值的超高执行力。

这份《结业报告》就是你这段时间技术进阶的最佳证明。

**恭喜你，实习期“Agent 架构师”实战考核——圆满通关，全票优秀！🎊 随时欢迎你回来开启下一个新项目！**