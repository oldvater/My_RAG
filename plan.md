## Plan: 企业级 RAG 与多智能体系统 (Enterprise-grade RAG & Multi-Agent System)

这是一个专为“大模型应用开发实习岗”量身定制的核心项目，结合您之前的工程部署背景，这套方案重点展示您在系统底层架构、检索增强生成（RAG）优化以及复杂智能体任务编排上的能力。

**🎯 当前项目进度 (Current Status)**
- ✅ **Phase 1 (基础建设)**: FastAPI 底座、BGE 中文模型接入、ChromaDB 向量库基础集成完毕。
- 🐛 **发现问题**: 朴素的“固定字数滑动窗口”切片会导致不同主题（如 BERT 与《百年孤独》）发生跨界语义污染（缝合怪现象）。
- 🚀 **下一步走向**: 优化切片策略并闭环大模型生成链路。

**📋 迭代实施路径 (Steps)**

**Phase 1: 基础设施构建 (Foundation & Basic RAG)**
- [x] 1. 初始化 FastAPI 后端服务架构（路由、配置、CORS 跨域）。
- [x] 2. 接入本地算力部署 BGE 中文 Embedding 模型与 ChromaDB 向量数据库。
- [x] 3. 构建纯向量搜索的 `/rag/query` 检索接口。
- [x] 4. **进阶文档解析与清洗 (Structural Chunking)**：放弃纯滑动窗口，改用基于正则表达式（如按段落 `\n\n` 或标点）的结构化切片，附带 Metadata隔离，彻底修复“语义缝合怪”Bug。

**Phase 1.5: RAG 生成闭环 (Generation Closed Loop) [新增]**
- [x] 5. 接入大语言模型 (LLM)，构建 Prompt 模板。将检索出的 Top-K `text` 喂给 LLM，实现基于上下文的流式回答生成。完成真正的“检索 + 增强 + 生成”完整链路。
- [x] 5.5. 端到端的 PDF 文件解析与入库链路。

**Phase 2: RAG 高阶召回优化 (Advanced Retrieval Optimization)**
- [x] 6. *depends on 4* 实现 **混合检索 (Hybrid Search)**：结合 BM25（稀疏检索）与 Dense Vector（稠密检索）。
- [x] 7. *depends on 6* 引入 **重排模型 (Reranker)** (如 BGE-Reranker)，提升 Top-K 检索的准确率，避免召回不相关的片段。
- [x] 8. 架构升级：传统的 Query Rewrite（查询重写）已被大语言模型在 ReAct 循环中的 `Action Input` 原生生成能力所取代（Agent会自主推断并重写搜索词）。

**Phase 3: 多智能体中枢构建 (Agentic Core)**
- [x] 9. *depends on 5 & 8* 不依赖高度聚合框架底座，手写基于 ReAct (Reason+Act) 架构的自主智能体处理循环。
- [x] 10. 为 Agent 注册工具（Tools）：包括 RAG 检索器工具、DuckDuckGo 网络搜索工具。
- [x] 11. *平行开发* 引入多 Agent 协同（可通过 LangGraph 或 AutoGen 概念简化实现），区分 “Research Agent (查询资料)” 与 “Reviewer Agent (内容校验)”。

**Phase 4: 评测、监控与部署 (Evaluation & Observability)**
- [x] 12. 引入客观评测框架（如 RAGAS），对系统的 Context Precision (检索精度) 和 Answer Relevance (回答相关性) 进行客观打分。这种用“模型评测模型”的能力是业务团队极度渴望的技能。
- [x] 13. 接入 LLM 观测（Langfuse 集成）：通过 `langfuse.openai` 实现无侵入式的可观测性埋点（Zero-code instrumentation），实时监控完整调用链 Trace、Token 消耗及 API 延迟。

**Phase 5: 交互体验与多轮增强 (Experience & Memory) [新增]**
- [x] 14. 引入 Session Memory (短期记忆)：改造当前的系统，使其能够接收并携带历史聊天记录，彻底解锁多轮并发问答能力，替代传统的 Query Rewrite 方案。

**Relevant files**
- [app/main.py](app/main.py) — FastAPI 服务入口和中间件
- [app/rag/retriever.py](app/rag/retriever.py) — 混合检索、重排与 Query Rewrite 逻辑
- [app/agents/react_agent.py](app/agents/react_agent.py) — Agent 路由架构编排
- [app/core/config.py](app/core/config.py) — 基于 Pydantic 的全局环境与配置管理

**Verification**
1. 构建 50 条测试集（包含简单事实、复杂逻辑），运行 RAGAS 评估，最终出具一份项目指标分析报告。
2. 启动服务并测试多轮并发对话流式输出，确保不断连或阻塞。

**Decisions**
- **技术栈深度倾向**: 尽量使用底层库手写，以此展现深入理解框架思想、能够排查疑难杂症的能力。
- **范围节制**: 不做重资产 Vue/React 前端，直接采用 Streamlit 提供带聊天界面的交互端。

**Further Considerations**
1. **本地模型结合**: 考虑到您的部署经验，我们是否考虑用 vLLM 或 Ollama 在本地部署基座模型（如 Llama3/Qwen），把这点打造为简历亮点？
2. **业务数据场景**: 知识文件准备聚焦哪个领域？（如：财务法律文档 / AI顶会PDF合集 / 内部代码库）建议找格式挑战大的文件类型。