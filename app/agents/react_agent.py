# app/agents/react_agent.py


REACT_SYSTEM_PROMPT = """
你是一个极其智能的 AI 助手。为了回答用户的问题，你可以使用以下工具：

1. tool_rag_search: 当用户询问关于特定名词、文献、专业知识或系统自有知识库内容时，使用此工具。输入应为简短的搜索关键词。
2. tool_web_search: 当用户询问的内容在使用tool_rag_search仍未发现时，使用此工具。输入与tool_rag_search保持一致。

请严格按照以下格式进行思考和输出（不要随意打乱格式！）：

Question: 用户提出的问题
Thought: 你在思考该怎么做。你需要判断是否需要调用工具，以及应该用什么关键词去查。
Action: 你要使用的工具名称，必须是 [tool_rag_search, tool_web_search] 之一。
Action Input: 传给工具的输入参数。

（在这个阶段，你必须停止输出！等待外界系统将工具的运行结果返回给你。外界系统返回的格式永远是 Observation: 工具返回的结果）

Thought: 我现在拿到了观察结果。这些结果足够我回答问题了吗？
...（如果你觉得还需要查，你可以循环调用 Action 和 Action Input）...

一旦你认为你掌握了足够的信息，或者不需要任何工具就能回答：
Final Answer: 最终回答用户的答案内容。
"""


# 💡 你可能需要导入刚才写好的 retriever 并实例化
# from app.rag.retriever import HybridRetriever
# vector_store = HybridRetriever()
from ddgs import DDGS

def tool_rag_search(query: str, contexts: list, vector_store) -> str:
    """
    【给 Agent 使用的工具】
    作用：接收查询词，检索自有知识库，返回最相关的文本片段。同时将相关文本写入contexts
    """
    # 你的任务：
    # 1. 调用 vector_store.search(query) 获取前 3 名结果
    # 2. 从返回的列表中抠出每个元素的 "text"
    # 3. 把这 3 段 text 用换行符拼成一个毫无格式的大长字符串（因为大模型读纯文本最快）
    # 4. return 这个大长字符串
    pres = vector_store.search(query, top_k=3)
    result = ""
    valid_pre = []
    for pre in pres:
        if pre['rerank_scores'] >= 0:
            pre_text = pre['text']
            result += pre_text + '\n'
            valid_pre.append(pre)
    if len(result) < 1:
        return "自有知识库未检索到相关内容，请使用 tool_web_search 查寻公网。"
    contexts.extend(valid_pre)
    return result

def tool_web_search(query: str, contexts: list) -> str:
    """
    【给 Agent 使用的工具】
    作用：接收查询词，检索网络，返回搜索的结果。同时将相关文本写入contexts
    """
    ddgs = DDGS()
    texts = ""
    try:
        results = ddgs.text(query, max_results=3)
        for result in results:
            contexts.append({
                "text": result['body'],
                "metadata": {"source": result['href']}
            })
            texts = texts + result['title'] + ':' + result['body'] + '\n'
        return texts
    except Exception as e:
        print(f"网络搜索失败，原因:{e}")
        return "搜索失败或无网络结果，请尝试其他关键词查寻或根据已有知识作答。"



import os
import re
from langfuse.openai import OpenAI
from typing import List, Dict, Any

#引入统一配置
from app.core.config import settings
from app.agents.reviewer_agent import review_answer

# 将 settings 里的值塞给系统环境变量，供 Langfuse 底层抓取
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST

# 初始化 OpenAI 客户端 (DeepSeek 配置)
client = OpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

def run_react_agent(user_query: str, vector_store, chat_history: list = None) -> tuple:
    """运行 ReAct 主循环"""
    if chat_history is None:
        chat_history = []
    # --- 🚀 加入 Sliding Window 滑动窗口 ---
    # 我们只保留最近的 4 条消息（即最近的 2 轮多轮对话）
    # TODO: 用 Python 的负数切片语法更新 chat_history
    chat_history = chat_history[-4:]
    print(f"✂️ Agent内部确认，裁剪后的记忆条数：{len(chat_history)}")
    #创建空context，准备返回召回chunk
    used_contexts = []
 
    messages_sys: List[Dict[str, Any]] = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
    messages_user: List[Dict[str, Any]] = [{"role": "user", "content": f"Question: {user_query}"}]
    # 初始化我们给大模型看的“聊天记录”
    messages: List[Dict[str, Any]] = messages_sys + chat_history + messages_user
    
    # 设定一个最大循环次数，防止它陷入死循环变成“人工智障”
    max_iterations = 5 
    
    for i in range(max_iterations):
        print(f"\n--- [Agent 思考轮次 {i+1}] ---")
        
        # 1. 💡调用大模型生成回答（传入 messages，不要用流式，因为我们要拦截并解析它的文本）
        # 你的代码： response_text = ... (这里抄一下你在 generator.py 里写的非流式调用)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages = messages, #type: ignore
            temperature=0,
            stream=False
        )
        response_text = response.choices[0].message.content
        print(response_text) 
        
        # 将它的思考过程加入历史记录，这是最重要的！不然它会失忆
        messages.append({"role": "assistant", "content": response_text})
        
        # 2. 💡正则表达式检查：它是不是觉得够了，直接输出了 Final Answer？
        if "Final Answer:" in response_text:
            # 说明找到了答案！提取 Final Answer: 后面的内容并 return
            # (你可以用 split 或者正则提取)
            final_answer = response_text.split("Final Answer:")[-1].strip()

            #加入审查
             # --- 🚀 多智能体协同拦截点 ---
            print(f"👀 主Agent认为已找到答案，正在提交给 Reviewer Agent 审查...")

            is_pass, reason = review_answer(user_query, final_answer, used_contexts, chat_history)

            if is_pass:
                print("✅ Reviewer 审查通过！")
                return (final_answer, used_contexts)
            else:
                # 审查被拒！把拒绝理由当做 Observation 丢回给主 Agent，让它重新思考！
                print(f"❌ Reviewer 审查驳回！理由：{reason}")
                messages.append({"role":"user", 
                                "content": f"你的答案被审查员驳回（无需输出Action）。请根据以下理由重新思考并重写 Final Answer：\n{reason}"})
                continue
            
        # 3. 如果没结束，它一定是调用了工具。用正则表达式提取工具名和参数。
        # 这里我把正则送给你：
        action_match = re.search(r"Action: (.*?)\n", response_text)
        action_input_match = re.search(r"Action Input: (.*)", response_text)
        
        if action_match and action_input_match:
            action_name = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            print(f"🛠️ Agent 决定调用工具：{action_name}，参数：{action_input}")
            
            # 4. 💡执行工具
            observation = ""
            if action_name == "tool_rag_search":
                # 把上面写好的工具函数拿来用
                # 你的代码：observation = ...
                observation = tool_rag_search(action_input, used_contexts, vector_store)
            elif action_name == "tool_web_search":
                observation = tool_web_search(action_input, used_contexts)
            else:
                observation = "Error: 这个工具不存在！"
                
            # 5. 💡把拿到的结果拼成 Observation 告知大模型
            obs_message = f"Observation: {observation}"
            messages.append({"role": "user", "content": obs_message})
            print("📝 工具返回结果已录入上下文，进入下一轮思考...")
            
        else:
            # 如果大模型忘了格式，胡言乱语，提醒它
            messages.append({"role": "user", "content": "格式错误，请严格按照 Thought/Action/Action Input 或 Final Answer 的格式输出。"})
            
    return ("非常抱歉，我想了太久还是没找到答案。", [])