# app/agents/reviewer_agent.py
import os
import json
from typing import Tuple
from langfuse.openai import OpenAI
from app.core.config import settings

# 将 settings 里的值塞给系统环境变量，供 Langfuse 底层抓取
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST

client = OpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

REVIEWER_PROMPT = """
你现在是一个严苛的【AI审查官】。
你的任务是审查另一个AI模型给出的最终答案(Draft Answer)。

请你【只根据提供给你的参考资料】以及【过往的历史对话记录】进行审查，绝对不要依赖你自己的历史知识（你的知识可能落后）！

你需要仔细检查两个方面：
1. 幻觉检查：这个答案是不是在毫无根据地瞎编烂造？它所陈述的关键事实，是否都能在【参考资料】或【历史对话记录】中找到支撑？（提示：如果草稿答案是在顺着历史对话的语境继续聊天，或者总结已有信息，这也是完全合法的，不算幻觉。如果用户要求发挥想象力，只要它声明了是想象，也不算幻觉）
2. 完整性：这个答案有没有回答用户最初的提问？

请严格输出以下JSON格式（不需要任何其他废话）：
{
    "status": "PASS" 或者 "REJECT",
    "reason": "如果你给了 REJECT，请说明要求它重写的理由；如果是 PASS 则写无"
}
"""

def review_answer(user_query: str, draft_answer: str, context_data: list, chat_history: list = None) -> Tuple[bool, str]:
    """
    审查生成的回复。
    返回一个元组：(是否通过 Boolean, 拒回的理由 String)
    """
    # 你的任务：
    context_str = json.dumps(context_data, ensure_ascii=False)
    chat_history_str = json.dumps(chat_history, ensure_ascii=False) if chat_history else "无"
    # 1. 组装 messages，将上面的 REVIEWER_PROMPT 作为 system，将用户的问题和 react_agent 生成的 draft_answer 给到 user。
    messages = [
        {"role": "system", "content":REVIEWER_PROMPT},
        {"role": "user", "content": f"【参考资料】:\n{context_str}\n\n【历史对话记录】:{chat_history_str}\n\n【用户提问】:{user_query}\n\n【待审查答案】:{draft_answer}"} 
    ]
    
    # 2. 发起 client.chat.completions.create 请求，记得加 response_format={ "type": "json_object" } 让它强制输出 JSON。
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,#type:ignore
        temperature=0,
        response_format={"type": "json_object"}
    )
    response_text = response.choices[0].message.content
    # 3. 将返回结果解析成字典（用 json.loads）。
    if not response_text:
        return (False, "大模型返回了空内容。")
    parsed_dict = json.loads(response_text)
    # 4. 根据解析出的 "status" 字段，决定 return (True, "") 还是 return (False, "理由...")。
    if parsed_dict["status"] == "PASS":
        return (True, "审查通过，运行执行。")
    else:
        rea = parsed_dict["reason"]
        return (False, f"{rea}")