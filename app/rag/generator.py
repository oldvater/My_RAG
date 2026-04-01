# app/rag/generator.py
from openai import OpenAI
from app.core.config import settings

# 初始化一个“指向 DeepSeek 服务器”的 OpenAI 客户端
client = OpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

def generate_rag_response(query: str, retrieved_contexts: list[dict]) -> str:
    """
    接收用户的查询 和 向量库找回来的内容，打包发给大模型。
    """
    # 1. 把刚才检索到的 3 个 chunk 的纯文本提取出来，拼成一大段参考资料
    context_text = ""
    for item in retrieved_contexts:
        context_text += f"- {item['text']}\n"
        
    # 2. 核心：构建 RAG 专属系统指令 (Prompt)
    system_prompt = f"""
    你是一个严谨的智能知识库助手。请根据我提供的【参考资料】来回答用户的提问。
    
    【约束条件】
    1. 只能根据【参考资料】中的信息回答，千万不要自己瞎编。
    2. 如果【参考资料】中有些内容与问题无关（比如小说、毫不相干的故事），请直接忽略它。
    3. 如果【参考资料】里完全没有能回答该问题的信息，请直接回答“抱歉，资料库中没有找到相关答案”。
    
    【参考资料】
    {context_text}
    """
    
    # 3. 调用大语言模型（请你补齐这里的代码）
    response = client.chat.completions.create(
        model="deepseek-chat", # DeepSeek 的通用对话模型
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0 # RAG 场景通常把温度设低，防止它乱发散
    )
    
    # 4. 把大模型说的话 return 出去
    return response.choices[0].message.content