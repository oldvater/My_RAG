# app/rag/parser.py

import re

def sliding_window_chunker(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    将长文本切分为带有重叠区域的多个文本块。
    
    参数:
    - text: 原始长文本
    - chunk_size: 每个文本块的最大字符数
    - overlap: 相邻文本块之间重叠的字符数，用于保证上下文连贯
    
    返回:
    - 一个包含多个字符串片段的列表
    """
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size，否则会导致死循环或切分异常。")
    
    ans = []
    start = 0
    while start < len(text):
        ans.append(text[start : start + chunk_size])
        start += (chunk_size - overlap)
    return ans

def structural_chunker(text: str, max_chunk_size: int = 500) -> list[str]:
    """
    基于正则表达式的结构化切片。
    核心思想：优先保证“段落”的完整性，防止不同领域的文本被生硬拼接到一起。
    """
    # 1. 采用正则按照一条或多条“换行符”切分成段落（丢弃空白段落）
    paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # 如果单个段落极长（比如没有排版的巨大TXT小说），触发兜底策略，将长段落再按滑动切片处理
        if len(para) > max_chunk_size:
            # 先把已经攒好的小段落存起来
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # 把超长的段落用之前的滑动窗口处理
            sub_chunks = sliding_window_chunker(para, max_chunk_size, overlap=50)
            chunks.extend(sub_chunks)
            continue
            
        # 尝试把短段落拼装在一起，提高上下文的连贯性
        # 如果 [当前累存的内容 + 新段落] 还没超标，就拼上
        if len(current_chunk) + len(para) + 1 <= max_chunk_size:
            current_chunk += para + "\n"
        else:
            # 装不下了，就把现在的包裹封口，把新段落作为下一个包裹的开头
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
            
    # 收尾工作：别忘了把最后没封口的那个小包裹加进去
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks