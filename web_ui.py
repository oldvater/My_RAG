##拼图 1：页面基础设置与请求常量
import streamlit as st
import requests

# 你的 FastAPI 后端地址（我们在 Swagger UI 调用的就是这个前缀）
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Enterprise RAG System", page_icon="🤖", layout="wide")
st.title("🤖 极简企业级 RAG 检索对话系统")

##拼图 2：左侧边栏 (Sidebar) - 用于纯文本测试

with st.sidebar:
    st.header("🗂️ 知识库管理")
    # 创建一个多行文本输入框
    new_doc = st.text_area("输入整段长博文/小说测试语料")
    
    # st.button 返回 True 说明用户点击了
    if st.button("上传并向量化进知识库"):
        if new_doc:
            # 💡 TODO 1: 拼接你之前写的 FastAPI 上传接口地址
            # 提示：上传接口是什么来着？ /rag/xxxx
            upload_url = f"{API_BASE_URL}/rag/upload" 
            
            # 使用 requests 发送 POST 请求，注意我们之前后端的参数名就是 'text'
            with st.spinner('正在切片与生成向量中...'):
                response = requests.post(upload_url, json={"text": new_doc})
                
                if response.status_code == 200:
                    st.success(f"上传成功！{response.json()}")
                else:
                    st.error("上传失败，请看后端报错")

    # st.file_uploader 返回 True 说明用户点击了
    file_obj = st.file_uploader("上传PDF", type=["pdf"])
    if file_obj:
        if st.button("开始处理并入库"):
            files = {"file":(file_obj.name, file_obj.getvalue(), "application/pdf")}
            upload_url = f"{API_BASE_URL}/rag/upload_pdf" 

            with st.spinner('正在上传中...'):
                response = requests.post(upload_url, files=files)

                if response.status_code == 200:
                    st.success(f"上传成功！{response.json()}")
                else:
                    st.error("上传失败，请看后端报错")
            


##拼图 3：主界面右侧 (Main Area) - 核心 RAG 聊天

# 初始化会话记录（State），Streamlit 每次刷新都会重置变量，所以需要存在 session_state 里
if "messages" not in st.session_state:
    st.session_state.messages = []

# 把之前的历史聊天气泡展示出来
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 底部聊天输入框
user_input = st.chat_input("搜点什么...（例如：BERT是什么？）")

if user_input:
    # 1. 马上用户输入显示成气泡，并存入历史
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. 从你的 FastAPI 后端获取大模型的结果
    with st.chat_message("assistant"):
        with st.spinner("AI 翻阅资料中..."):
            # 💡 TODO 2: 拼接你的 Query 接口地址
            query_url = f"{API_BASE_URL}/rag/query"
            
            # 💡 TODO 3: 发送网络请求
            # 提示：传入 JSON 的 key 是什么？我们在 main.py 里写的 QueryInput 的字段是什么？
            res = requests.post(query_url, json={"query": user_input, "chat_history": st.session_state.messages[:-1]})
            
            if res.status_code == 200:
                data = res.json()
                # 💡 TODO 4: 从后端返回的 dict(data) 中，提取出那个我们让 DeepSeek 生成的“干净答案”字段
                answer = data.get("answer", "解析大模型回复失败")
                
                # 可选：你可以顺便把搜到的上下文放到收起折叠窗里，显得很极客
                with st.expander("点击查看召回的 Chunk (Top-K)"):
                    st.json(data.get("context", []))
                    
                # 显示模型回答
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("RAG 服务出错了~ 检查后端")
