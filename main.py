import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from utils import qa_agent

st.title("📑 芯片制造知识库AI助手")

with st.sidebar:
    st.header("配置设置")

    # 环境变量密钥
    env_key = os.getenv("OPENAI_API_KEY", "")

    # 用户密钥输入
    openai_api_key = st.text_input(
        "OpenAI API密钥：",
        type="password",
        value=env_key,
        help="管理员已配置密钥时可留空" if env_key else None
    )

    # 模型选择
    model_name = st.selectbox(
        "选择AI模型：",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0
    )

    # API基地址
    api_base = st.text_input(
        "API基地址（可选）：",
        value=os.getenv("OPENAI_API_BASE", "https://api.aigc369.com/v1"),
        help="默认使用课程提供的API地址"
    )

    if env_key:
        st.info("ℹ️ 检测到预配置密钥，输入框可留空直接使用")
    st.markdown("[获取API密钥](https://platform.openai.com/account/api-keys)")

# 密钥验证
if not openai_api_key:
    if env_key:
        openai_api_key = env_key
    else:
        st.error("❌ 需要提供OpenAI API密钥（输入或环境变量）")
        st.stop()

# 初始化会话内存
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

# 文件上传和提问
uploaded_file = st.file_uploader("上传芯片制造相关PDF文档", type="pdf")
question = st.text_input("请输入技术问题", disabled=not uploaded_file)

if uploaded_file and question:
    with st.spinner("AI分析中..."):
        response = qa_agent(
            openai_api_key,
            st.session_state["memory"],
            uploaded_file,
            question,
            model_name,
            api_base
        )
    st.write("### 专家解答")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

# 历史消息展示
if "chat_history" in st.session_state:
    with st.expander("对话历史"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human = st.session_state["chat_history"][i]
            ai = st.session_state["chat_history"][i + 1]
            st.markdown(f"**用户**：{human.content}")
            st.markdown(f"**AI专家**：{ai.content}")
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()