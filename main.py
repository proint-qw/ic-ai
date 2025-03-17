import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from utils import qa_agent

st.title("🧠 芯片制造知识库智能助手")

with st.sidebar:
    # 模型选择
    selected_model = st.selectbox(
        "选择AI模型",
        ("gpt-3.5-turbo", "gpt-4"),
        index=0
    )

    # API密钥处理
    env_api_key = os.getenv("COURSE_API_KEY")
    if env_api_key:
        use_env_key = st.checkbox("使用环境变量API密钥", value=True)
        if use_env_key:
            openai_api_key = env_api_key
        else:
            openai_api_key = st.text_input("自定义API密钥：", type="password")
    else:
        openai_api_key = st.text_input("请输入API密钥：", type="password")

    st.markdown("[API密钥获取指南](https://platform.openai.com/account/api-keys)")

# 初始化对话记忆
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

# 文件上传
uploaded_file = st.file_uploader("上传芯片制造技术文档（PDF）", type="pdf")
question = st.text_input("请输入您的问题", disabled=not uploaded_file)

# 处理问答流程
if uploaded_file and question:
    if not openai_api_key:
        st.warning("⚠️ 请输入有效的API密钥")
        st.stop()

    with st.spinner("正在分析芯片制造文档..."):
        try:
            response = qa_agent(
                openai_api_key=openai_api_key,
                memory=st.session_state["memory"],
                uploaded_file=uploaded_file,
                question=question,
                model_name=selected_model
            )
            st.write("## 专家解答")
            st.success(response["answer"])

            st.session_state["chat_history"] = response["chat_history"]
        except Exception as e:
            st.error(f"芯片知识引擎异常：{str(e)}")

# 显示历史对话
if "chat_history" in st.session_state:
    with st.expander("🧾 对话记录"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown("**工程师**")
                st.write(st.session_state["chat_history"][i].content)
            with col2:
                st.markdown("**芯片专家**")
                st.write(st.session_state["chat_history"][i + 1].content)
            st.divider()