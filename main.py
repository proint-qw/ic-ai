import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from utils import qa_agent

st.set_page_config(
    page_title="芯智通-半导体知识库",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .stChatInput {padding: 20px;}
    .st-expander {border: 1px solid #4a4a4a;}
    .reportview-container .main .block-container {padding-top: 2rem;}
    .sidebar .sidebar-content {background: #f0f2f6;}
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

# 侧边栏配置
with st.sidebar:
    st.title("⚙️ 系统配置")

    # 自动获取环境变量
    env_key = os.getenv("COURSE_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.aigc369.com/v1")

    # 动态显示密钥输入
    if not env_key:
        user_key = st.text_input(
            "OpenAI API密钥：",
            type="password",
            help="未配置环境变量时需手动输入"
        )
    else:
        st.success("✅ 已检测到环境变量密钥")
        user_key = ""

    # 模型选择
    ai_model = st.selectbox(
        "选择AI模型",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0,
        help="根据问题复杂度选择模型"
    )

    st.divider()
    st.markdown("### 支持文档格式")
    st.markdown("""
    - 工艺手册 (PDF/DOCX)
    - 设备参数表 (CSV/TXT) 
    - 研发文档 (Markdown)
    - 专利文件 (PDF)
    """)

# 主界面
st.title("🔬 芯智通 - 半导体知识库系统")
st.caption("支持芯片制造全流程文档分析：光刻/刻蚀/沉积/封装工艺")

# 文件上传
uploaded_file = st.file_uploader(
    "上传技术文档",
    type=["pdf", "docx", "txt", "csv", "md"],
    help="最大文件大小：50MB"
)

# 对话界面
if uploaded_file:
    # 输入区域
    col1, col2 = st.columns([6, 1])
    with col1:
        question = st.text_input("技术问题", placeholder="请输入关于芯片制造的技术问题...")
    with col2:
        submit_btn = st.button("发送", use_container_width=True)

    # 处理问答
    if submit_btn and question:
        final_key = user_key or env_key
        if not final_key:
            st.error("需要有效的API密钥")
            st.stop()

        with st.spinner("🔍 正在解析技术文档..."):
            response = qa_agent(
                final_key,
                st.session_state["memory"],
                uploaded_file,
                question,
                ai_model
            )

        # 显示答案
        with st.chat_message("assistant"):
            st.markdown(f"**技术解答**  \n{response['answer']}")
            if response.get("source_documents"):
                with st.expander("📁 参考来源"):
                    sources = list(
                        {doc.metadata.get("source", uploaded_file.name) for doc in response["source_documents"]})
                    st.markdown("**相关文档：**  \n" + "  \n".join(f"- {s}" for s in sources))

        # 更新历史
        st.session_state["chat_history"] = response["chat_history"]

# 历史对话管理
if "chat_history" in st.session_state:
    with st.expander("📜 历史对话（点击展开）", expanded=False):
        for msg in st.session_state["chat_history"]:
            if msg.type == "human":
                st.markdown(f"**👤 用户**  \n{msg.content}")
            else:
                st.markdown(f"**🤖 AI助手**  \n{msg.content}")
            st.divider()