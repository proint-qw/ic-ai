from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    Docx2txtLoader, UnstructuredMarkdownLoader
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def qa_agent(api_key, memory, uploaded_file, question, model_name="gpt-3.5-turbo"):
    # 动态获取API配置
    api_base = os.getenv("OPENAI_API_BASE", "https://api.aigc369.com/v1")

    # 初始化模型
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=0.2
    )

    # 处理多格式文档
    file_ext = uploaded_file.name.split(".")[-1].lower()
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 文档加载器选择
    loaders = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "txt": TextLoader,
        "csv": lambda path: CSVLoader(path, encoding="utf-8"),
        "md": UnstructuredMarkdownLoader
    }

    if file_ext not in loaders:
        raise ValueError(f"不支持的文件格式: {file_ext}")

    try:
        docs = loaders[file_ext](temp_path).load()
    except Exception as e:
        os.remove(temp_path)
        raise RuntimeError(f"文档解析失败: {str(e)}")

    # 专业文本分块（针对芯片文档优化）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=[
            "\n\n", "## 工艺流程", "### 技术参数",
            "表1：", "图2：", "注意事项："
        ]
    )
    texts = text_splitter.split_documents(docs)

    # 向量数据库
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        openai_api_base=api_base
    )
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 专业Prompt模板
    from langchain_core.prompts import PromptTemplate
    prompt_template = """作为芯片制造专家，请严格根据技术文档回答：

**文档内容：**
{context}

**历史对话：** {chat_history}
**当前问题：** {question}
请给出符合半导体行业标准的专业解答："""

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate.from_template(prompt_template)
        },
        return_source_documents=True
    )

    response = qa.invoke({"question": question})
    os.remove(temp_path)
    return response