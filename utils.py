from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def qa_agent(openai_api_key, memory, uploaded_file, question, model_name="gpt-3.5-turbo"):
    # 优先使用环境变量中的API密钥
    env_api_key = os.getenv("COURSE_API_KEY")
    final_api_key = env_api_key or openai_api_key

    # 初始化大模型
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=final_api_key,
        openai_api_base="https://api.aigc369.com/v1",  # 课程专用API地址
        temperature=0.3
    )

    # 处理PDF文档
    temp_file_path = "temp_doc.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 文档分割策略
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", "§"]
    )
    texts = text_splitter.split_documents(docs)

    # 半导体领域专用向量库
    embeddings = OpenAIEmbeddings(
        openai_api_key=final_api_key,
        openai_api_base="https://api.aigc369.com/v1"
    )
    vector_db = FAISS.from_documents(texts, embeddings)

    # 构建问答链
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        ),
        memory=memory,
        chain_type="stuff",
        verbose=True
    )

    return qa_chain.invoke({"question": question})