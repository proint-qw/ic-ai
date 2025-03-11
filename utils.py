from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def qa_agent(openai_api_key, memory, uploaded_file, question, model_name, api_base):
    # 合并密钥来源
    final_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    # 合并基地址来源
    final_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.aigc369.com/v1")

    # 初始化模型
    model = ChatOpenAI(
        model=model_name,
        openai_api_key=final_key,
        openai_api_base=final_base
    )

    # 初始化Embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=final_key,
        openai_api_base=final_base
    )

    # 处理PDF文件
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 文档处理
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", "；"]
    )
    texts = text_splitter.split_documents(docs)

    # 向量数据库
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 对话链
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    response = qa.invoke({"question": question})
    return response