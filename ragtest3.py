import streamlit as st
import ollama
import faiss
import pandas as pd
import numpy as np
import os
import pickle

# 定義資料庫文件路徑
FAISS_INDEX_FILE = "faiss_index.index"
DOCS_FILE = "docs.pkl"

# 初始化 Faiss 索引和文檔列表
def initialize_faiss(dimension=768):
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCS_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(DOCS_FILE, "rb") as f:
            documents = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(dimension)  # 使用 L2 距離
        documents = []
    return index, documents

# 保存 Faiss 索引和文檔列表
def save_faiss(index, documents):
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(documents, f)

# 設置資料庫
def setup_database_faiss():
    with st.spinner('正在讀取資料...'):
        if st.session_state.select_old:
            index = st.session_state.index
            documents = st.session_state.documents
        else:
            file = st.session_state.source_docs
            if file is not None:
                documents_df = pd.read_excel(file, header=None)
                documents = documents_df[0].tolist()
                index, existing_docs = initialize_faiss()
                for content in documents:
                    embedding = ollama.embeddings(model="mxbai-embed-large", prompt=content)["embedding"]
                    embedding_np = np.array(embedding).astype('float32')
                    index.add(np.expand_dims(embedding_np, axis=0))
                save_faiss(index, documents)
                st.session_state.index = index
                st.session_state.documents = documents
        st.success('資料庫加載完成！')

# 處理用戶輸入
def handle_user_input_faiss(user_input, index, documents):
    response = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")
    query_embedding = np.array(response["embedding"]).astype('float32').reshape(1, -1)
    k = 3  # 取最相關的3個結果
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[idx] for idx in indices[0] if idx < len(documents)]

    output = ollama.generate(
        model=st.session_state.selected_model,
        prompt=f"你是最好的模型，請使用這些資料: {retrieved_docs}. 回應這個問題並使用中文: {user_input}"
    )

    st.text("回答：")
    st.write(output['response'])

# 主函數
def main():
    mode = st.sidebar.radio(
        "資料庫類型：",
        ('資料庫列表', '無資料'))
    
    if mode == '資料庫列表':
        # 由於 Faiss 不提供多個集合的管理，假設只有一個索引
        st.session_state.select_old = True
        st.session_state.select_chat = False
        index, documents = initialize_faiss()
        if index.ntotal > 0:
            st.session_state.index = index
            st.session_state.documents = documents
            st.success('已加載現有資料庫。')
        else:
            st.info('請上傳文檔。')
    elif mode == '無資料':
        st.session_state.source_docs = st.sidebar.file_uploader(label="上傳文檔", type="xlsx", accept_multiple_files=False)
        if st.sidebar.button("提交文檔"):
            setup_database_faiss()
        st.session_state.select_old = False
        st.session_state.select_chat = False

    st.title("LLM+RAG本地知識問答")
    st.session_state.selected_model = st.selectbox(
        "請選擇模型：", [model["name"] for model in ollama.list()["models"]])
    user_input = st.text_area("您想問什麼？", "")

    if st.button("送出"):
        if user_input:
            if st.session_state.select_old and 'index' in st.session_state and 'documents' in st.session_state:
                handle_user_input_faiss(user_input, st.session_state.index, st.session_state.documents)
            else:
                st.warning("資料庫尚未加載或上傳文檔！")
        else:
            st.warning("請輸入問題！")

if __name__ == '__main__':
    main()
