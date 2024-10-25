import os
import psycopg2
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


# Pinecone 數據庫

API_KEY ="be8b9a00-97b4-47d4-8e8a-56f0c0da9700"

# 数据库连接参数
DB_NAME = "your_db"
DB_USER = "your_user"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = "5432"

# 加载嵌入模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 根据需要选择模型

# 加载 LLaMA3 模型和 tokenizer
MODEL_NAME = "facebook/llama-3"  # 替换为实际模型名称
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# 数据库设置
Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    embedding = Column(Text)  # pgvector 类型，具体视 ORM 支持情况而定

# 创建数据库引擎和会话
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
Session = sessionmaker(bind=engine)
session = Session()

# 向量嵌入生成函数
def get_embedding(text):
    embedding = embedding_model.encode(text)
    return embedding.tolist()

# 搜索相似文档函数
def search_similar_documents(query, top_k=5):
    query_embedding = get_embedding(query)
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    
    # 使用 pgvector 的 <-> 运算符进行向量距离计算
    cur.execute("""
        SELECT id, content
        FROM documents
        ORDER BY embedding <-> %s
        LIMIT %s
    """, (query_embedding, top_k))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return [row[1] for row in results]

# 回答生成函数
def generate_answer(query, context, max_length=150, temperature=0.7):
    prompt = f"Context: {context}\n\nQ: {query}\nA:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("A:")[-1].strip()
    return answer

# 回答问题函数
def answer_question(query):
    similar_docs = search_similar_documents(query)
    context = "\n\n".join(similar_docs)
    answer = generate_answer(query, context)
    return answer

# 创建 Flask 应用
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("question")
    if not query:
        return jsonify({"error": "No question provided"}), 400
    
    answer = answer_question(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

    