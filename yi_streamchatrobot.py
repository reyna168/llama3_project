import streamlit as st
from groq import Groq
from langchain_pinecone import PineconeVectorStore
import os
import pinecone

from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Set API keys and initialize Pinecone
#groq_api_key = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
#pinecone_api_key = os.getenv("PINECONE_API_KEY", "your_pinecone_api_key_here")
from pinecone import Pinecone, ServerlessSpec


groq_api_key = "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"
pinecone_api_key="pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"



pc = Pinecone(
        api_key=pinecone_api_key
    )

    # Now do stuff
# if 'yilrobot' not in pc.list_indexes().names():
#         pc.create_index(
#             name='my_index',
#             dimension=1536,
#             metric='euclidean',
#             spec=ServerlessSpec(
#                 cloud='aws',
#                 region='us-west-2'
#             )
#         )

# Initialize Groq client and language model
client = Groq(api_key=groq_api_key)
model = "llama3-8b-8192"

# Connect to Pinecone index and set up embeddings
pinecone_index_name = "yilrobot"
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Pinecone Vector Store
docsearch = PineconeVectorStore(embedding_function=embedding_function, index_name=pinecone_index_name)

# Define Q&A function for Streamlit
def qa_speech_chat_completion(client, model, user_question, relevant_excerpts):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "你是問答機器人(ignoring the chunk)."},
            {
                "role": "user",
                "content": f"User Question: {user_question}\n\nRelevant Specification(s):\n\n{relevant_excerpts}",
            },
        ],
        model=model,
    )
    response = chat_completion.choices[0].message.content
    return response

# Streamlit interface
st.title("人機產品問答")
user_question = st.text_input("請輸入您的問題：")

if st.button("搜尋"):
    # Search for relevant documents
    relevent_docs = docsearch.similarity_search(user_question)
    relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join(
        [doc.page_content for doc in relevent_docs[:3]]
    )

    # Display relevant excerpts
    st.write("相關資訊：")
    st.code(relevant_excerpts, language="markdown")

    # Get AI's answer
    response = qa_speech_chat_completion(client, model, user_question, relevant_excerpts)

    # Display response
    st.write("回答：")
    st.write(response)
