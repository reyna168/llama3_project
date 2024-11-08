import os
import streamlit as st
from groq import Groq
from pinecone import Pinecone
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore

# Function to retrieve the most relevant excerpts
def get_relevant_excerpts(user_question, docsearch):
    relevent_docs = docsearch.similarity_search(user_question)
    relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join(
        [doc.page_content for doc in relevent_docs[:3]]
    )
    return relevant_excerpts

# Function to generate a response based on the user's question and relevant excerpts
def presidential_speech_chat_completion(client, model, user_question, relevant_excerpts):
    system_prompt = '''
    You are a presidential historian. Given the user's question and relevant excerpts from 
    presidential speeches, answer the question by including direct quotes from presidential speeches. 
    When using a quote, cite the speech that it was from (ignoring the chunk).
    '''
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Question: {user_question}\n\nRelevant Speech Excerpt(s):\n\n{relevant_excerpts}"},
        ],
        model=model
    )
    
    response = chat_completion.choices[0].message.content
    return response

# Initialize the Streamlit app
def main():
    st.title("Presidential Speeches RAG")
    st.write("""
    Welcome! Ask questions about U.S. presidents, like "What were George Washington's views on democracy?" or "What did Abraham Lincoln say about national unity?". 
    The app matches your question to relevant excerpts from presidential speeches and generates a response using a pre-trained model.
    """)

    model = 'llama3-8b-8192'
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load API keys
    groq_api_key = "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"
    pinecone_api_key = "pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"
    
    # Initialize Groq and Pinecone clients
    client = Groq(api_key=groq_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index_name = "presidential-speeches"
    docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)
    
    # User question input
    user_question = st.text_input("Ask a question about a U.S. president:")

    # Handle question submission
    if user_question:
        relevant_excerpts = get_relevant_excerpts(user_question, docsearch)
        response = presidential_speech_chat_completion(client, model, user_question, relevant_excerpts)
        st.write("### Response")
        st.write(response)
        st.write("### Relevant Speech Excerpts")
        st.write(relevant_excerpts)

if __name__ == "__main__":
    
    os.environ['PINECONE_API_KEY'] = "pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"

    main()
