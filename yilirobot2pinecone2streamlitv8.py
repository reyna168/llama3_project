import os
import streamlit as st
from groq import Groq
from pinecone import Pinecone
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore

class ChatApplication:
    def __init__(self, model, groq_api_key, pinecone_api_key, pinecone_index_name):
        # Set API keys
        os.environ['PINECONE_API_KEY'] = pinecone_api_key
        
        # Initialize API clients and embeddings
        self.model = model
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.client = Groq(api_key=groq_api_key)
        self.pinecone = Pinecone(api_key=pinecone_api_key)
        self.docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=self.embedding_function)

    def get_relevant_excerpts(self, user_question):
        relevent_docs = self.docsearch.similarity_search(user_question)
        relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join(
            [doc.page_content for doc in relevent_docs[:3]]
        )
        return relevant_excerpts

    def presidential_speech_chat_completion(self, user_question, relevant_excerpts):
        system_prompt = '''
        你是問答系統。針對用戶的問題和回答合適的人機產品(ignoring the chunk).
        '''
        
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {user_question}\n\nRelevant Speech Excerpt(s):\n\n{relevant_excerpts}"},
            ],
            model=self.model
        )
        
        response = chat_completion.choices[0].message.content
        return response

    def run(self):
        # Set up Streamlit app
        st.title("人機產品 Q&A")
        st.write("歡迎！詢問有關人機產品的問題，例如支援規格尺寸")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("詢問人機面版相關問題:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            relevant_excerpts = self.get_relevant_excerpts(prompt)
            response = self.presidential_speech_chat_completion(prompt, relevant_excerpts)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.write(response)
            
            response2 = relevant_excerpts + response
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response2})


if __name__ == "__main__":
    # API keys and configuration
    model_name = 'llama3-8b-8192'
    groq_api_key = "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"
    pinecone_api_key = "pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"
    pinecone_index_name = "yilrobot"

    # Initialize and run app
    app = ChatApplication(model_name, groq_api_key, pinecone_api_key, pinecone_index_name)
    app.run()
