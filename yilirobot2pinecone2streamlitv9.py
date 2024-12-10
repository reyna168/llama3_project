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
    print("get_relevant_excerpts")
    print("get_relevant_excerpts")
    return relevant_excerpts


# Function to generate a response based on the user's question and relevant excerpts
def presidential_speech_chat_completion(client, model, user_question, relevant_excerpts):
    system_prompt = '''
    你是問答系統。針對用戶的問題和回答合適的人機產品(ignoring the chunk).
    '''
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Question: {user_question}\n\nRelevant Speech Excerpt(s):\n\n{relevant_excerpts}"},
        ],
        model=model
    )
    
    response = chat_completion.choices[0].message.content
    print("presidential_speech_chat_completion")
    return response

# Initialize the Streamlit app
def main():
    st.title("怡良人機產品 Q&A")
    st.write("""歡迎！詢問有關人機產品的問題，例如支援規格尺寸""")

    model = 'llama3-8b-8192'
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load API keys
    groq_api_key = "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"
    pinecone_api_key = "pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"
    
    # Initialize Groq and Pinecone clients
    client = Groq(api_key=groq_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index_name = "yilrobot"

    docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)
    
    # User question input
    # user_question = st.text_input("詢問人機面版相關問題:")
    
    # if st.button("Submit"):
    #     if user_question:
    #         relevant_excerpts = get_relevant_excerpts(user_question, docsearch)
    #         response = presidential_speech_chat_completion(client, model, user_question, relevant_excerpts)
    #         st.write("### Response")
    #         st.write(response)
    #         st.write("### Relevant Speech Excerpts")
    #         st.write(relevant_excerpts)
    #     else:
    #         st.write("Please enter a question.")

    
    
    #######
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Accept user input
    if prompt := st.chat_input("詢問人機面版相關問題:"):
    #if prompt := st.text_input("詢問人機面版相關問題:"):
    # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        relevant_excerpts = get_relevant_excerpts(prompt, docsearch)
        response = presidential_speech_chat_completion(client, model, prompt, relevant_excerpts)
            
        #print(type(relevant_excerpts))
        #print(type(response))
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write(relevant_excerpts+response)
        #Add assistant response to chat history
        
        #st.session_state.messages.append({"role": "assistant", "content": response2})

        st.session_state.messages.append({"role": "assistant", "content": response+relevant_excerpts})

    ########

if __name__ == "__main__":
    
    os.environ['PINECONE_API_KEY'] = "pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"

    #main()
    st.title("怡良人機產品 Q&A")
    st.write("""歡迎！詢問有關人機產品的問題，例如支援規格尺寸""")

    model = 'llama3-8b-8192'
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load API keys
    groq_api_key = "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"
    pinecone_api_key = "pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"
    
    # Initialize Groq and Pinecone clients
    client = Groq(api_key=groq_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index_name = "yilrobot"

    docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)
    
    
    
    #######
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Accept user input
    if prompt := st.chat_input("詢問人機面版相關問題:"):
    #if prompt := st.text_input("詢問人機面版相關問題:"):
    # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        relevant_excerpts = get_relevant_excerpts(prompt, docsearch)
        response = presidential_speech_chat_completion(client, model, prompt, relevant_excerpts)
            
        #print(type(relevant_excerpts))
        #print(type(response))
        # Display assistant response in chat message 
        msg = relevant_excerpts+response
        with st.chat_message("assistant"):
            msg = relevant_excerpts+response
            response2 = st.write(msg)
        #Add assistant response to chat history
        
        #st.session_state.messages.append({"role": "assistant", "content": response2})
        st.session_state.messages.append({"role": "assistant", "content": msg})


