import pandas as pd
import numpy as np
from groq import Groq
import os
import pinecone

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


#groq_api_key = os.getenv('GROQ_API_KEY')
#pinecone_api_key = os.getenv('PINECONE_API_KEY')

groq_api_key = "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"
pinecone_api_key="pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"
    

client = Groq(api_key = groq_api_key)
model = "llama3-8b-8192"

presidential_speeches_df = pd.read_csv('presidential_speeches.csv')
print(presidential_speeches_df.head())

garfield_inaugural = presidential_speeches_df.iloc[309].Transcript

print(garfield_inaugural)


model_id = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# create the length function
def token_len(text):
    tokens = tokenizer.encode(
        text
    )
    return len(tokens)

token_len(garfield_inaugural)


print(token_len(garfield_inaugural))


text_splitter = TokenTextSplitter(
    chunk_size=450, # 500 tokens is the max
    chunk_overlap=20 # Overlap of N tokens between chunks (to reduce chance of cutting out relevant connected text like middle of sentence)
)


chunks = text_splitter.split_text(garfield_inaugural)

for chunk in chunks:
    print(token_len(chunk))



chunk_embeddings = []
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
for chunk in chunks:
    chunk_embeddings.append(embedding_function.embed_query(chunk))

print(len(chunk_embeddings[0]),chunk_embeddings[0][:20]) #Shows first 25 embeddings out of 384

user_question = "What were James Garfield's views on civil service reform?"

prompt_embeddings = embedding_function.embed_query(user_question) 
similarities = cosine_similarity([prompt_embeddings], chunk_embeddings)[0] 
closest_similarity_index = np.argmax(similarities) 
most_relevant_chunk = chunks[closest_similarity_index]

print(most_relevant_chunk)



#display(HTML(most_relevant_chunk))
# A chat completion function that will use the most relevant exerpt(s) from presidential speeches to answer the user's question
def presidential_speech_chat_completion(client, model, user_question, relevant_excerpts):
    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "system",
                "content": "You are a presidential historian. Given the user's question and relevant excerpts from presidential speeches, answer the question by including direct quotes from presidential speeches. When using a quote, site the speech that it was from (ignoring the chunk)." 
            },
            {
                "role": "user",
                "content": "User Question: " + user_question + "\n\nRelevant Speech Exerpt(s):\n\n" + relevant_excerpts,
            }
        ],
        model = model
    )
    
    response = chat_completion.choices[0].message.content
    return response


presidential_speech_chat_completion(client, model, user_question, most_relevant_chunk)

documents = []
for index, row in presidential_speeches_df[presidential_speeches_df['Transcript'].notnull()].iterrows():
    chunks = text_splitter.split_text(row.Transcript)
    total_chunks = len(chunks)
    for chunk_num in range(1,total_chunks+1):
        header = f"Date: {row['Date']}\nPresident: {row['President']}\nSpeech Title: {row['Speech Title']} (chunk {chunk_num} of {total_chunks})\n\n"
        chunk = chunks[chunk_num-1]
        documents.append(Document(page_content=header + chunk, metadata={"source": "local"}))

print(len(documents))



print("111")
os.environ['PINECONE_API_KEY'] = "pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"

pinecone_index_name = "presidential-speeches"
docsearch = PineconeVectorStore.from_documents(documents, embedding_function, index_name=pinecone_index_name)
print("222")
### Use Chroma for open source option
#docsearch = Chroma.from_documents(documents, embedding_function)

user_question = "What were James Garfield's views on civil service reform?"
print("333")
relevent_docs = docsearch.similarity_search(user_question)
print("555")
print(relevent_docs)

relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in relevent_docs[:3]])

print(relevant_excerpts.replace("\n", "<br>"))


presidential_speech_chat_completion(client, model, user_question, relevant_excerpts)


#display(HTML(relevant_excerpts.replace("\n", "<br>")))
