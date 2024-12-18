import streamlit as st
from datasets import load_dataset
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import os
import time
from io import BytesIO
from base64 import b64encode
from torch import torch

from sentence_transformers import SentenceTransformer, util




# Streamlit UI setup
st.title("Fashion Product Search with Pinecone and CLIP")
st.write("This application allows you to search for similar fashion products using hybrid vector search.")

# Load dataset
@st.cache_data
def load_fashion_dataset():
    return load_dataset("ashraq/fashion-product-images-small", split="train")

fashion = load_fashion_dataset()
st.write("Dataset loaded with", len(fashion), "items.")

# Pinecone initialization
#api_key = os.environ.get('PINECONE_API_KEY') or "your_pinecone_api_key"
# initialize connection to pinecone
api_key = os.environ.get('PINECONE_API_KEY') or "pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"


cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = 'dl-ai'

# Configure Pinecone client
pc = Pinecone(api_key=api_key)
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=512, metric='dotproduct', spec=spec)
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
index = pc.Index(index_name)
st.write("Pinecone index ready.")

# Model initialization
bm25 = BM25Encoder()
metadata = fashion.remove_columns('image').to_pandas()
bm25.fit(metadata['productDisplayName'])
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device='cuda' if torch.cuda.is_available() else 'cpu')

# Display search input
query = st.text_input("Enter your search query (e.g., 'dark blue jeans for men'):")

if query:
    # Encode query
    sparse = bm25.encode_queries(query)
    dense = model.encode(query).tolist()

    # Perform search
    result = index.query(
        top_k=14,
        vector=dense,
        sparse_vector=sparse,
        include_metadata=True
    )

    # Fetch matching images
    images = fashion['image']
    imgs = [images[int(r["id"])].resize((100, 150)) for r in result["matches"]]

    # Display results
    st.write("Search Results:")
    cols = st.columns(5)
    for i, img in enumerate(imgs):
        with cols[i % 5]:
            st.image(img, caption=result["matches"][i]["metadata"]["productDisplayName"], use_column_width=True)

st.write("---")
st.write("Powered by Streamlit, Pinecone, and SentenceTransformers.")
