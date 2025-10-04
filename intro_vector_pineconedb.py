from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from huggingface_hub import InferenceClient
import numpy as np
client = InferenceClient(
    provider="nebius",
    api_key=os.environ["HF_TOKEN"],
)

# result = client.feature_extraction(
#     "Today is a sunny day and I will get some ice cream.",
#     model="Qwen/Qwen3-Embedding-8B",
# )

Pinecone_api_key=os.environ.get("Pinecone_api_key")
pc = Pinecone(api_key=Pinecone_api_key)
# Create index only if not exists
index_name = "new"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
index = pc.Index(index_name)



loader=TextLoader(r"D:\GenAi Code\Agent\langchain-course\medium_blog.txt",encoding="utf-8")
documet=loader.load()
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts=text_splitter.split_documents(documet)
print(f"created {len(texts)} chunks")


# Generate embeddings for all chunks (flattening each vector)
embeddings = []
for doc in texts:
    vec = client.feature_extraction(doc.page_content, model="Qwen/Qwen3-Embedding-8B")
    # vec is usually [[...]] → take first element
    embeddings.append(vec[0] if isinstance(vec[0], list) else vec)



#store in pinecone
upsert_data = [
    (str(i), vec.flatten().tolist(), {"text": doc.page_content})
    for i, (vec, doc) in enumerate(zip(embeddings, texts))
]

index.upsert(vectors=upsert_data)


print(f"✅ {len(upsert_data)} chunks embedded and stored in Pinecone")




