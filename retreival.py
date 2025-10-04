import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from huggingface_hub import InferenceClient
import json
from langchain.schema import Document

load_dotenv(override=True)

# Initialize HuggingFace client
client = InferenceClient(provider="nebius"
                         , api_key=os.environ["HF_TOKEN"])

# Wrapper for LangChain embeddings
class QwenEmbeddings:
    model_name = "Qwen/Qwen3-Embedding-8B"

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            vec = client.feature_extraction(text, model=self.model_name)
            # Convert to list to be compatible with Pinecone
            embeddings.append(vec[0] if isinstance(vec[0], list) else vec[0].tolist())
        return embeddings

    def embed_query(self, text):
        vec = client.feature_extraction(text, model=self.model_name)
        # Convert to list to be compatible with Pinecone
        return vec[0] if isinstance(vec[0], list) else vec[0].tolist()
    


if __name__=="__main__":
    # Initialize embeddings
    embeddings = QwenEmbeddings()

    # Initialize Google Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=os.environ["GEMINI_API_KEY"]
    )

    # Define query
    query = "what is difference between HNSW and IVF in machine learning?"

    # Initialize Pinecone vector store
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    # Pull LangChain retrieval QA chat prompt
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Combine documents chain
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    # Run query
    result = retrieval_chain.invoke(input={"input": query})
    print(result['answer'])

    # def pretty_print_result(result):
    #     print("### Query:\n")
    #     print(result['input'], "\n")
        
    #     print("### Retrieved Context:\n")
    #     for i, doc in enumerate(result['context']):
    #         print(f"#### Document {i} (ID: {doc.id}):\n")
    #         print(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))
    #         print("\n---\n")
        
    #     print("### Answer:\n")
    #     print(result['answer'])

    # # Example usage
    # pretty_print_result(result)