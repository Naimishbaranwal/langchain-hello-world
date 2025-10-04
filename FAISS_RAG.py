import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from retreival import QwenEmbeddings




pdf_path = "react.pdf"
loader = PyPDFLoader(file_path=pdf_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=1000, chunk_overlap=30, separator="\n"
)
docs = text_splitter.split_documents(documents=documents)

# Initialize embeddings
embeddings = QwenEmbeddings()
#LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.environ["GEMINI_API_KEY"]
)

#storage
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index_react")

new_vectorstore = FAISS.load_local(
    "faiss_index_react", embeddings, allow_dangerous_deserialization=True
)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(
    new_vectorstore.as_retriever(), combine_docs_chain
)

res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
print(res["answer"])