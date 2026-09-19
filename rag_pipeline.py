from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class RAGWrapper:
    def __init__(self, retriever, chain):
        self.retriever = retriever
        self.chain = chain

    def invoke(self, inputs):
        query = inputs["input"]
        docs = self.retriever.invoke(query)
        answer = self.chain.invoke(query)
        return {"answer": answer, "context": docs}

def build_rag_system(pdf_path: str):
    # 1. Load document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    # 3. Vector Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. LLM & Chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer strictly based on the provided context. Cite page numbers.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return RAGWrapper(retriever, chain)
