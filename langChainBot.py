import os
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import tempfile
from typing import List
# import boto3

# Load environment variables
load_dotenv(find_dotenv())
MODEL = os.getenv("MODEL",'text-embedding-3-large')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY",'pcsk_3kpoCH_MKKLpZgMWNcL3q239kNKgwjX8VivWJRK714vRvvUhGNJYwkavWPqhZ6mWYv7TTm')
INDEX_NAME = os.getenv("INDEX_NAME","rag-chatbot")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY",'pcsk_3kpoCH_MKKLpZgMWNcL3q239kNKgwjX8VivWJRK714vRvvUhGNJYwkavWPqhZ6mWYv7TTm')
ENVIRNOMENT = os.getenv("ENVIRNOMENT",'us-east-1')
PINECONE_DIMENSION = int(os.getenv('PINECONE_DIMENSION',1024))

# S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
# AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
# AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
# AWS_REGION = os.getenv('AWS_REGION')

# Initialize Pinecone client (new style)
pc = Pinecone(api_key=PINECONE_API_KEY,environment=ENVIRNOMENT)
# print(pc.list_indexes(),">>>>>>>>>>>>>>>>>>>>>>>>>>>pc")

#Function to save norm to pincone
def save_pdf_pincone(relative_path):
    # s3_client = boto3.client(
    #     's3',
    #     aws_access_key_id = AWS_ACCESS_KEY,
    #     aws_secret_access_key = AWS_SECRET_KEY,
    #     region_name = AWS_REGION
    # )
    # bucket_name = S3_BUCKET_NAME  
    # local_filename = relative_path.split('/')[-1]
    local_filename = relative_path

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        # s3_client.download_file(bucket_name, relative_path, tmp_file.name)
        chunks = load_and_chunk_pdf(tmp_file.name)
        save_to_pinecone(chunks, local_filename)

#Norm pdf Loading and Chunking
def load_and_chunk_pdf(pdf_path, chunk_size=800, chunk_overlap=200):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"File splits into chunks.")
    return text_splitter.split_documents(documents)

#Function to Save to Pinecone
def save_to_pinecone(docs, filename):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model=MODEL)
    for doc in docs:
        doc.metadata = {"filename": filename}

    index_name =INDEX_NAME
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index_name
    )
    print(f"Chunks saved into pinecone index.")

#Function to delete from  pinecone
def delete_pdf_from_pinecone(pdf_name):
    pc = Pinecone(api_key=PINECONE_API_KEY,environment=ENVIRNOMENT)
    index = pc.Index(INDEX_NAME)

    filter_query = {"filename": pdf_name}
    BATCH_SIZE = 1000  # Adjust batch size as needed

    ids_to_delete = []

    while True:
        query_result = index.query(
            vector=[0] * PINECONE_DIMENSION,
            top_k=BATCH_SIZE,
            filter=filter_query,
            include_metadata=True
        )
        batch_ids = [match['id'] for match in query_result['matches']]
        if not batch_ids:
            break
        ids_to_delete.extend(batch_ids)
        if len(batch_ids) < BATCH_SIZE:
            break

    if ids_to_delete:
        index.delete(ids=ids_to_delete)
        print(f"Deleted {len(ids_to_delete)} vectors.")
        return (f"Deleted {len(ids_to_delete)} vectors.")
    else:
        return("No vectors found matching the filter.")

#Prompt for Open Ai
template = """
        You are an expert PDF Question & Answer assistant named Lorenzo, designed to answer questions strictly based on the provided PDF content. 

        ### 🌍 **Language Specification:**  
        - Respond in English only, regardless of the original PDF content’s language.  
        - Use proper grammar, sentence structure, and idiomatic expressions of the selected language.  

        ### 🛑 **Important Instructions:**  
        - **ONLY** use the content from the provided PDFs to answer the question.  
        - If the PDF content does not contain relevant information, say: 
            # "Nessun contenuto rilevante trovato nei PDF forniti." 
            "No data Found."   
            (Use the corresponding language.)  
        - Do **NOT** make assumptions or provide external knowledge.  

        ### 🔍 **Spelling and Term Variations:**  
        If you detect possible spelling variations or alternate terms, consider them before responding.  
        For example:  
        - "Elon Musk" → "ElonMusk", "E. Musk"  
        - "Barack Obama" → "B. Obama", "BarackO"  

        ---

        ### 📄 **PDF Content:**  
        {context}

        ---

        ### ❓ **Question:**  
        {question}

        ---

        ### ✅ **Helpful Answer (strictly based on the PDF content, in Italian):**
"""


QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question","language"],
    template=template,
)

# Question Answering
def answer_question(query:str,pdf_names:List[str],top_k=1):
    if not pdf_names:
        raise ValueError("È necessario specificare almeno un file PDF.")
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model=MODEL)
    vectordb = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    retriever = vectordb.as_retriever(
        search_kwargs={
            "k":top_k,
            "filter":{"filename": {"$in": pdf_names}}
        })

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT
            }
    )
    result = qa_chain.invoke(query)
    clean_result = result["result"].replace("\n", " ").replace("  ", " ").strip()

    source_docs = result.get("source_documents", [])
    if not source_docs or not clean_result:
        return {
            "query": query,
            "result": "Nessun contenuto rilevante trovato nei PDF forniti."
        }
    return {
    "query": query,
    "result": clean_result,
    # "source_files": list(set(doc.metadata['filename'] for doc in source_docs))
}