import os
import shutil
import traceback
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from typing import List, Optional, Dict
from dotenv import load_dotenv
import gc
import uuid

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
vectorstore: Optional[FAISS] = None
session_history: Dict[str, List] = {}

# Models for requests
class ChatMessageSent(BaseModel):
    session_id: Optional[str] = None
    user_input: str


def initialize_vectorstore():
    """
    Initialize an empty FAISS vector store.
    """
    global vectorstore
    if vectorstore is None:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS(embeddings)


def add_documents_to_vectorstore(file_path: str):
    """
    Incrementally add documents from a single PDF to the vectorstore.
    """
    global vectorstore

    # Load the PDF and split into chunks
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    document_chunks = text_splitter.split_documents(documents)

    # Add chunks to the existing vector store
    embeddings = OpenAIEmbeddings()
    if vectorstore is None:
        vectorstore = FAISS.from_documents(document_chunks, embeddings)
    else:
        vectorstore.add_documents(document_chunks)


def get_response(session_id: str, query: str, model: str = "gpt-4o-mini", temperature: float = 0):
    """
    Generate a response using a conversational model and an existing vectorstore.
    """
    global vectorstore
    if vectorstore is None:
        raise HTTPException(status_code=400, detail="No documents available. Please upload files first.")

    # Initialize chat model
    llm = ChatOpenAI(model_name=model, temperature=temperature)

    # Build retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # Generate response
    with get_openai_callback() as cb:
        result = qa_chain({"question": query, "chat_history": session_history.get(session_id, [])})
        answer = result["answer"]
        source_documents = result["source_documents"]

        # Append citations to the answer
        citations = [f"[{doc.metadata.get('source', 'Unknown Source')}]" for doc in source_documents]
        if citations:
            answer += " " + " ".join(citations)

        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    gc.collect()
    return {"answer": answer, "total_tokens_used": cb.total_tokens}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a file, save it locally, and add its content to the vectorstore incrementally.
    """
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Incrementally add the new file to the vectorstore
        add_documents_to_vectorstore(file_path)

        return JSONResponse(content={"message": f"File '{file.filename}' uploaded and added to vectorstore."})
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/chat")
async def create_chat_message(chats: ChatMessageSent):
    """
    Handle user queries by retrieving from the vectorstore and generating a response.
    """
    try:
        session_id = chats.session_id or str(uuid.uuid4())
        response = get_response(session_id=session_id, query=chats.user_input)
        session_history.setdefault(session_id, []).append((chats.user_input, response["answer"]))
        return JSONResponse(content={"response": response, "session_id": session_id})
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
