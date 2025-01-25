import os
import boto3
import pymongo
import traceback
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.callbacks import get_openai_callback
import gc
import uuid
from typing import List

# Environment variables
<AWS, OPENAI, MongoDB secret keys>

client = pymongo.MongoClient(MONGO_URL, uuidRepresentation="standard")
db = client["chat_with_doc"]
conversationcol = db["chat-history"]
conversationcol.create_index([("session_id", pymongo.ASCENDING)], unique=True)

# AWS S3 session
aws_s3 = boto3.Session(
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=S3_REGION,
)

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessageSent(BaseModel):
    session_id: str = None
    user_input: str

def load_documents_from_s3():
    """
    Load all PDF and DOCX documents from the specified S3 bucket.
    """
    s3_client = aws_s3.client('s3')
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET)
    documents = []

    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.lower().endswith('.pdf') or key.lower().endswith('.docx'):
            local_path = f"/tmp/{os.path.basename(key)}"
            s3_client.download_file(S3_BUCKET, key, local_path)
            if key.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path=local_path)
            else:
                loader = Docx2txtLoader(file_path=local_path)
            documents.extend(loader.load())
            os.remove(local_path)

    return documents

def get_response(session_id: str, query: str, model: str = "gpt-3.5-turbo-16k", temperature: float = 0):
    """
    Generate a response using a conversational model with inline citations.
    """
    # Load all documents from S3
    documents = load_documents_from_s3()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)

    # Create embeddings and store in vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(all_splits, embeddings)

    # Initialize OpenAI model
    llm = ChatOpenAI(model_name=model, temperature=temperature)

    # Create Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True  # To get source documents for citations
    )

    # Generate response with citations
    with get_openai_callback() as cb:
        result = qa_chain(
            {
                "question": query,
                "chat_history": load_memory_to_pass(session_id=session_id),
            }
        )
        answer = result["answer"]
        source_documents = result["source_documents"]

        # Construct inline citations
        citations = []
        for doc in source_documents:
            source = doc.metadata.get("source", "Unknown Source")
            citations.append(f"[{source}]")

        # Append citations to the answer
        if citations:
            answer += " " + " ".join(citations)

        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    gc.collect()
    return {"answer": answer, "total_tokens_used": cb.total_tokens}

def load_memory_to_pass(session_id: str):
    """
    Load conversation history for a given session ID.
    """
    data = conversationcol.find_one({"session_id": session_id})
    history = []
    if data:
        data = data["conversation"]
        for x in range(0, len(data), 2):
            history.extend([(data[x], data[x + 1])])
    return history

def get_session() -> str:
    """
    Generate a new session ID.
    """
    return str(uuid.uuid4())

def add_session_history(session_id: str, new_values: List):
    """
    Add conversation history to an existing session or create a new session.
    """
    document = conversationcol.find_one({"session_id": session_id})
    if document:
        conversation = document["conversation"]
        conversation.extend(new_values)
        conversationcol.update_one(
            {"session_id": session_id}, {"$set": {"conversation": conversation}}
        )
    else:
        conversationcol.insert_one(
            {
                "session_id": session_id,
                "conversation": new_values,
            }
        )

@app.post("/chat")
async def create_chat_message(chats: ChatMessageSent):
    """
    Create a chat message and obtain a response based on user input and session.
    """
    try:
        if chats.session_id is None:
            session_id = get_session()
        else:
            session_id = chats.session_id

        response = get_response(
            session_id=session_id,
            query=chats.user_input,
        )

        add_session_history(
            session_id=session_id,
            new_values=[chats.user_input, response["answer"]],
        )

        return JSONResponse(
            content={
                "response": response,
                "session_id": session_id,
            }
        )

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
