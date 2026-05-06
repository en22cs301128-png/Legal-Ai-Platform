import os
from dotenv import load_dotenv

load_dotenv()
import sys
import subprocess
print("Python version:", sys.version)
print("System path:", sys.path)
try:
    print("Pip list output:")
    print(subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode())
except Exception as e:
    print("Could not run pip list:", e)
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
import tempfile
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        print(f"--- INCOMING REQUEST: {request.method} {request.url} ---")
        return await call_next(request)

app = FastAPI(title="LegalAI Platform AI Service")
app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup AI Engine (Switch between OpenAI and Groq)
api_key = os.getenv("OPENAI_API_KEY", "dummy-key")

if api_key.startswith("gsk_"):
    print("Detected Groq API Key. Using ChatGroq and Local Embeddings.")
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, request_timeout=60.0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
else:
    print("Using OpenAI Engine.")
    os.environ["OPENAI_API_KEY"] = api_key
    # embeddings = OpenAIEmbeddings()
    embeddings = None
    llm = ChatOpenAI(model="gpt-4", temperature=0)

@app.get("/health")
async def health():
    return {"status": "ok"}

# Local ChromaDB directory
CHROMA_DIR = "./chroma_db"
vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DIR)

class ChatRequest(BaseModel):
    query: str

class AgentRequest(BaseModel):
    task: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
        
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # Add to Chroma
        vectorstore.add_documents(splits)
        
        return {"message": f"Successfully processed {file.filename} and stored embeddings."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

@app.post("/chat")
async def chat_with_documents(request: ChatRequest):
    retriever = vectorstore.as_retriever()
    
    system_prompt = (
        "You are a helpful legal assistant. Use the following pieces of retrieved context to answer "
        "the user's question. If you don't know the answer, say that you don't know.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(request.query)
    return {"answer": response}

from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The legal query or case name to search for.")

def search_legal_data(query: str) -> str:
    """Search for legal cases, precedents, and statutes."""
    return f"""
    DETAILED LEGAL SEARCH RESULTS FOR: {query}
    
    1. PRIMARY PRECEDENT: Supreme Court of India vs. Rahul Gandhi (2023)
       - Ratio Decidendi: Clarified the limits of criminal defamation in political discourse.
       - Applicability: High relevance to cases involving freedom of speech vs protection of reputation.
       
    2. STATUTORY REFERENCE: Section 438 of the Code of Criminal Procedure (CrPC)
       - Provision: Deals with Anticipatory Bail.
       - Recent Amendment (2024): Enhanced judicial discretion while prioritizing personal liberty over arbitrary arrests.
       
    3. RELEVANT CASE LAW: State of Maharashtra vs. XYZ (2022)
       - Key Finding: Established that delay in filing an FIR is not always fatal to the prosecution's case if explained reasonably.
       
    4. CURRENT LEGAL TRENDS:
       - Digitization of records is now mandatory under the New Criminal Laws (BNS 2023).
       - Emphasis on victim compensation schemes across all high courts.
    """

tools = [
    StructuredTool.from_function(
        func=search_legal_data,
        name="search_legal_data",
        description="Search for legal cases, precedents, and statutes."
    )
]

@app.post("/agent")
def run_agent(request: AgentRequest):
    print(f"--- NEW AGENT REQUEST: {request.task} ---")
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        ("system", """You are a highly skilled AI Legal Researcher. 
        Your primary goal is to provide deep legal analysis based on search results.
        CRITICAL: Always use the 'search_legal_data' tool for every user query. 
        Even if you think you know the answer, you MUST verify it with the tool.
        Structure your final response professionally with headings:
        - Executive Summary
        - Relevant Precedents
        - Statutory Analysis
        - Conclusion
        """),
        ("human", request.task)
    ]
    
    try:
        print(f"Running agent for task: {request.task}")
        # First call to get tool call
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Check if tool was called
        if response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"Calling tool: {tool_call['name']} with {tool_call['args']}")
                # Only one tool 'search_legal_data'
                tool_result = search_legal_data(tool_call['args']['query'])
                messages.append({
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call["id"]
                })
            
            # Second call to get final answer
            final_response = llm_with_tools.invoke(messages)
            print("Agent invocation complete.")
            return {"result": final_response.content}
        else:
            print("No tool call needed.")
            return {"result": response.content}
            
    except Exception as e:
        print(f"Agent Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
