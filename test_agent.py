import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search tool."""
    return "Paris"

api_key = os.getenv("OPENAI_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)
agent = create_react_agent(llm, [search])

try:
    result = agent.invoke({"messages": [("human", "What is the capital of France?")]})
    print(result["messages"][-1].content)
except Exception as e:
    print(f"Error: {e}")
