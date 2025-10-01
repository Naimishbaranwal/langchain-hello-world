import sys, os
sys.path.append(os.path.dirname(__file__))
from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from callbacks import AgentCallbackHandler
from pathlib import Path
# current_dir = Path(__file__).parent 
# Get current working directory
# current_path = Path.cwd()
# Load environment variables
load_dotenv(override=True)
google_api_key = os.environ.get("GEMINI_API_KEY")

@tool
def get_text_length(text: str) -> int:
    """Return the length of text by characters."""
    text = text.strip("'\n").strip('"')
    return len(text)

if __name__ == "__main__":
    tools = [get_text_length]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                                 api_key=google_api_key,
                                 callbacks=[AgentCallbackHandler()])
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools,callbacks=[AgentCallbackHandler()])

    res = agent_executor.invoke({
        "input": "What is the length of characters in DOG?"
    })
    print(res)



















