from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent,AgentExecutor
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
# from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
import re

load_dotenv(override=True)
api_key=os.environ.get("Groq_key")
google_api_key=os.environ.get("GEMINI_API_KEY")
tavily_api_key=os.environ.get("TAVILY_API_KEY")
tavily = TavilySearch(api_key="tavily_api_key")

@tool
def multiply(a:float,b:float)->float:
    """Multiply 'a' times 'b' ."""
    return a*b


if __name__=="__main__":

    prompt=ChatPromptTemplate.from_messages(
        [
            ("system", "you're a helpful assistant"),
            ("human","{input}"),
            ("placeholder","{agent_scratchpad}")
        ]
    )

    tools=[TavilySearch(),multiply]
    llm= ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=google_api_key)

    agent=create_tool_calling_agent(llm,tools,prompt)
    agent_executor=AgentExecutor(agent=agent, tools=tools)

    res=agent_executor.invoke(
        {
            "input":"what is the weather in dubai right now? compare it with san francisco, output should be in celsius"
        }
    )
    print(res)