from dotenv import load_dotenv
load_dotenv(override=True)
import os
from langchain import hub
from langchain.agents import AgentExecutor #loop
from langchain.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_groq import ChatGroq

api_key=os.environ.get("Groq_key")
google_api_key=os.environ.get("GEMINI_API_KEY")

tools=[TavilySearch()]
model={"a":"qwen/qwen3-32b",
       "b":"gemma2-9b-it",
       "c":"llama-3.3-70b-versatile"}
react_prompt=hub.pull("hwchase17/react")
def main():
    for name, model_name in model.items():
        print(f"\n--- Running model {name}: {model_name} ---")
        
        llm = ChatGroq(
            temperature=0,
            api_key=api_key,
            model=model_name
        )
    # llm = ChatGroq(temperature=0,api_key=api_key ,model="deepseek-r1-distill-llama-70b")

        agent=create_react_agent(llm=llm,
                                tools=tools,
                                prompt=react_prompt)
        agent_executor=AgentExecutor(agent=agent,
                                    tools=tools,
                                    verbose=True)
        chain=agent_executor



        # def main():
        #     # print("Hello from langchain-course!")
        result = chain.invoke(
            input={
                "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
            }
        )
        print(result)




if __name__=="__main__":
    main()





