from dotenv import load_dotenv
load_dotenv(override=True)
import os
from langchain import hub
from langchain.agents import AgentExecutor #loop
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda #convert python to callable
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schema import AgentResponse
from langchain_openai import ChatOpenAI

api_key=os.environ.get("Groq_key")
google_api_key=os.environ.get("GEMINI_API_KEY")

tools=[TavilySearch()]
models={
    # "a": "qwen/qwen3-32b",
    # "b": "gemma2-9b-it",
    # "c": "llama-3.3-70b-versatile",
    # "d": "llama-3.1-8b-instant",
    # "e": "groq/compound",
    # "f": "openai/gpt-oss-120b",
    # "g": "moonshotai/kimi-k2-instruct",
    "h":"openai/gpt-oss-120b",
    # "i": "deepseek-r1-distill-llama-70b",
    # "j":"meta-llama/llama-4-maverick-17b-128e-instruct",
    # "k":"meta-llama/llama-guard-4-12b"#no tool calling support

       }
react_prompt=hub.pull("hwchase17/react")
# output_parser=PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_instruction=PromptTemplate(template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS+"\n\nIMPORTANT: ALWAYS respond in the following format:\nThought: ...\nAction: ...\nAction Input: ...",
input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions="")

def main():
    for name, model_name in models.items():
        print(f"\n--- Running model {name}: {model_name} ---")
        try:
            llm = ChatGroq(
                temperature=0,
                api_key=api_key,
                model=model_name
            )
            structured_llm=llm.with_structured_output(AgentResponse)
            agent = create_react_agent(
                llm=llm,
                tools=tools,
                prompt=react_prompt_with_instruction
            )

            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )

            extract_output = RunnableLambda(lambda x: x["output"])
            # parse_output = RunnableLambda(lambda x: output_parser.parse(x))

            chain = agent_executor | extract_output | structured_llm

            result = chain.invoke(
                input={
                    "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
                }
            )
            print(result)

        except Exception as e:
            print(f"⚠️ Error running model {name} ({model_name}): {e}")
            print("Skipping to next model...\n")


if __name__ == "__main__":
    main()



