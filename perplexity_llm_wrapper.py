from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from openai import OpenAI
from pydantic import Field

class PerplexityLLM(LLM):
    """LangChain wrapper for Perplexity API"""
    
    client: OpenAI = Field(default=None)
    model_name: str = "sonar-pro"
    
    def __init__(self, api_key: str, model_name: str = "sonar-pro", **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "perplexity"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return resp.choices[0].message.content  # just string

    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model_name}
