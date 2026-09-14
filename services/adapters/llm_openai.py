"""
LLM adapter using OpenAI's chat completions API.

Requires OPENAI_API_KEY environment variable.

API reference: POST https://api.openai.com/v1/chat/completions
  body: {"model": "gpt-4o-mini", "messages": [...]}
  response: {"choices": [{"message": {"role": "assistant", "content": "..."}}]}
"""

import os

from services.adapters._http_retry import post_with_retry
from services.interfaces import LLMClient


class OpenAILLMClient(LLMClient):
    """OpenAI LLM client using gpt-4o-mini by default."""

    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate text using OpenAI chat completions.
        
        Args:
            system_prompt: System instruction.
            user_prompt: User query with context.
        
        Returns:
            Generated text response.
        """
        result = post_with_retry(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60,
        )
        return result["choices"][0]["message"]["content"]