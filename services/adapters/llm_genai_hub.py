"""
LLM adapter using SAP Generative AI Hub (via gen-ai-hub-sdk).

Requires the following environment variables:
- AICORE_AUTH_URL
- AICORE_CLIENT_ID
- AICORE_CLIENT_SECRET
- AICORE_BASE_URL
- AICORE_RESOURCE_GROUP
- AICORE_OPENAI_DEPLOYMENT_ID (for OpenAI models)
- AICORE_OPENAI_MODEL
- AICORE_CLAUDE_DEPLOYMENT_ID (optional, for Claude)
- AICORE_CLAUDE_MODEL (optional)

The gen-ai-hub-sdk handles authentication and token refresh automatically.
"""

import os
from typing import Optional

from services.interfaces import LLMClient

# Lazy imports keep the optional Claude dependency separate from OpenAI.
try:
    from gen_ai_hub.proxy.native.openai import OpenAI
    from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
    GENAI_HUB_AVAILABLE = True
except ImportError:
    GENAI_HUB_AVAILABLE = False

try:
    from gen_ai_hub.proxy.native.amazon.clients import Session
    AMAZON_CLIENT_AVAILABLE = True
except ImportError:
    AMAZON_CLIENT_AVAILABLE = False


class GenAIHubLLMClient(LLMClient):
    """SAP Generative AI Hub LLM client supporting OpenAI and Claude models."""

    def __init__(self, model_provider: str = "openai"):
        """
        Initialize the GenAI Hub LLM client.
        
        Args:
            model_provider: "openai" or "claude" - which model to use.
        """
        if not GENAI_HUB_AVAILABLE:
            raise RuntimeError(
                "gen-ai-hub-sdk not installed. Run: pip install gen-ai-hub-sdk"
            )
        
        self.model_provider = model_provider
        self.proxy_client = get_proxy_client('gen-ai-hub')
        
        if model_provider == "openai":
            self.deployment_id = os.environ["AICORE_OPENAI_DEPLOYMENT_ID"]
            self.model = os.environ["AICORE_OPENAI_MODEL"]
            self.openai_client = OpenAI(proxy_client=self.proxy_client)
        elif model_provider == "claude":
            if not AMAZON_CLIENT_AVAILABLE:
                raise RuntimeError(
                    "Claude support requires the Amazon extras for the SAP AI SDK."
                )
            self.deployment_id = os.environ["AICORE_CLAUDE_DEPLOYMENT_ID"]
            self.model = os.environ["AICORE_CLAUDE_MODEL"]
            self.amazon_session = Session(proxy_client=self.proxy_client)
        else:
            raise ValueError(f"Unknown model_provider: {model_provider}. Use 'openai' or 'claude'")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate text using the configured model via SAP AI Core.
        
        Args:
            system_prompt: System instruction.
            user_prompt: User query with context.
        
        Returns:
            Generated text response.
        """
        if self.model_provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)
        else:
            return self._generate_claude(system_prompt, user_prompt)

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using OpenAI model via GenAI Hub."""
        response = self.openai_client.chat.completions.create(
            deployment_id=self.deployment_id,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def _generate_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using Claude model via GenAI Hub (Amazon Bedrock)."""
        import json
        
        client = self.amazon_session.client(
            deployment_id=self.deployment_id,
            model_name=self.model,
        )
        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        response = client.invoke_model(
            body=json.dumps(request),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["content"][0]["text"]