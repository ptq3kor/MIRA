import os
import json

from dotenv import load_dotenv
from gen_ai_hub.proxy.native.amazon.clients import Session
from gen_ai_hub.proxy.native.openai import OpenAI

load_dotenv()


class AIClient:
    def __init__(self):
        self.openai_client = OpenAI()
        self.amazon_session = Session()

    def ask_openai(self, question, deployment_id, model_name):
        response = self.openai_client.chat.completions.create(
            deployment_id=deployment_id,
            model=model_name,
            messages=[
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content

    def embed_text(self, text, deployment_id, model_name):
        response = self.openai_client.embeddings.create(
            input=text,
            deployment_id=deployment_id,
            model=model_name,
        )
        return response.data[0].embedding

    def ask_claude(self, question, deployment_id, model_name):
        client = self.amazon_session.client(
            deployment_id=deployment_id,
            model_name=model_name,
        )
        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": question}],
        }
        response = client.invoke_model(
            body=json.dumps(request),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["content"][0]["text"]


if __name__ == "__main__":
    prompt = "Count 1 to 10 in English and Tamil."

    ai_client = AIClient()
    openai_deployment_id = os.getenv("AICORE_OPENAI_DEPLOYMENT_ID")
    openai_model_name = os.getenv("AICORE_OPENAI_MODEL")

    openai_response = ai_client.ask_openai(
        prompt,
        openai_deployment_id,
        openai_model_name,
    )

    print("OpenAI Response:", openai_response)

    claude_deployment_id = os.getenv("AICORE_CLUADE_DEPLOYEMENT_ID")
    claude_model_name = os.getenv("AICORE_CLAUDE_MODEL")
    claude_response = ai_client.ask_claude(
        prompt,
        claude_deployment_id,
        claude_model_name,
    )

    print("Claude Response:", claude_response)

    embedding_deployment_id = os.getenv("AICORE_OPENAI_EMBEDDING_DEPLOYMENT_ID")
    embedding_model_name = os.getenv("AICORE_OPENAI_EMBEDDING_MODEL")

    text_to_embed = "Man Women Child Dog Cat"
    embedding = ai_client.embed_text(
           text_to_embed,
            embedding_deployment_id,
            embedding_model_name,
        )
    print("Embedding dimensions:", len(embedding))
