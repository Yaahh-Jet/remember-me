import boto3
import json
import os
from interaction_store import InteractionRecord


class SummaryGenerator:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        self.model_id = "anthropic.claude-haiku-4-5-20251001-v1:0"

    def generate(self, person_id: str, interactions: list[InteractionRecord]) -> str:
        interaction_text = "\n".join(
            f"- {r.timestamp[:10]}: {r.description}"
            for r in interactions
        )

        prompt = f"""You are a compassionate assistant helping a dementia patient remember someone they just saw.

Person name: {person_id}
Recent interactions:
{interaction_text}

Write a warm, calm, reassuring summary in 2-3 sentences that:
- Tells the patient who this person is
- Mentions something personal and comforting from their recent interactions
- Uses simple, clear language

Do not use medical or clinical language. Be warm and gentle."""

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        })

        try:
            response = self.client.invoke_model(modelId=self.model_id, body=body)
            result = json.loads(response["body"].read())
            summary = result["content"][0]["text"]
            print(f"[Bedrock] Summary generated for {person_id}")
            return summary
        except Exception as e:
            raise RuntimeError(f"Bedrock error: {e}")
