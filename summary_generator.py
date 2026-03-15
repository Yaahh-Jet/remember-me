import os
from groq import Groq
from interaction_store import InteractionRecord
from dotenv import load_dotenv

load_dotenv()


class SummaryGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)

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

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            summary = response.choices[0].message.content
            print(f"[Groq] Summary generated for {person_id}")
            return summary
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}")