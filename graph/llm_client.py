"""
LLM client wrapper (Gemini 3.6 Flash).

A single thin wrapper around the google-genai SDK so every node calls
the same .complete(prompt, system=None) interface, regardless of which
provider sits behind it. Keeping this in one place means swapping
providers/models later only touches this file.

Requires GEMINI_API_KEY in the environment (see .env.example).
"""

import os
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str = MODEL_NAME):
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self._model = model

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Single-turn completion. Returns plain text.

        system is passed as system_instruction rather than folded into
        the user prompt, so node prompt templates (intake_plan.py,
        draft.py, reflect.py) stay simple and provider-agnostic.
        """
        config = types.GenerateContentConfig(system_instruction=system) if system else None

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )

        # .text can be None if the response was blocked/empty (e.g. safety
        # filters) -- fail loudly rather than silently passing None
        # downstream, where nodes assume a string and call .strip()/.lower().
        if response.text is None:
            raise RuntimeError(
                f"Gemini returned no text for this prompt "
                f"(possible safety block or empty response): {response}"
            )

        return response.text


def create_llm_client() -> LLMClient:
    """Factory used by run.py / build_graph.py to construct the shared
    client once per run, then inject it into every node that needs it."""
    return LLMClient()