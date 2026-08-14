"""Thin wrapper around the local Ollama HTTP API (https://ollama.com).

Ollama must be installed and running locally (`ollama serve`), with the
desired model already pulled (e.g. `ollama pull llama3.1`). No data leaves
the machine — everything is processed by the locally running model.
"""
from typing import Optional

import requests


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Quick check that the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        """Send a single-turn prompt to the model and return the raw text response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not reach Ollama at {self.host}. Is 'ollama serve' running?"
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s") from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Ollama returned an error (is model '{self.model}' pulled? "
                f"try: ollama pull {self.model}): {e}"
            ) from e

        data = resp.json()
        return data.get("response", "").strip()