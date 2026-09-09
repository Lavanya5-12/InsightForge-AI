import requests
from typing import List, Dict, Tuple, Optional
from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_URL,
    OLLAMA_GENERATE_URL,
    OLLAMA_TAGS_URL,
    EMBEDDING_MODEL,
    LLM_MODEL
)


class OllamaError(Exception):
    """Base exception for Ollama interactions."""
    pass


class OllamaUnavailableError(OllamaError):
    """Raised when Ollama server is unreachable."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when a requested Ollama model is not pulled/installed."""
    pass


class OllamaClient:
    """
    Centralized client for interacting with local Ollama service.
    Handles embedding generation, LLM text generation, and status checks.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL
    ):
        self.base_url = base_url.rstrip("/")
        self.embed_url = f"{self.base_url}/api/embed"
        self.generate_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    def check_health(self) -> Tuple[bool, str]:
        """
        Check if Ollama service is running and reachable.
        """
        try:
            res = requests.get(self.tags_url, timeout=5)
            if res.status_code == 200:
                return True, "Ollama service is online."
            return False, f"Ollama returned status code {res.status_code}."
        except requests.exceptions.RequestException as e:
            return False, f"Ollama service unavailable at {self.base_url}. Please ensure Ollama is running. Error: {str(e)}"

    def check_model_available(self, model_name: str) -> Tuple[bool, str]:
        """
        Check if a specific model is available in local Ollama instance.
        """
        try:
            res = requests.get(self.tags_url, timeout=5)
            if res.status_code == 200:
                models_data = res.json().get("models", [])
                available_names = [m.get("name", "") for m in models_data]
                # Normalize names (e.g., llama3.2:3b vs llama3.2:3b-latest)
                for name in available_names:
                    if name == model_name or name.startswith(model_name):
                        return True, f"Model '{model_name}' is available."
                return False, f"Model '{model_name}' not found. Run 'ollama pull {model_name}' to install it."
            return False, f"Unable to list Ollama models. HTTP {res.status_code}."
        except requests.exceptions.RequestException as e:
            return False, f"Could not check model availability: {str(e)}"

    def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        timeout: int = 120
    ) -> List[List[float]]:
        """
        Generate vector embeddings for a list of strings.
        """
        if not texts:
            return []

        target_model = model or self.embedding_model

        try:
            response = requests.post(
                self.embed_url,
                json={
                    "model": target_model,
                    "input": texts
                },
                timeout=timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise OllamaUnavailableError(
                f"Failed to connect to Ollama at {self.embed_url}. Is Ollama running? Error: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise OllamaError(
                f"Embedding request to Ollama timed out after {timeout} seconds."
            ) from e
        except requests.exceptions.RequestException as e:
            raise OllamaError(
                f"Ollama request error during embedding generation: {e}"
            ) from e

        if response.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Embedding model '{target_model}' not found in Ollama. Run 'ollama pull {target_model}'."
            )

        if response.status_code != 200:
            raise OllamaError(
                f"Ollama embedding API error (status {response.status_code}): {response.text}"
            )

        data = response.json()
        embeddings = data.get("embeddings", [])

        if len(embeddings) != len(texts):
            raise OllamaError(
                f"Expected {len(texts)} embeddings, but received {len(embeddings)} from Ollama."
            )

        return embeddings

    def generate_answer(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 120
    ) -> str:
        """
        Generate text response from Ollama LLM.
        """
        target_model = model or self.llm_model

        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": target_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                },
                timeout=timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise OllamaUnavailableError(
                f"Failed to connect to Ollama at {self.generate_url}. Is Ollama running? Error: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise OllamaError(
                f"Generation request to Ollama timed out after {timeout} seconds."
            ) from e
        except requests.exceptions.RequestException as e:
            raise OllamaError(
                f"Ollama request error during text generation: {e}"
            ) from e

        if response.status_code == 404:
            raise OllamaModelNotFoundError(
                f"LLM model '{target_model}' not found in Ollama. Run 'ollama pull {target_model}'."
            )

        if response.status_code != 200:
            raise OllamaError(
                f"Ollama generation API error (status {response.status_code}): {response.text}"
            )

        data = response.json()
        return data.get("response", "").strip()


# Default singleton instance
default_ollama_client = OllamaClient()
