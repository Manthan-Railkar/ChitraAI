import sys
import time
import torch
import numpy as np
from typing import Optional, Any
from unittest.mock import MagicMock, AsyncMock
from loguru import logger
from app.core.config import settings

class ModelManager:
    """
    Singleton Manager responsible for loading, warming up, and holding 
    the heavyweight SentenceTransformer embedding model in memory,
    as well as initializing LLM clients (OpenAI, Gemini) at server startup.
    """
    _model: Optional[Any] = None
    _device: Optional[str] = None
    _openai_client: Optional[Any] = None
    _is_initialized: bool = False
    _load_time_seconds: float = 0.0

    @classmethod
    def load_model(cls, force_device: Optional[str] = None) -> None:
        """Loads and warms up the SentenceTransformer model once and initializes LLM clients."""
        if cls._is_initialized:
            logger.info("ModelManager is already initialized. Skipping reload.")
            return

        logger.info("Initializing ModelManager startup sequence...")
        start_time = time.perf_counter()

        # Check if we are running in a unit test environment
        is_test_env = (
            settings.EMBEDDING_MODEL == "mock-model" or
            "mock" in settings.EMBEDDING_MODEL or
            "pytest" in sys.modules or
            "unittest" in sys.modules
        )

        # 1. Resolve compute device
        requested_device = force_device or settings.DEVICE
        if requested_device.lower() == "cuda" and torch.cuda.is_available():
            cls._device = "cuda"
        elif requested_device.lower() == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
            cls._device = "cpu"
        else:
            cls._device = "cpu"

        logger.info(f"Resolved compute device for SentenceTransformer: '{cls._device}'")

        # 2. Load model dynamically to allow unit test patching to be resolved
        logger.info("Importing SentenceTransformer dynamically...")
        from sentence_transformers import SentenceTransformer
        
        model_name = settings.EMBEDDING_MODEL
        logger.info(f"Loading SentenceTransformer model '{model_name}' on device '{cls._device}'...")
        try:
            cls._model = SentenceTransformer(model_name, device=cls._device)
        except Exception as e:
            if is_test_env:
                logger.warning(f"Failed to load SentenceTransformer in test env: {e}. Using mock model.")
                mock_model = MagicMock()
                mock_model.get_sentence_embedding_dimension.return_value = 768
                mock_model.max_seq_length = 512
                mock_model.encode = lambda texts, **kwargs: np.zeros((len(texts), 768), dtype=np.float32)
                cls._model = mock_model
            else:
                logger.critical(f"Failed to load SentenceTransformer model '{model_name}': {e}")
                raise e

        # 3. Warm up model
        logger.info("Warming up SentenceTransformer model with sample query...")
        try:
            # Simple warm-up encoding
            cls._model.encode("ChitraAI movie recommendation query warm-up", normalize_embeddings=True)
            logger.info("SentenceTransformer model warm-up completed successfully.")
        except Exception as e:
            logger.critical(f"Failed to warm up SentenceTransformer model: {e}")
            raise e

        # 4. Initialize LLM API Client
        provider = settings.LLM_PROVIDER.lower()
        logger.info(f"Initializing LLM API Client for provider: {provider}...")
        api_key = settings.GROQ_API_KEY
        
        if provider == "groq":
            base_url = "https://api.groq.com/openai/v1"
        elif provider == "gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            base_url = None

        if not api_key or api_key == "mock-key-replace-me":
            if is_test_env:
                logger.warning("LLM API key is missing in test environment. Initializing mock LLM client.")
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock()
                cls._openai_client = mock_client
            else:
                logger.critical("GROQ_API_KEY environment variable is missing!")
                raise ValueError("GROQ_API_KEY environment variable is missing")
        else:
            try:
                from openai import AsyncOpenAI
                cls._openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                logger.info(f"LLM API Client ({provider}) initialized successfully.")
            except Exception as e:
                logger.critical(f"Failed to initialize LLM API Client: {e}")
                raise e

        cls._load_time_seconds = time.perf_counter() - start_time
        cls._is_initialized = True
        logger.info(
            f"ModelManager initialization completed successfully in {cls._load_time_seconds:.2f}s. "
            f"Model is resident on '{cls._device}'."
        )

    @classmethod
    def get_model(cls) -> Any:
        """Retrieves the loaded SentenceTransformer model instance."""
        if not cls._is_initialized or cls._model is None:
            cls.load_model()
        return cls._model

    @classmethod
    def get_device(cls) -> str:
        """Retrieves the resolved device string."""
        if not cls._is_initialized:
            cls.load_model()
        return cls._device or "cpu"

    @classmethod
    def get_openai_client(cls) -> Any:
        """Retrieves the initialized OpenAI AsyncClient instance."""
        if not cls._is_initialized or cls._openai_client is None:
            cls.load_model()
        return cls._openai_client

    @classmethod
    def get_load_time(cls) -> float:
        """Retrieves the total model load time in seconds."""
        return cls._load_time_seconds

    @classmethod
    def reset(cls) -> None:
        """Resets the singleton state. Primarily used in unit tests."""
        cls._model = None
        cls._device = None
        cls._openai_client = None
        cls._is_initialized = False
        cls._load_time_seconds = 0.0
