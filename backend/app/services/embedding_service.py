import numpy as np
import threading
from collections import OrderedDict
from typing import List, Optional, Any
from loguru import logger
from app.core.config import settings
from app.core.model_manager import ModelManager


class EmbeddingCache:
    """Thread-safe LRU cache for query embeddings."""
    def __init__(self, maxsize: int = 1000) -> None:
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key: str, value: np.ndarray) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)  # Pop oldest (LRU)


class EmbeddingService:
    """
    Service for generating vector embeddings from text data using Sentence Transformers.
    Retrieves the resident, preloaded model from the ModelManager singleton.
    Uses a thread-safe LRU cache to reuse query embeddings.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._cache = EmbeddingCache(maxsize=1000)
        self._local_is_loaded = False
        
        # Reset ModelManager if running in unit tests with a mock model name
        if model_name == "mock-model" or (model_name and "mock" in model_name):
            logger.info("Mock model name detected. Resetting ModelManager for test environment.")
            ModelManager.reset()
            
        logger.info("EmbeddingService initialized with shared ModelManager client and query cache.")

    @property
    def device(self) -> str:
        """Returns the resolved compute device from the ModelManager."""
        try:
            if not ModelManager._is_initialized:
                ModelManager.load_model()
            return ModelManager.get_device()
        except RuntimeError:
            return settings.DEVICE

    def get_embedding_dimension(self) -> int:
        """Returns the output embedding dimension of the loaded model."""
        if settings.EMBEDDING_MODEL == "all-MiniLM-L6-v2" or "MiniLM" in settings.EMBEDDING_MODEL:
            return 384
        if "mock" in settings.EMBEDDING_MODEL:
            return 768
            
        try:
            if not ModelManager._is_initialized:
                ModelManager.load_model()
            model = ModelManager.get_model()
            if hasattr(model, 'get_embedding_dimension'):
                return model.get_embedding_dimension()
            return model.get_sentence_embedding_dimension()
        except Exception:
            return 768

    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generates vector embeddings for a batch of text inputs using the resident model.
        
        Args:
            texts: List of text strings to encode.
            normalize: If True, L2-normalize embeddings (required for cosine similarity).
            show_progress: If True, show SentenceTransformer progress bar.
            
        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with float32 embeddings.
        """
        if not ModelManager._is_initialized:
            logger.info("ModelManager not initialized. Initializing lazily for current context...")
            ModelManager.load_model()

        model = ModelManager.get_model()
        self._local_is_loaded = True

        embeddings = model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generates a vector embedding for a single text input, utilizing the LRU cache.
        
        Returns:
            np.ndarray of shape (embedding_dim,) with float32 values.
        """
        cache_key = f"{text}_{normalize}"
        cached_val = self._cache.get(cache_key)
        if cached_val is not None:
            logger.debug(f"[Embedding Cache] HIT for text: '{text[:40]}...'")
            return cached_val

        logger.debug(f"[Embedding Cache] MISS for text: '{text[:40]}...'. Computing embedding...")
        embedding = self.encode_batch([text], normalize=normalize)[0]
        self._cache.set(cache_key, embedding)
        return embedding

    @property
    def is_loaded(self) -> bool:
        """Returns True if this specific service has triggered a load operation."""
        return self._local_is_loaded
