import numpy as np
import torch
from typing import List, Optional
from loguru import logger
from app.core.config import settings


class EmbeddingService:
    """
    Service for generating vector embeddings from text data using Sentence Transformers.
    
    Implements lazy model loading to avoid startup overhead and supports both
    GPU and CPU execution with automatic device detection.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._requested_device = device or settings.DEVICE
        self._model = None
        self._device = None
        self._embedding_dim: Optional[int] = None

    @property
    def device(self) -> str:
        """Returns the resolved device string after model loading."""
        if self._device is None:
            self._resolve_device()
        return self._device

    def _resolve_device(self) -> None:
        """Auto-detect the best available device."""
        requested = self._requested_device.lower()
        if requested == "cuda" and torch.cuda.is_available():
            self._device = "cuda"
        elif requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self._device = "cpu"
        else:
            self._device = "cpu"
        logger.info(f"Resolved compute device: {self._device}")

    def _load_model(self) -> None:
        """Lazily load the SentenceTransformer model on first use."""
        if self._model is not None:
            return

        self._resolve_device()

        logger.info(f"Loading SentenceTransformer model '{self.model_name}' on device '{self._device}'...")

        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self._device)

        # Cache embedding dimension — use newer API name with fallback
        if hasattr(self._model, 'get_embedding_dimension'):
            self._embedding_dim = self._model.get_embedding_dimension()
        else:
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Model loaded successfully. "
            f"Embedding dimension: {self._embedding_dim}, "
            f"Max sequence length: {self._model.max_seq_length}"
        )

    def get_embedding_dimension(self) -> int:
        """Returns the output embedding dimension of the loaded model."""
        self._load_model()
        return self._embedding_dim

    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generates vector embeddings for a batch of text inputs.
        
        Args:
            texts: List of text strings to encode.
            normalize: If True, L2-normalize embeddings (required for cosine similarity).
            show_progress: If True, show sentence-transformers internal progress bar.
            
        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with float32 embeddings.
        """
        self._load_model()

        embeddings = self._model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generates a vector embedding for a single text input.
        
        Returns:
            np.ndarray of shape (embedding_dim,) with float32 values.
        """
        return self.encode_batch([text], normalize=normalize)[0]

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model has been loaded into memory."""
        return self._model is not None
