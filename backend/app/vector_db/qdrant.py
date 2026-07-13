import time
from typing import Any, List, Dict, Optional
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import settings

class QdrantWrapper:
    """
    Wrapper for Qdrant vector database operations.
    Manages client connection, collection creation, embedding ingestion, and search.
    Supports automatic fallback to local persistent storage on disk if connection fails.
    """
    def __init__(self, url: Optional[str] = None, path: Optional[str] = None, collection_name: Optional[str] = None) -> None:
        self.url = url or settings.QDRANT_URL
        self.path = path or settings.QDRANT_PATH
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.client: QdrantClient | None = None
        self._is_local: bool = False

    def connect(self) -> bool:
        """
        Initializes and verifies the connection to the Qdrant server.
        Falls back to a local disk-based client if connection to remote fails.
        Returns:
            bool: True if connection (or local fallback) is successful, False otherwise.
        """
        # Try remote first if URL is set
        if self.url:
            try:
                logger.info(f"Connecting to remote Qdrant at {self.url}...")
                self.client = QdrantClient(url=self.url, timeout=3.0)
                # Test connection
                self.client.get_collections()
                self._is_local = False
                logger.info("Successfully connected to remote Qdrant.")
                return True
            except Exception as e:
                logger.warning(f"Could not connect to remote Qdrant at {self.url}: {e}")
                logger.info("Attempting fallback to local storage...")
        
        # Fallback to local disk persistence
        try:
            storage_path = self.path or "./qdrant_local"
            logger.info(f"Initializing local persistent Qdrant at {storage_path}...")
            self.client = QdrantClient(path=storage_path)
            self._is_local = True
            logger.info("Successfully initialized local persistent Qdrant client.")
            return True
        except Exception as local_err:
            logger.error(f"Failed to initialize local persistent Qdrant: {local_err}")
            self.client = None
            return False

    def collection_exists(self) -> bool:
        """
        Checks if the configured collection exists.
        """
        if not self.client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception as e:
            logger.warning(f"Error checking if collection exists: {e}")
            return False

    def create_collection(self, vector_size: int, distance_metric: str = "Cosine") -> bool:
        """
        Creates a new collection in Qdrant with the specified dimensions and distance metric.
        If it already exists, does nothing to preserve existing data.
        """
        if not self.client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")

        if self.collection_exists():
            logger.info(f"Collection '{self.collection_name}' already exists. Skipping creation.")
            return True

        # Map distance metric string to Qdrant Distance enum
        metric_upper = distance_metric.upper()
        if metric_upper == "COSINE":
            distance = Distance.COSINE
        elif metric_upper == "EUCLID" or metric_upper == "EUCLIDEAN":
            distance = Distance.EUCLID
        elif metric_upper == "DOT" or metric_upper == "DOT_PRODUCT":
            distance = Distance.DOT
        else:
            logger.warning(f"Unknown distance metric '{distance_metric}'. Defaulting to Cosine.")
            distance = Distance.COSINE

        try:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={vector_size}, metric={distance})")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            logger.info(f"Successfully created collection '{self.collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection '{self.collection_name}': {e}")
            return False

    def upload_batch(self, points: List[PointStruct], max_retries: int = 5) -> bool:
        """
        Uploads a batch of PointStruct items to the Qdrant collection with exponential backoff retries.
        """
        if not self.client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")

        if not points:
            return True

        retries = 0
        backoff = 1.0
        while retries < max_retries:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                logger.debug(f"Successfully uploaded batch of {len(points)} points to '{self.collection_name}'.")
                return True
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Failed to upload batch after {max_retries} attempts: {e}")
                    return False
                logger.warning(f"Error uploading batch (attempt {retries}/{max_retries}): {e}. Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2.0

        return False

    def count_points(self) -> int:
        """
        Returns the total number of points in the configured collection.
        """
        if not self.client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")

        try:
            if not self.collection_exists():
                return 0
            response = self.client.count(
                collection_name=self.collection_name,
                exact=True
            )
            return response.count
        except Exception as e:
            logger.error(f"Error counting points in collection '{self.collection_name}': {e}")
            return 0

    def search(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Searches the Qdrant collection for vectors similar to the query_vector.
        """
        if not self.client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")

        try:
            # Use query_points (modern API)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            )
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                for hit in response.points
            ]
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []

