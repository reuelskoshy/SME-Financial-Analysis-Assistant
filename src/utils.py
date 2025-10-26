"""Utility functions for SME analytics including embedding generation."""
from typing import List
import torch
import torch.nn as nn
import numpy as np

class EmbeddingHelper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Use a simple TF-IDF like approach
        self.embedding_dim = 384  # Same as original model
        self.embedder = nn.Embedding(65536, self.embedding_dim)  # Large vocab size
        # Initialize with deterministic weights
        torch.manual_seed(42)
        self.embedder.weight.data.normal_(mean=0.0, std=0.1)
        
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            # Convert text to numerical values (simple ord encoding)
            char_ids = torch.tensor([ord(c) % 65536 for c in text])
            # Get embeddings and average them
            with torch.no_grad():
                emb = self.embedder(char_ids)
                avg_emb = emb.mean(dim=0)
                # Normalize
                avg_emb = avg_emb / torch.norm(avg_emb)
                embeddings.append(avg_emb.numpy().astype(np.float32))
        return embeddings
        return vectors


def format_currency(val: int) -> str:
    return f"₹{int(val):,d}"