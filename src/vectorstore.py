"""FAISS vector store wrapper for document storage and retrieval."""
import os
import pickle
import numpy as np
import faiss

class FAISSStore:
    def __init__(self, dimension: int = 384):
        """Initialize a FAISS index for vector similarity search.
        
        Args:
            dimension: Embedding dimension (default 384 to match previous setup)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.documents = []  # store original texts
        self.metadatas = []  # store metadata dicts
        
    def add(self, documents, embeddings, metadatas=None):
        """Add documents and their embeddings to the index.
        
        Args:
            documents: List of text documents
            embeddings: List of embedding vectors (must match dimension)
            metadatas: Optional list of metadata dicts
        """
        if not documents or not embeddings:
            return
        
        vectors = np.array(embeddings).astype('float32')
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected {self.dimension}-dim vectors, got {vectors.shape[1]}")
            
        self.index.add(vectors)
        self.documents.extend(documents)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{} for _ in documents])
    
    def search(self, query_embeddings, n_results=4):
        '''Search for nearest neighbors using query embeddings.
        
        Args:
            query_embeddings: List of query embeddings
            n_results: Number of results to return per query
            
        Returns:
            dict with:
            - documents: List of matching documents per query
            - metadatas: List of metadata dicts per query
            - distances: List of L2 distances per query
        '''
        if not isinstance(query_embeddings, list):
            query_embeddings = [query_embeddings]
            
        vectors = np.array(query_embeddings).astype('float32')
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected {self.dimension}-dim vectors, got {vectors.shape[1]}")
            
        D, I = self.index.search(vectors, n_results)  # distances, indices
        
        results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        for query_idx in range(len(query_embeddings)):
            docs = []
            metas = []
            for idx in I[query_idx]:
                if idx < len(self.documents):  # guard against out of bounds
                    docs.append(self.documents[idx])
                    metas.append(self.metadatas[idx])
            results['documents'].append(docs)
            results['metadatas'].append(metas)
            results['distances'].append(D[query_idx].tolist())
        return results

    
    def save(self, directory):
        """Save the index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "store.faiss"))
        with open(os.path.join(directory, "store.pkl"), "wb") as f:
            pickle.dump({
                "dimension": self.dimension,
                "documents": self.documents,
                "metadatas": self.metadatas
            }, f)
    
    @classmethod
    def load(cls, directory):
        """Load a saved index and metadata from disk."""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        index_path = os.path.join(directory, "store.faiss")
        meta_path = os.path.join(directory, "store.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing store files in {directory}")
            
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            
        store = cls(dimension=data["dimension"])
        store.index = faiss.read_index(index_path)
        store.documents = data["documents"]
        store.metadatas = data["metadatas"]
        return store