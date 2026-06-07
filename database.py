import os
import numpy as np
import faiss
import pandas as pd
import config

class VectorDatabase:
    def __init__(self, dimension=2048):
        """
        Initializes the FAISS index. 
        ResNet-50 outputs vectors of size 2048.
        """
        self.dimension = dimension
        # IndexFlatL2 measures similarity using Euclidean distance
        self.index = faiss.IndexFlatL2(self.dimension)
        self.image_paths = []

    def add_vectors(self, vectors, paths):
        """Adds a batch of vectors and their corresponding image paths."""
        if len(vectors) == 0:
            return
            
        # Convert list/array to float32 numpy array (required by FAISS)
        vectors_np = np.array(vectors).astype('float32')
        self.index.add(vectors_np)
        self.image_paths.extend(paths)

    def save(self):
        """Saves the index and metadata to disk."""
        faiss.write_index(self.index, config.INDEX_PATH)
        # Save mapping of vector IDs to actual image file paths
        df = pd.DataFrame({"image_path": self.image_paths})
        df.to_csv(config.METADATA_PATH, index=False)
        print(f"Database saved! Index: {config.INDEX_PATH}, Metadata: {config.METADATA_PATH}")

    def load(self):
        """Loads the index and metadata from disk."""
        if os.path.exists(config.INDEX_PATH) and os.path.exists(config.METADATA_PATH):
            self.index = faiss.read_index(config.INDEX_PATH)
            self.image_paths = pd.read_csv(config.METADATA_PATH)["image_path"].tolist()
            print("Database loaded successfully!")
            return True
        else:
            print("Database files not found. You need to build the index first.")
            return False

    def search(self, query_vector, top_k=5):
        """Searches for the top_k most similar images given a query vector."""
        if self.index.ntotal == 0:
            print("Database is empty!")
            return []

        # Format query vector for FAISS
        query_np = np.array([query_vector]).astype('float32')
        
        # D = Distances, I = Indices of the closest matches
        D, I = self.index.search(query_np, top_k)
        
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.image_paths):
                results.append(self.image_paths[idx])
                
        return results

if __name__ == "__main__":
    print("Initializing Database Structure...")
    db = VectorDatabase()
    print(f"Ready to accept vectors with dimension size: {db.dimension}")
