"""
Qdrant Vector Database Setup
Embedded mode - No Docker required
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os

class QdrantSetup:
    def __init__(self, persist_path: str = "./qdrant_data"):
        """
        Initialize Qdrant client
        
        Args:
            persist_path: Directory to store vector data (None = in-memory only)
        """
        self.persist_path = persist_path
        
        # Create client
        if persist_path:
            os.makedirs(persist_path, exist_ok=True)
            self.client = QdrantClient(path=persist_path)
            print(f"✅ Qdrant initialized with persistence: {persist_path}")
        else:
            self.client = QdrantClient(":memory:")
            print("✅ Qdrant initialized (in-memory mode)")
    
    def create_collection(
        self, 
        collection_name: str = "bdu_chunks_gemma",
        vector_size: int = 768, 
        distance_metric: str = "Cosine"
    ):
        """
        Create vector collection
        
        Args:
            collection_name: Name of the collection
            vector_size: Embedding dimension (768 for vietnamese-bi-encoder)
            distance_metric: "Cosine", "Euclid", or "Dot"
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            print(f"⚠️  Collection '{collection_name}' already exists")
            return
        
        # Map distance metric
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT
        }
        
        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map[distance_metric]
            )
        )
        
        print(f"✅ Collection '{collection_name}' created successfully")
        print(f"   - Vector size: {vector_size}")
        print(f"   - Distance metric: {distance_metric}")
    
    def get_collection_info(self, collection_name: str = "bdu_chunks_gemma"):
        """Get collection statistics"""
        try:
            info = self.client.get_collection(collection_name)
            print(f"\n📊 Collection: {collection_name}")
            print(f"   - Vectors count: {info.points_count}")
            print(f"   - Vector size: {info.config.params.vectors.size}")
            print(f"   - Distance: {info.config.params.vectors.distance}")
            return info
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def delete_collection(self, collection_name: str):
        """Delete collection (careful!)"""
        try:
            self.client.delete_collection(collection_name)
            print(f"✅ Deleted collection: {collection_name}")
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
    
    def list_collections(self):
        """List all collections"""
        collections = self.client.get_collections().collections
        print(f"\n📚 Available collections ({len(collections)}):")
        for c in collections:
            print(f"   - {c.name}")
        return collections


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("QDRANT VECTOR DATABASE SETUP")
    print("="*60)
    
    # 1. Initialize Qdrant
    qdrant = QdrantSetup(persist_path="./qdrant_data")
    
    qdrant.delete_collection("bdu_chunks_gemma")

    # 2. Create collection for BDU chatbot
    qdrant.create_collection(
        collection_name="bdu_chunks_gemma",
        vector_size=768, 
        distance_metric="Cosine"
    )
    
    # 3. List collections
    qdrant.list_collections()
    
    # 4. Get collection info
    qdrant.get_collection_info("bdu_chunks_gemma")
    
    print("\n" + "="*60)
    print("✅ Setup completed!")
    print("="*60)
