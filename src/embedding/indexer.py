"""
Indexer - Read chunks.jsonl and upload to Qdrant
"""

import json
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import hashlib


class QdrantIndexer:
    def __init__(
        self,
        qdrant_path: str = "./qdrant_data",
        collection_name: str = "bdu_chunks_gemma",
        embedding_model: str = "google/embeddinggemma-300m"
    ):
        self.collection_name = collection_name
        
        print(f"📦 Connecting to Qdrant: {qdrant_path}")
        self.client = QdrantClient(path=qdrant_path)
        
        print(f"🔧 Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        print("✅ Model loaded")
    
    def _generate_uuid(self, chunk_id: str) -> str:
        """Generate consistent UUID from chunk_id"""
        hash_obj = hashlib.md5(chunk_id.encode())
        return hash_obj.hexdigest()[:32]
    
    def embed(self, text: str):
        """Embed single text"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def index_jsonl(self, jsonl_path: str, batch_size: int = 100):

        print(f"\n📄 Reading chunks from: {jsonl_path}")
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"📊 Total chunks: {len(lines)}")
        
        total_indexed = 0
        
        for i in tqdm(range(0, len(lines), batch_size), desc="Indexing batches"):
            batch_lines = lines[i:i+batch_size]
            points = []
            
            for line in batch_lines:
                try:
                    item = json.loads(line)
                    
                    content = item.get("content", "")
                    chunk_id = item.get("chunk_id", f"chunk_{total_indexed}")
                    
                    if not content:
                        continue
                    
                    # Embed content (tóm tắt) để tìm kiếm
                    vector = self.embed(content)
                    point_id = self._generate_uuid(chunk_id)
                    
                    # Build payload
                    payload = {
                        "chunk_id": chunk_id,
                        "content": content,
                        "url": item.get("url", "unknown"),
                        "type": item.get("type", "text"),
                    }
                    
                    # Thêm metadata
                    metadata = item.get("metadata", {})
                    
                    if "full_content" in metadata:
                        payload["full_content"] = metadata["full_content"]
                    elif "full_content" in item:  
                        payload["full_content"] = item["full_content"]
                                        
                    if "title" in metadata and metadata["title"]:
                        payload["title"] = metadata["title"]
                    elif "title" in item and item["title"]: 
                        payload["title"] = item["title"]
                    
                    if "order" in metadata:
                        payload["order"] = metadata["order"]
                    
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vector.tolist(),
                            payload=payload
                        )
                    )
                    
                    total_indexed += 1
                    
                except Exception as e:
                    print(f"\n⚠️  Error processing line {i}: {e}")
                    continue
            
            # Upload batch
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
        
        print(f"\n✅ Indexing completed!")
        print(f"   Total indexed: {total_indexed} chunks")
        
        collection_info = self.client.get_collection(self.collection_name)
        print(f"   Qdrant count: {collection_info.points_count} points")


if __name__ == "__main__":
    print("="*60)
    print("QDRANT INDEXER")
    print("="*60)
    
    # Chọn model và collection
    COLLECTION_NAME = "bdu_chunks_gemma"
    MODEL_NAME = "google/embeddinggemma-300m"
    
    indexer = QdrantIndexer(
        qdrant_path="./qdrant_data",
        collection_name=COLLECTION_NAME,
        embedding_model=MODEL_NAME
    )
    
    indexer.index_jsonl("./data/chunks.jsonl", batch_size=100)
    
    print("\n" + "="*60)
    print("🎯 Done!")
    print("="*60)