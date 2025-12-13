"""
Multi-Query Retriever
Retrieve for multiple sub-queries and merge results intelligently
"""

from typing import List, Dict, Any
from collections import defaultdict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from retrieval.crag_retriever import CRAGRetriever


class MultiQueryRetriever:
    def __init__(self, crag_retriever: CRAGRetriever):
        """Initialize Multi-Query Retriever"""
        self.retriever = crag_retriever
        print("✅ MultiQueryRetriever initialized")
    
    def retrieve_multi(
        self, 
        sub_queries: List[str],
        top_k_per_query: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieve for multiple sub-queries and merge results
        
        Returns:
            {
                "merged_chunks": Final merged chunks,
                "per_query_results": Results per sub-query,
                "stats": Statistics
            }
        """
        print(f"\n🔍 Multi-Query Retrieval for {len(sub_queries)} quer{'ies' if len(sub_queries) > 1 else 'y'}")
        
        per_query_results = {}
        all_chunks = []
        
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"   [{i}/{len(sub_queries)}] {sub_q}")
            
            result = self.retriever.retrieve(
                sub_q,
                top_k_initial=4,
                top_k_final=top_k_per_query
            )
            
            chunks = result["refined_chunks"]
            per_query_results[sub_q] = chunks
            
            # Tag chunks with source query for tracking
            for chunk in chunks:
                chunk["source_query"] = sub_q
            
            all_chunks.extend(chunks)
            print(f"      → {len(chunks)} chunks")
        
        # Merge and deduplicate
        merged_chunks = self._merge_chunks(all_chunks)
        
        print(f"\n📊 Merge Stats:")
        print(f"   Total retrieved: {len(all_chunks)}")
        print(f"   After merge: {len(merged_chunks)}")
        
        return {
            "merged_chunks": merged_chunks,
            "per_query_results": per_query_results,
            "stats": {
                "total_queries": len(sub_queries),
                "total_retrieved": len(all_chunks),
                "after_merge": len(merged_chunks)
            }
        }
    
    def _merge_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Intelligent chunk merging:
        1. Remove exact duplicates (same chunk_id)
        2. Sort by score (highest first)
        3. Diversity: max 3 chunks per URL
        4. Keep top 10 overall
        """
        if not chunks:
            return []
        
        # Deduplicate by chunk_id
        seen_ids = set()
        deduped = []
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                deduped.append(chunk)
        
        # Sort by score
        deduped.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Ensure diversity - max 3 chunks per URL
        url_counts = defaultdict(int)
        diverse_chunks = []
        
        for chunk in deduped:
            url = chunk.get("url", "unknown")
            
            if url_counts[url] < 3:
                diverse_chunks.append(chunk)
                url_counts[url] += 1
        
       
        return diverse_chunks[:6]