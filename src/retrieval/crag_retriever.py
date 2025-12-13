from typing import List, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from .relevance_evaluator import RelevanceEvaluator  # Revert: Cross-Encoder không hỗ trợ tốt tiếng Việt
from .web_search_corrector import WebSearchCorrector
from Advanced_Query.query_expander import QueryExpander


class CRAGRetriever:
    """True CRAG Retriever with Optimized Lazy Query Expansion"""
    
    def __init__(
        self,
        qdrant_path: str = "./qdrant_data",
        collection_name: str = "bdu_chunks_gemma",
        embedding_model: str = "google/embeddinggemma-300m",
        relevance_threshold: float = 0.6,
        min_correct_threshold: int = 2,
        preloaded_model: SentenceTransformer = None
    ):
        self.collection_name = collection_name
        self.relevance_threshold = relevance_threshold
        self.min_correct_threshold = min_correct_threshold
        
        print(f"📦 Connecting to Qdrant: {qdrant_path}")
        self.client = QdrantClient(path=qdrant_path)
        
        if preloaded_model:
            print("✅ Using preloaded embedding model")
            self.model = preloaded_model
        else:
            print(f"🔧 Loading embedding model: {embedding_model}")
            try:
                self.model = SentenceTransformer(embedding_model)
            except Exception as e:
                print(f"⚠️ Connection failed ({str(e)[:50]}...). Trying offline mode...")
                # Thử load offline từ cache
                self.model = SentenceTransformer(embedding_model, local_files_only=True)
        
        # Initialize Components
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY required for CRAG")
        
        # Revert về LLM-based evaluator: Cross-Encoder ms-marco không hỗ trợ tốt tiếng Việt
        llm_client = Groq(api_key=groq_api_key)
        self.evaluator = RelevanceEvaluator(llm_client)
        self.web_corrector = WebSearchCorrector()
        
        # Initialize Query Expander với model đã load
        # ✅ FIX: Pass embedding model để tránh load lại
        self.expander = QueryExpander(
            groq_api_key=groq_api_key,
            embedding_model=self.model  # Dùng chung model
        )
        
        print("✅ True CRAG Retriever ready (Optimized Lazy Expansion mode)")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed query with normalization"""
        normalized_query = query.strip().lower()
        
        # Chuẩn hóa thời gian
        time_replacements = {
            "năm nay": "năm 2025",
            "hiện nay": "năm 2025", 
            "hiện tại": "năm 2025"
        }
        
        for old, new in time_replacements.items():
            pattern = rf'\b{re.escape(old)}\b'
            normalized_query = re.sub(pattern, new, normalized_query, flags=re.IGNORECASE)
        
        return self.model.encode(normalized_query, convert_to_numpy=True)
    
    def semantic_search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Semantic search in Qdrant"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        candidates = []
        for hit in results:
            content = hit.payload.get("content", "")  
            title = hit.payload.get("title")
            if not title:
                # Fallback 1: Lấy từ chunk_id
                title = hit.payload.get("chunk_id", "").replace("-", " ").title()            
            if not title:
                title = "Tài liệu tuyển sinh"            
            candidates.append({
                "id": hit.id,
                "score": hit.score,
                "chunk_id": hit.payload.get("chunk_id"),
                "content": content,
                "full_content": hit.payload.get("full_content"),
                "url": hit.payload.get("url"),
                "type": hit.payload.get("type"),
                "title": title,  # ✅ Đã có fallback
                "order": hit.payload.get("order"),
                "source": "database"
            })
        
        return candidates
    
    def evaluate_relevance(self, query: str, candidates: List[Dict]) -> Dict[str, List[Dict]]:
        """LLM-based relevance evaluation (Reverted from Cross-Encoder)"""
        print(f"[CRAG] Evaluating {len(candidates)} candidates...")
        
        labels = self.evaluator.evaluate_batch(query, candidates)
        
        graded = {
            "correct": [],
            "incorrect": [],
            "ambiguous": []
        }
        
        for doc, label in zip(candidates, labels):
            graded[label.lower()].append(doc)
        
        print(f"[CRAG] Evaluation results:")
        print(f"   ✅ CORRECT: {len(graded['correct'])}")
        print(f"   ⚠️  AMBIGUOUS: {len(graded['ambiguous'])}")
        print(f"   ❌ INCORRECT: {len(graded['incorrect'])}")
        
        return graded
    
    def needs_expansion(self, graded: Dict[str, List[Dict]]) -> bool:
        """
        Quyết định có cần Query Expansion không
        
        Returns True nếu:
        - Không có CORRECT chunks hoặc
        - Có ít hơn min_correct_threshold CORRECT chunks
        """
        correct_count = len(graded["correct"])
        return correct_count < self.min_correct_threshold
    
    def decide_action(self, graded: Dict[str, List[Dict]]) -> str:
        """Decide corrective action"""
        correct_count = len(graded["correct"])
        ambiguous_count = len(graded["ambiguous"])
        
        if correct_count == 0 and ambiguous_count == 0:
            return "WEB_SEARCH"
        
        if correct_count >= 2:
            return "KNOWLEDGE_REFINEMENT"
        
        return "HYBRID"
    
    def apply_correction(
        self, 
        query: str,
        graded: Dict[str, List[Dict]],
        action: str
    ) -> List[Dict]:
        """Apply corrective action"""
        print(f"[CRAG] Action: {action}")
        
        if action == "WEB_SEARCH":
            web_results = self.web_corrector.search(query, max_results=3)
            print(f"[CRAG] Using {len(web_results)} web search results")
            return web_results
        
        elif action == "KNOWLEDGE_REFINEMENT":
            refined = graded["correct"][:5]
            print(f"[CRAG] Using {len(refined)} correct documents")
            return refined
        
        else:  # HYBRID
            internal = graded["correct"] + graded["ambiguous"]
            internal = sorted(internal, key=lambda x: x["score"], reverse=True)[:3]
            
            web_results = self.web_corrector.search(query, max_results=2)
            
            combined = internal + web_results
            print(f"[CRAG] Hybrid: {len(internal)} internal + {len(web_results)} web")
            return combined
    
    def retrieve(
        self, 
        query: str, 
        top_k_initial: int = 4,
        top_k_final: int = 2
    ) -> Dict[str, Any]:
        """
        Main CRAG pipeline with Optimized Lazy Query Expansion
        
        Flow:
        1. Initial Retrieval (no expansion)
        2. Evaluate
        3. If insufficient CORRECT → Query Expansion → Retrieve ONLY variations
        4. Final refinement
        """
        # === PHASE 1: INITIAL RETRIEVAL (NO EXPANSION) ===
        print("[CRAG] Phase 1: Initial retrieval...")
        
        query_vector = self.embed_query(query)
        initial_candidates = self.semantic_search(query_vector, top_k=top_k_initial)
        
        if len(initial_candidates) == 0:
            print("[CRAG] No candidates found")
            return {
                "query": query,
                "refined_chunks": [],
                "graded_stats": {"correct": 0, "incorrect": 0, "ambiguous": 0},
                "action_taken": "NONE",
                "expansion_triggered": False
            }
        
        # === PHASE 2: EVALUATE INITIAL RESULTS ===
        graded = self.evaluate_relevance(query, initial_candidates)
        
        # === PHASE 3: OPTIMIZED LAZY EXPANSION ===
        expansion_triggered = False
        
        if self.needs_expansion(graded):
            print(f"\n[CRAG] ⚠️  Insufficient CORRECT chunks ({len(graded['correct'])} < {self.min_correct_threshold})")
            print("[CRAG] 🔄 Triggering Query Expansion...")
            
            expansion_triggered = True
            
            # ✅ FIX: Chỉ lấy variations, KHÔNG bao gồm original query
            expanded_queries = self.expander.expand(
                query, 
                num_variations=2,
                include_original=False  # ← KEY FIX: Không lấy original
            )
            
            # Track chunks đã có để tránh duplicate
            expansion_candidates = []
            seen_ids = set(c["chunk_id"] for c in initial_candidates)
            
            # ✅ PARALLEL PROCESSING: Embed và search song song cho các expanded queries
            def search_expanded_query(exp_q: str) -> List[Dict]:
                """Thread-safe function để xử lý một expanded query"""
                exp_vector = self.embed_query(exp_q)
                return self.semantic_search(exp_vector, top_k=top_k_initial)
            
            print(f"[CRAG] 🚀 Parallel expansion with {len(expanded_queries)} queries...")
            with ThreadPoolExecutor(max_workers=max(1, min(len(expanded_queries), 3))) as executor:
                # Submit all queries at once
                future_to_query = {
                    executor.submit(search_expanded_query, eq): eq 
                    for eq in expanded_queries
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_query):
                    exp_q = future_to_query[future]
                    try:
                        exp_results = future.result()
                        print(f"[CRAG]    ✓ Expanded: {exp_q[:50]}... ({len(exp_results)} results)")
                        
                        # Only add new chunks
                        for cand in exp_results:
                            cand_id = cand.get("chunk_id")
                            if cand_id and cand_id not in seen_ids:
                                seen_ids.add(cand_id)
                                expansion_candidates.append(cand)
                    except Exception as e:
                        print(f"[CRAG]    ✗ Expansion error for '{exp_q[:30]}...': {e}")
            
            print(f"[CRAG] Found {len(expansion_candidates)} new chunks via parallel expansion")
            
            # Merge và re-evaluate ALL candidates
            all_candidates = initial_candidates + expansion_candidates
            graded = self.evaluate_relevance(query, all_candidates)
        
        else:
            print(f"[CRAG] ✅ Sufficient CORRECT chunks ({len(graded['correct'])}), no expansion needed")
        
        # === PHASE 4: DECIDE ACTION & REFINE ===
        action = self.decide_action(graded)
        refined_chunks = self.apply_correction(query, graded, action)
        refined_chunks = refined_chunks[:top_k_final]
        
        return {
            "query": query,
            "refined_chunks": refined_chunks,
            "graded_stats": {
                "correct": len(graded["correct"]),
                "incorrect": len(graded["incorrect"]),
                "ambiguous": len(graded["ambiguous"])
            },
            "action_taken": action,
            "expansion_triggered": expansion_triggered
        }
    
    def close(self):
        """Close connections"""
        if hasattr(self.client, 'close'):
            self.client.close()