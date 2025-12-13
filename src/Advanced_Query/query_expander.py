
from typing import List, Dict
import re
import json
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()


class QueryExpander:
    """LLM-based query expansion with similarity filtering"""
    
    def __init__(
        self, 
        groq_api_key: str = None,
        embedding_model: SentenceTransformer = None
    ):
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"
        
        # Embedding model để filter variations
        self.embed_model = embedding_model
        
        print("✅ QueryExpander initialized")
    
    def expand(
        self, 
        query: str, 
        num_variations: int = 2,
        use_filtering: bool = True,
        include_original: bool = True
        
        
    ) -> List[str]:      
        # Short/simple queries don't need expansion
        if len(query.split()) <= 3:
            print(f"[Expander] Query too short, no expansion")
            return [query] if include_original else []
        
        try:
            # Generate variations với LLM
            raw_variations = self._llm_expand(query, num_variations)
            
            if not raw_variations:
                return [query] if include_original else []
            
            # Filter by similarity (nếu có embedding model)
            if use_filtering and self.embed_model:
                variations = self._filter_by_similarity(query, raw_variations, top_k=num_variations)
            else:
                variations = raw_variations[:num_variations]
            
            # ✅ FIX: Tuỳ theo include_original
            if include_original:
                final_queries = [query] + variations
                print(f"[Expander] ✅ Expanded into {len(final_queries)} queries (with original)")
            else:
                final_queries = variations
                print(f"[Expander] ✅ Generated {len(final_queries)} variations (without original)")
            
            for i, q in enumerate(final_queries):
                prefix = "[Original]" if (include_original and i == 0) else f"[Var {i}]"
                print(f"   {prefix} {q}")
            
            return final_queries
            
        except Exception as e:
            print(f"[Expander] ❌ Error: {e}, using original only")
            return [query] if include_original else []
    
    def _llm_expand(self, query: str, num_variations: int) -> List[str]:
        """Generate query variations using LLM"""
        
        prompt = f"""Bạn là hệ thống tạo biến thể câu hỏi để cải thiện tìm kiếm.

NHIỆM VỤ: Tạo {num_variations} cách hỏi KHÁC NHAU cho cùng 1 ý nghĩa.

QUY TẮC:
1. Giữ NGUYÊN Ý NGHĨA gốc
2. Thay đổi CẤU TRÚC CÂU và TỪ NGỮ
3. Dùng TỪ ĐỒNG NGHĨA, PARAPHRASE
4. KHÔNG thêm/bớt thông tin
5. Mỗi biến thể phải KHÁC BIỆT rõ ràng

VÍ DỤ:
Query: "Học phí ngành CNTT là bao nhiêu?"
Variations:
1. "Chi phí học tập ngành Công nghệ thông tin?"
2. "Mức thu học phí chuyên ngành IT?"

Query: "Điều kiện xét tuyển ngành Logistics?"
Variations:
1. "Yêu cầu để vào ngành Quản lý chuỗi cung ứng?"
2. "Tiêu chuẩn tuyển sinh chuyên ngành Logistics?"

---

CÂU HỎI GỐC:
"{query}"

Trả về JSON array với {num_variations} biến thể:"""

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.7,  # Cao hơn để có diversity
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        content = re.sub(r'```json\s*|\s*```', '', content)
        
        try:
            variations = json.loads(content)
            
            if not isinstance(variations, list):
                return []
            
            # Clean và validate
            variations = [
                v.strip() 
                for v in variations 
                if isinstance(v, str) and v.strip() and v.strip() != query
            ]
            
            return variations
            
        except json.JSONDecodeError:
            # Fallback: extract from numbered list
            matches = re.findall(r'[0-9]+\.\s*"?([^"\n]+)"?', content)
            return [m.strip() for m in matches if m.strip() != query]
    
    def _filter_by_similarity(
        self, 
        query: str, 
        variations: List[str], 
        top_k: int,
        min_similarity: float = 0.5
    ) -> List[str]:
        """
        Filter variations by embedding similarity
        Keep variations that are:
        - Similar enough (> min_similarity)
        - Not too similar (< 0.95 để tránh duplicate)
        """
        if not variations:
            return []
        
        # Embed all
        query_vec = self.embed_model.encode(query, convert_to_numpy=True)
        var_vecs = self.embed_model.encode(variations, convert_to_numpy=True)
        
        # Calculate cosine similarity
        from numpy.linalg import norm
        scored = []
        for i, var in enumerate(variations):
            sim = np.dot(query_vec, var_vecs[i]) / (norm(query_vec) * norm(var_vecs[i]))
            
            # Keep if similar enough but not duplicate
            if min_similarity < sim < 0.95:
                scored.append((sim, var))
        
        # Sort by similarity (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Return top K
        return [var for _, var in scored[:top_k]]


# ============================================
# USAGE IN CRAG
# ============================================

class CRAGRetrieverWithExpansion:
    """
    Enhanced CRAG với Query Expansion
    """
    
    def __init__(self, base_retriever, expander: QueryExpander):
        self.retriever = base_retriever
        self.expander = expander
    
    def retrieve(self, query: str, top_k_initial: int = 4, top_k_final: int = 2):
        """
        Retrieve với query expansion
        
        Flow:
        1. Expand query → [q1, q2, q3]
        2. Retrieve từng query → chunks
        3. Merge + deduplicate
        4. CRAG grading
        5. Return refined chunks
        """
        # Step 1: Expand query
        expanded_queries = self.expander.expand(query, num_variations=2)
        
        # Step 2: Retrieve for all variations
        all_candidates = []
        seen_ids = set()
        
        for exp_q in expanded_queries:
            # Semantic search
            query_vec = self.retriever.embed_query(exp_q)
            candidates = self.retriever.semantic_search(query_vec, top_k=top_k_initial)
            
            # Deduplicate
            for cand in candidates:
                cand_id = cand.get("chunk_id")
                if cand_id and cand_id not in seen_ids:
                    seen_ids.add(cand_id)
                    all_candidates.append(cand)
        
        print(f"[CRAG+Expansion] Retrieved {len(all_candidates)} unique chunks from {len(expanded_queries)} query variations")
        
        if not all_candidates:
            return {
                "query": query,
                "refined_chunks": [],
                "graded_stats": {"correct": 0, "incorrect": 0, "ambiguous": 0},
                "action_taken": "NONE"
            }
        
        # Step 3: CRAG grading (dùng original query)
        graded = self.retriever.evaluate_relevance(query, all_candidates)
        
        # Step 4: Decide action
        action = self.retriever.decide_action(graded)
        
        # Step 5: Apply correction
        refined_chunks = self.retriever.apply_correction(query, graded, action)
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
            "num_expanded_queries": len(expanded_queries)
        }