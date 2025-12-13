"""
Cross-Encoder Reranker - Thay thế RelevanceEvaluator (LLM-based)
Sử dụng model local để rerank documents, giảm latency từ ~1-2s xuống ~0.1-0.3s
"""

from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
import numpy as np


class CrossEncoderReranker:
    """
    Reranker sử dụng Cross-Encoder để đánh giá độ liên quan
    Thay thế cho RelevanceEvaluator (LLM-based) trong CRAG
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        preloaded_model: CrossEncoder = None
    ):
        """
        Args:
            model_name: Tên model Cross-Encoder
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (nhỏ, nhanh, ~80MB)
                - "BAAI/bge-reranker-base" (lớn hơn, chính xác hơn, ~400MB)
            preloaded_model: Model đã load sẵn (để tránh load lại)
        """
        if preloaded_model:
            print("✅ Using preloaded Cross-Encoder model")
            self.model = preloaded_model
        else:
            print(f"🔧 Loading Cross-Encoder: {model_name}")
            self.model = CrossEncoder(model_name)
            print("✅ Cross-Encoder ready")
        
        # Thresholds cho việc phân loại (tương đương CORRECT/AMBIGUOUS/INCORRECT)
        # ✅ Giảm threshold để giữ lại nhiều chunks chi tiết hơn
        self.high_threshold = 0.5   # >= 0.5: CORRECT (giảm từ 0.7)
        self.low_threshold = 0.2    # < 0.2: INCORRECT (giảm từ 0.3)
    
    def get_scores(self, query: str, documents: List[Dict]) -> List[float]:
        """
        Tính điểm relevance cho mỗi document
        
        Args:
            query: Câu hỏi của user
            documents: List[Dict] với key 'content' hoặc 'full_content'
            
        Returns:
            List[float]: Điểm relevance (0-1) cho mỗi document
        """
        if not documents:
            return []
        
        # Chuẩn bị pairs (query, document_content)
        pairs = []
        for doc in documents:
            content = doc.get("full_content") or doc.get("content", "")
            # Giới hạn độ dài để tăng tốc
            content = content[:1000] if len(content) > 1000 else content
            pairs.append([query, content])
        
        # Tính điểm
        scores = self.model.predict(pairs)
        
        # Normalize về 0-1 (ms-marco model trả về logits, cần sigmoid)
        normalized_scores = 1 / (1 + np.exp(-np.array(scores)))
        
        return normalized_scores.tolist()
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: int = None
    ) -> List[Dict]:
        """
        Rerank documents theo độ liên quan
        
        Args:
            query: Câu hỏi của user
            documents: List documents cần rerank
            top_k: Số lượng documents trả về (None = tất cả)
            
        Returns:
            List[Dict]: Documents đã được sắp xếp theo điểm giảm dần
        """
        if not documents:
            return []
        
        scores = self.get_scores(query, documents)
        
        # Gắn score vào documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score
        
        # Sắp xếp theo điểm giảm dần
        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        if top_k:
            sorted_docs = sorted_docs[:top_k]
        
        return sorted_docs
    
    def grade_documents(
        self, 
        query: str, 
        documents: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Phân loại documents thành CORRECT/AMBIGUOUS/INCORRECT
        (Tương thích với interface cũ của RelevanceEvaluator)
        
        Args:
            query: Câu hỏi của user
            documents: List documents cần đánh giá
            
        Returns:
            Dict với keys: 'correct', 'ambiguous', 'incorrect'
        """
        if not documents:
            return {"correct": [], "ambiguous": [], "incorrect": []}
        
        scores = self.get_scores(query, documents)
        
        graded = {
            "correct": [],
            "ambiguous": [],
            "incorrect": []
        }
        
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score
            
            if score >= self.high_threshold:
                graded["correct"].append(doc)
            elif score >= self.low_threshold:
                graded["ambiguous"].append(doc)
            else:
                graded["incorrect"].append(doc)
        
        # Log kết quả
        print(f"[CrossEncoder] Grading results:")
        print(f"   ✅ CORRECT: {len(graded['correct'])}")
        print(f"   ⚠️  AMBIGUOUS: {len(graded['ambiguous'])}")
        print(f"   ❌ INCORRECT: {len(graded['incorrect'])}")
        
        return graded


# Singleton để preload model
_reranker_instance = None

def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    """
    Factory function để lấy singleton instance
    Tránh load model nhiều lần
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker(model_name=model_name)
    return _reranker_instance
