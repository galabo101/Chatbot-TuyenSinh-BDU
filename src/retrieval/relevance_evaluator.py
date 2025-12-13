# [File: src/retrieval/relevance_evaluator.py]
from typing import List, Dict
from groq import Groq
import json
import re


class RelevanceEvaluator:
    """
    Advanced Evaluator: Batch Processing + Smart Extraction + Confidence Score
    """
    
    def __init__(self, llm_client: Groq):
        self.llm = llm_client
        self.model_name = "llama-3.1-8b-instant"
        # Ngưỡng tin cậy: Nếu dưới mức này sẽ bị đánh tụt hạng
        self.confidence_threshold = 0.7 
    
    def evaluate_batch(self, query: str, documents: List[Dict]) -> List[str]:
        """
        Đánh giá độ liên quan kèm độ tin cậy
        """
        if not documents:
            return []
        
        # 1. Chuẩn bị documents với Smart Extraction
        docs_text = ""
        for i, doc in enumerate(documents, 1):
            content = self._extract_relevant_content(query, doc, max_length=500)
            docs_text += f"DOC {i}:\n{content}\n---\n"
        
        # 2. Prompt yêu cầu trả về Label + Confidence
        prompt = f"""Đánh giá tài liệu dựa trên câu hỏi.

CÂU HỎI: "{query}"

TIÊU CHÍ PHÂN LOẠI:
- CORRECT: Chứa thông tin trả lời trực tiếp, cụ thể.
- INCORRECT: Không liên quan.
- AMBIGUOUS: Liên quan nhưng chung chung/thiếu ý.

YÊU CẦU OUTPUT:
Trả về JSON object chứa danh sách "evaluations". Mỗi phần tử gồm:
- "label": [CORRECT/INCORRECT/AMBIGUOUS]
- "confidence": [0.0 đến 1.0] (Độ tự tin của bạn)

INPUT:
{docs_text}

JSON OUTPUT FORMAT (Mẫu):
{{
  "evaluations": [
    {{"label": "CORRECT", "confidence": 0.95}},
    {{"label": "AMBIGUOUS", "confidence": 0.4}}
  ]
}}

CHỈ trả về JSON hợp lệ."""
        
        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            
            # Lấy danh sách evaluations
            evals = data.get("evaluations", [])
            
            # Validate số lượng
            if len(evals) != len(documents):
                print(f"[Evaluator] ⚠️ Mismatch: {len(evals)} evals for {len(documents)} docs")
                # Fill thiếu bằng default
                default_eval = {"label": "AMBIGUOUS", "confidence": 0.0}
                evals = evals[:len(documents)] + [default_eval] * (len(documents) - len(evals))
            
            # 3. Xử lý Logic Confidence Threshold
            final_labels = []
            
            for item in evals:
                raw_label = str(item.get("label", "AMBIGUOUS")).upper()
                confidence = float(item.get("confidence", 0.5))
                
                label = "AMBIGUOUS"
                if "CORRECT" in raw_label:
                    label = "CORRECT"
                elif "INCORRECT" in raw_label:
                    label = "INCORRECT"
                
                # --- LOGIC QUAN TRỌNG: Kiểm tra Confidence ---
                # Nếu máy bảo CORRECT nhưng không tự tin (< 0.7) -> Hạ xuống AMBIGUOUS
                if label == "CORRECT" and confidence < self.confidence_threshold:
                    print(f"[Eval] Downgraded CORRECT (conf={confidence:.2f}) to AMBIGUOUS")
                    label = "AMBIGUOUS"
                # ---------------------------------------------
                
                final_labels.append(label)
            
            # Thống kê nhanh để debug
            print(f"[Batch Eval] ✅ Rated {len(final_labels)} docs (Threshold: {self.confidence_threshold})")
            return final_labels
            
        except Exception as e:
            print(f"[Evaluator] ❌ Error: {e}")
            return ["AMBIGUOUS"] * len(documents)
    
    def _extract_relevant_content(self, query: str, document: Dict, max_length: int = 600) -> str:
        """
        Smart content extraction - tìm đoạn có keywords
        """
        content = document.get("full_content") or document.get("content", "")
        
        if len(content) <= max_length:
            return content
        
        # Lấy từ khóa (bỏ từ ngắn < 3 ký tự)
        query_keywords = {kw for kw in query.lower().split() if len(kw) > 2}
        
        # Tách câu (tương đối)
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', content) if len(s.strip()) > 10]
        
        if not sentences:
            return content[:max_length]
        
        # Chấm điểm câu
        scored = []
        for sent in sentences:
            score = sum(1 for kw in query_keywords if kw in sent.lower())
            scored.append((score, sent))
        
        # Sắp xếp câu điểm cao lên đầu
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Lấy top câu sao cho không vượt quá max_length
        selected = []
        current_length = 0
        
        for score, sent in scored:
            if score == 0 and current_length > max_length / 2: 
                continue # Bỏ qua câu không liên quan nếu đã lấy được 1 ít
            
            if current_length + len(sent) > max_length:
                break
            selected.append(sent)
            current_length += len(sent)
        
        # Fallback
        if not selected:
            half = max_length // 2
            return content[:half] + "\n...\n" + content[-half:]
        
        return " ... ".join(selected)