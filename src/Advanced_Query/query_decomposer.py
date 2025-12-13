"""
Query Decomposer - Hybrid Approach
Fast keyword detection + LLM decomposition when needed
"""

from typing import List, Dict
import re
import json
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


class QueryDecomposer:
    def __init__(self, groq_api_key: str = None):
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        
        # Keywords phải đứng giữa 2 mệnh đề có nghĩa
        self.multi_intent_patterns = [
            r'.{10,}\s+và\s+.{10,}',      # "...và..." (cả 2 vế > 10 ký tự)
            r'.{10,}\s+hay\s+.{10,}',     # "...hay..."
            r'.{10,}\s+hoặc\s+.{10,}',    # "...hoặc..."
            r'.{10,}\s+còn\s+.{10,}',     # "...còn..."
            r'.{10,}\s+ngoài ra\s+.{10,}' # "...ngoài ra..."
        ]
        
        print("✅ QueryDecomposer initialized")
    
    def should_decompose(self, query: str) -> Dict[str, any]:
        """Fast heuristic check - NO LLM call"""
        query_lower = query.lower().strip()
        signals = []
        
        # Signal 1: Multi-intent patterns (phải có 2 vế đủ dài)
        for pattern in self.multi_intent_patterns:
            if re.search(pattern, query_lower):
                signals.append(("multi_clause", 0.8))
                break
        
        # Signal 2: Multiple question marks
        if query.count("?") >= 2:
            signals.append(("multiple_questions", 0.9))
        
        # Signal 3: Enumeration rõ ràng (1., 2., a., b.)
        if re.search(r'[0-9]\.\s+\w+.*[0-9]\.\s+\w+', query):
            signals.append(("enumeration", 0.85))
        
        if not signals:
            return {
                "should_decompose": False,
                "reason": "Simple single-intent query",
                "confidence": 0.0
            }
        
        confidence = sum(score for _, score in signals) / len(signals)
        should_decompose = confidence >= 0.75  # Tăng threshold
        
        return {
            "should_decompose": should_decompose,
            "reason": f"Detected: {[s[0] for s in signals]}",
            "confidence": confidence
        }
    
    def decompose(self, query: str) -> List[str]:
        """Main decomposition logic"""
        check_result = self.should_decompose(query)
        
        if not check_result["should_decompose"]:
            print(f"[Decomposer] Single query (confidence: {check_result['confidence']:.2f})")
            return [query]
        
        print(f"[Decomposer] Complex query detected - {check_result['reason']}")
        
        try:
            sub_queries = self._llm_decompose(query)
            
            # Validation: Nếu LLM trả về > 1 nhưng giống nhau hoặc vô nghĩa -> giữ nguyên
            if len(sub_queries) > 1:
                # Check nếu sub-queries quá ngắn hoặc giống query gốc
                valid_subs = [sq for sq in sub_queries if len(sq) > 15 and sq.lower() != query.lower()]
                if len(valid_subs) < 2:
                    print("[Decomposer] LLM output invalid, keeping original")
                    return [query]
                sub_queries = valid_subs
            
            if len(sub_queries) == 1:
                print("[Decomposer] LLM kept as single query")
                return sub_queries
            
            print(f"[Decomposer] ✅ Split into {len(sub_queries)} sub-queries:")
            for i, sq in enumerate(sub_queries, 1):
                print(f"   [{i}] {sq}")
            
            # Giải pháp 2: Soft limit - khi quá nhiều sub-queries
            if len(sub_queries) > 3:
                print("[Decomposer] ⚠️ Câu hỏi quá phức tạp, yêu cầu người dùng chia nhỏ")
                return ["TOO_COMPLEX"]
            
            return sub_queries
            
        except Exception as e:
            print(f"[Decomposer] ❌ Error: {e}, using original query")
            return [query]
    
    def _llm_decompose(self, query: str) -> List[str]:
        """Use LLM to decompose query"""
        prompt = f"""Phân tách câu hỏi phức tạp thành các câu hỏi đơn giản, độc lập.

QUY TẮC QUAN TRỌNG:
1. CHỈ phân tách nếu câu hỏi THỰC SỰ có NHIỀU Ý KHÁC NHAU
2. KHÔNG phân tách nếu chỉ là 1 câu hỏi duy nhất
3. Mỗi câu hỏi con phải HOÀN CHỈNH, độc lập
4. KHÔNG tạo thêm câu hỏi không có trong câu gốc
5. TỐI ĐA 3 câu hỏi con


VÍ DỤ ĐÚNG:
Input: "Học phí CNTT bao nhiêu và trường có học bổng không?"
Output: ["Học phí ngành CNTT là bao nhiêu?", "Trường có học bổng không?"]

VÍ DỤ SAI (KHÔNG PHÂN TÁCH):
Input: "Tôi có 18 điểm thì có thể đậu vào ngành nào?"
Output: ["Tôi có 18 điểm thì có thể đậu vào ngành nào?"]
(Đây là 1 câu hỏi duy nhất, KHÔNG phân tách)

Input: "Ngành nào có điểm chuẩn thấp nhất và cao nhất năm 2024?"
Output: ["Ngành nào có điểm chuẩn thấp nhất và cao nhất năm 2024?"]
(Đây là câu SO SÁNH CẶP, KHÔNG tách)

Input: "Điều kiện xét tuyển ngành Logistics?"
Output: ["Điều kiện xét tuyển ngành Logistics?"]

CÂU HỎI CẦN XỬ LÝ:
"{query}"

Trả về JSON array:"""

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2,  # Giảm để ổn định hơn
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        content = re.sub(r'```json\s*|\s*```', '', content)
        
        try:
            sub_queries = json.loads(content)
            
            if not isinstance(sub_queries, list):
                return [query]
            
            sub_queries = [q.strip() for q in sub_queries if isinstance(q, str) and q.strip()]
            return sub_queries if sub_queries else [query]
            
        except json.JSONDecodeError:
            match = re.search(r'\[(.*?)\]', content, re.DOTALL)
            if match:
                items = re.findall(r'"([^"]+)"', match.group(1))
                return items if items else [query]
            return [query]