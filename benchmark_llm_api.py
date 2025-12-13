import time
import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Test prompt
TEST_PROMPT = """Bạn là trợ lý tuyển sinh. Dựa trên thông tin sau, hãy trả lời ngắn gọn:

THÔNG TIN:
- Ngành CNTT: Học phí 8tr/kỳ, Điểm chuẩn 16.
- Ngành Dược: Học phí 15tr/kỳ, Điểm chuẩn 21.

CÂU HỎI: So sánh học phí ngành Dược và CNTT?

TRẢ LỜI:"""

def benchmark_groq_model(model_name: str, num_runs: int = 10):
    """Hàm chung để benchmark các model trên Groq"""
    print(f"\n{'='*60}")
    print(f"🚀 Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY missing")
        return None
    
    client = Groq(api_key=api_key)
    
    # 1. Warm-up (Kiểm tra xem model có tồn tại không)
    print("Warming up...", end=" ")
    try:
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            model=model_name,
            max_tokens=10
        )
        print("✅ Ready!")
    except Exception as e:
        print(f"\n❌ Model Error: {e}")
        print("👉 Gợi ý: Kiểm tra lại tên model trong config.py. Groq thường hỗ trợ: llama-3.3-70b, mixtral-8x7b...")
        return None

    times = []
    tokens_per_sec = []
    
    print(f"Running {num_runs} iterations...")
    for i in range(num_runs):
        start = time.time()
        try:
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": TEST_PROMPT}],
                model=model_name,
                max_tokens=200,
                temperature=0.1
            )
            elapsed = (time.time() - start) * 1000 # ms
            
            # Tính tốc độ
            n_tokens = resp.usage.completion_tokens
            tps = n_tokens / (elapsed / 1000)
            
            times.append(elapsed)
            tokens_per_sec.append(tps)
            
            # In tiến độ gọn
            print(f"  [{i+1}/{num_runs}] Time: {elapsed:.0f}ms | Speed: {tps:.0f} t/s")
            
            # Ngủ nhẹ để tránh rate limit của Groq
            time.sleep(0.5)
                
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
            time.sleep(1)
            
    return {
        "name": model_name,
        "time": np.array(times),
        "speed": np.array(tokens_per_sec)
    }

if __name__ == "__main__":
    results = []
    
    # 1. Benchmark Model Chính
    r1 = benchmark_groq_model("llama-3.3-70b-versatile", num_runs=10)
    if r1: results.append(r1)
    
    # 2. Benchmark Model Dự phòng (Theo config của bạn)
    # Lưu ý: Nếu tên model sai, nó sẽ báo lỗi ở bước Warm-up
    r2 = benchmark_groq_model("openai/gpt-oss-120b", num_runs=10)
    if r2: results.append(r2)
    
    # Summary Table
    if results:
        print(f"\n\n{'='*85}")
        print(f"{'MODEL':<30} | {'AVG TIME (ms)':<15} | {'SPEED (tok/s)':<15} | {'SCORE'}")
        print(f"{'-'*85}")
        
        # Lấy tốc độ cao nhất làm chuẩn
        max_speed = max(r['speed'].mean() for r in results)
        
        for r in results:
            avg_time = r['time'].mean()
            avg_speed = r['speed'].mean()
            score = avg_speed / max_speed * 100 # % so với model nhanh nhất
            
            print(f"{r['name']:<30} | {avg_time:>13.0f} | {avg_speed:>13.0f} | {score:>4.0f}%")
        print(f"{'='*85}\n")
    else:
        print("\n⚠️ Không có kết quả nào thành công.")