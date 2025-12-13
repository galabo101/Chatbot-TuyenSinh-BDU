import os
import sqlite3
import shutil
import base64
import time
from collections import Counter
import pandas as pd
import json
from datetime import datetime
import fitz  # PyMuPDF
from docx import Document
from groq import Groq
from src.embedding.indexer import QdrantIndexer
from src.security.security import SecurityManager 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from dotenv import load_dotenv

load_dotenv()

# --- CẤU HÌNH ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
DB_FILE = "chat_history.db"

# Đường dẫn file stopwords
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOPWORDS_PATH = os.path.join(BASE_DIR, "data", "vietnamese-stopwords.txt")

# Khởi tạo Security Manager
security_manager = SecurityManager()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
    length_function=len
)


def load_vietnamese_stopwords():
    """Đọc file stopwords từ đĩa"""
    default_stopwords = {
        "không", "được", "những", "trong", "nhưng", "cũng", "này", 
        "đang", "với", "theo", "rằng", "việc", "người", "chúng", "của", "và", "là"
    }
    
    if os.path.exists(STOPWORDS_PATH):
        try:
            with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except Exception as e:
            print(f"⚠️ Lỗi đọc file stopwords: {e}. Dùng mặc định.")
            return default_stopwords
    return default_stopwords

def get_chat_stats():
    """Lấy thống kê tổng quan"""
    try:
        conn = sqlite3.connect(DB_FILE)
        df_convs = pd.read_sql("SELECT COUNT(*) as cnt FROM conversations", conn)
        total_convs = df_convs.iloc[0]['cnt'] if not df_convs.empty else 0
        
        df_msgs = pd.read_sql("SELECT COUNT(*) as cnt FROM messages", conn)
        total_msgs = df_msgs.iloc[0]['cnt'] if not df_msgs.empty else 0
        
        recent = pd.read_sql("""
            SELECT content, created_at FROM messages 
            WHERE role='user' ORDER BY created_at DESC LIMIT 20
        """, conn)
        conn.close()
        
        return {
            'total_conversations': total_convs,
            'total_messages': total_msgs,
            'recent_questions': recent
        }
    except Exception as e:
        print(f"Stats Error: {e}")
        return {
            'total_conversations': 0,
            'total_messages': 0,
            'recent_questions': pd.DataFrame()
        }

def get_top_keywords():
    """Phân tích từ khóa nổi bật"""
    try:
        stopwords = load_vietnamese_stopwords()
        
        conn = sqlite3.connect(DB_FILE)
        msgs = pd.read_sql("SELECT content FROM messages WHERE role='user'", conn)
        conn.close()
        
        if msgs.empty: return []

        all_keywords = []
        for content in msgs['content']:
            if not content: continue
            words = str(content).lower().split()
            filtered = [w for w in words if len(w) > 2 and w not in stopwords]
            all_keywords.extend(filtered)
        
        return Counter(all_keywords).most_common(10)
    except Exception as e:
        print(f"Keyword Error: {e}")
        return []

# ============================================
# PHẦN 3: VISION PARSER (LLAMA 4 SCOUT)
# ============================================

class GroqParser:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def encode_image(self, image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')

    def parse_page_to_markdown(self, image_bytes, page_num, retry_count=0):        
        base64_image = self.encode_image(image_bytes)
        
        prompt = """Bạn là một công cụ chuyển đổi tài liệu chính xác (OCR & Layout Analysis).
Nhiệm vụ: Chuyển đổi toàn bộ nội dung trong hình ảnh trang tài liệu này thành định dạng MARKDOWN.

Yêu cầu bắt buộc:
1. GIỮ NGUYÊN cấu trúc: Tiêu đề dùng #, Bảng biểu dùng Markdown Table (|...|), Danh sách dùng (-).
2. TRÍCH XUẤT CHÍNH XÁC: Không tóm tắt, không bỏ sót chữ, giữ nguyên tiếng Việt.
3. KHÔNG thêm lời dẫn: Không trả lời "Đây là nội dung...", chỉ trả về nội dung Markdown thuần túy.
4. Nếu có hình ảnh minh họa (không phải văn bản), hãy mô tả nó trong ngoặc vuông: [Hình ảnh: mô tả...].
"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                model=MODEL_ID,
                temperature=0.0, 
                max_tokens=4096, 
            )
            print(f"   ✅ Đã parse xong trang {page_num} (Model: {MODEL_ID})")
            return response.choices[0].message.content
        except Exception as e:
            if retry_count < 2:
                print(f"   🔄 Lỗi mạng/Rate Limit trang {page_num}. Thử lại lần {retry_count + 1}...")
                time.sleep(5)
                return self.parse_page_to_markdown(image_bytes, page_num, retry_count + 1)
            print(f"   ❌ Bỏ qua trang {page_num}: {str(e)}")
            return ""

# ============================================
# PHẦN 4: HÀM XỬ LÝ CHÍNH (UPLOAD PIPELINE)
# ============================================

def process_uploaded_file(uploaded_file):
    
    # 1. SECURITY CHECK
    is_valid, error_msg = security_manager.validate_file(uploaded_file)
    if not is_valid:
        raise ValueError(f"Bảo mật từ chối: {error_msg}")

    save_dir = "./data/uploaded_temp"
    try:
        os.makedirs(save_dir, exist_ok=True)
        safe_filename = security_manager.get_safe_filename(uploaded_file.name)
        file_path = os.path.join(save_dir, safe_filename)
        
        # Đọc dữ liệu vào bộ nhớ đệm
        file_bytes = uploaded_file.getbuffer()
        with open(file_path, "wb") as f:
            f.write(file_bytes)
            
        parser = GroqParser()
        full_markdown_text = ""
        original_display_name = uploaded_file.name
        doc_type = "text_plain" # Mặc định
        
        # === A. PDF (Vision) ===
        if original_display_name.lower().endswith('.pdf'):
            doc_type = "text_markdown"
            try:
                doc = fitz.open(file_path)
                total_pages = len(doc)
                print(f"📄 Quét PDF {total_pages} trang...")
                for page_num in range(total_pages):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=150) 
                    img_bytes = pix.tobytes("png")
                    page_markdown = parser.parse_page_to_markdown(img_bytes, f"Trang {page_num+1}")
                    full_markdown_text += f"\n\n--- Trang {page_num + 1} ---\n\n" + (page_markdown or "")
                    time.sleep(2) 
            except Exception as e: print(f"Lỗi PDF: {e}")

        # === B. ẢNH PNG/JPG (Vision - MỚI) ===
        elif original_display_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            doc_type = "image_description"
            try:
                print(f"🖼️ Đang phân tích hình ảnh: {original_display_name}...")
                # Gửi thẳng bytes của ảnh lên Groq
                image_desc = parser.parse_page_to_markdown(bytes(file_bytes), "Image File")
                full_markdown_text = f"**[Mô tả hình ảnh: {original_display_name}]**\n\n{image_desc}"
            except Exception as e: print(f"Lỗi Ảnh: {e}")

        # === C. EXCEL (Pandas) ===
        elif original_display_name.lower().endswith('.xlsx'):
            doc_type = "table_markdown"
            try:
                df = pd.read_excel(file_path)
                full_markdown_text = df.to_markdown(index=False)
                print(f"📊 Đã đọc Excel: {len(df)} dòng")
            except Exception as e: print(f"Lỗi Excel: {e}")

        # === D. JSON (Text - MỚI) ===
        elif original_display_name.lower().endswith('.json'):
            doc_type = "json_text"
            try:
                # Đọc JSON và chuyển thành chuỗi đẹp
                json_content = json.load(open(file_path, 'r', encoding='utf-8'))
                full_markdown_text = json.dumps(json_content, ensure_ascii=False, indent=2)
                print(f"DATA Đã đọc JSON")
            except Exception as e: print(f"Lỗi JSON: {e}")
                
        # === E. WORD (Python-docx) ===
        elif original_display_name.endswith('.docx'):
            try:
                doc = Document(file_path)
                full_markdown_text = "\n".join([para.text for para in doc.paragraphs])
            except: pass
            
        # === F. TXT ===
        else: 
            try: full_markdown_text = str(uploaded_file.read(), "utf-8")
            except: pass

        # === CHUNKING & INDEXING (Dùng chung cho tất cả) ===
        chunks_data = []
        if full_markdown_text and full_markdown_text.strip():
            text_chunks = text_splitter.split_text(full_markdown_text)
            
            for content in text_chunks:
                chunks_data.append({
                    "content": content,
                    "type": doc_type,
                    "source": "admin_upload_multimodal"
                })

        if chunks_data:
            indexer = QdrantIndexer(
                qdrant_path="./qdrant_data",
                collection_name="bdu_chunks_gemma",
                embedding_model="google/embeddinggemma-300m"
            )
            
            temp_jsonl = os.path.join(save_dir, "temp_chunks.jsonl")
            with open(temp_jsonl, "w", encoding="utf-8") as f:
                for i, item in enumerate(chunks_data):
                    payload = {
                        "chunk_id": f"upload_{datetime.now().timestamp()}_{i}",
                        "content": item["content"], 
                        "url": "Tài liệu Admin Upload",
                        "title": original_display_name, 
                        "type": item["type"],
                        "full_content": item["content"] 
                    }
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            indexer.index_jsonl(temp_jsonl)
            
        return len(chunks_data)

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        raise e 
    finally:       
        if os.path.exists(save_dir):
            try: shutil.rmtree(save_dir)
            except: pass