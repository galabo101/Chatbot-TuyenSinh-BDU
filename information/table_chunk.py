import os
import json
import re
import time
import google.generativeai as genai

class TableConverter:
    def __init__(self, crawl_dir: str, output_file: str, metadata_file: str, api_key: str):
        self.crawl_dir = crawl_dir
        self.output_file = output_file
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()
        self.table_counter = 0
        
        # === (MỚI) Khởi tạo Gemini ===
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('models/gemini-2.5-flash')
            print("✅ Gemini model (table_chunk) initialized.")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo Gemini: {e}")
            raise

    def _extract_slug_from_url(self, url: str) -> str:
        """Extract slug from URL"""
        path = url.rstrip('/').split('/')[-1].replace('.html', '')
        return re.sub(r'-\d+$', '', path)[:100]

    def _load_metadata(self) -> dict:
        """Load metadata to map folder/file -> URL"""
        url_map = {}
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, dict) and "results" in data:
                    data = data["results"]
                
                for item in data:
                    if item.get("status") == "success" and item.get("url"):
                        url = item["url"]
                        slug = self._extract_slug_from_url(url)
                        url_map[slug] = url
                        
                        for key in ["file_hash", "filename", "file"]:
                            if item.get(key):
                                clean_key = item[key].replace(".txt", "")
                                url_map[clean_key] = url
        
        except Exception as e:
            print(f"Warning: Could not load metadata - {e}")
        
        return url_map

    def _convert_table_to_text(self, table_data: dict) -> str:
        """Convert table JSON to readable text"""
        data = table_data.get('data', [])
        if not data:
            return ""
        
        header = data[0]
        rows = data[1:] if len(data) > 1 else []
        
        text_lines = []
        
        if rows:
            for row in rows:
                row_text = []
                for i, cell in enumerate(row):
                    if i < len(header):
                        row_text.append(f"{header[i]}: {cell}")
                    else:
                        row_text.append(str(cell))
                text_lines.append(" | ".join(row_text))
        else:
            text_lines.append(" | ".join(header))
        
        return "\n".join(text_lines)

    # === (MỚI) Hàm tóm tắt ===
    def _summarize_content(self, full_content: str) -> str:
        """Summarize table content for embedding"""
        
        if len(full_content) < 300: # Nếu quá ngắn, dùng luôn
            return full_content
            
        try:
            summarize_prompt = f"""Tóm tắt nội dung BẢNG sau thành 1-2 câu mô tả ngắn gọn.
Mục đích là để tìm kiếm (embedding), không phải để trả lời.
Ví dụ: "Bảng học phí năm 2024" hoặc "Bảng điểm chuẩn các ngành".

Nội dung BẢNG cần tóm tắt:
{full_content[:2000]}... 
"""
            summary_response = self.model.generate_content(summarize_prompt)
            summary_content = summary_response.text.strip()
            
            # Làm sạch nhiễu AI
            ai_noise_patterns = [
                re.compile(r"Đây là.*?:", re.IGNORECASE),
                re.compile(r"Dưới đây là.*?:", re.IGNORECASE),
                re.compile(r"^\s*-\s*", re.MULTILINE),
                re.compile(r"[\"']")
            ]
            for pattern in ai_noise_patterns:
                summary_content = pattern.sub("", summary_content).strip()
            
            if not summary_content:
                return full_content[:200] # Fallback
            
            return summary_content
            
        except Exception as e:
            print(f"    ⚠️  Lỗi khi tóm tắt bảng: {e}")
            return full_content[:200] # Fallback

    def _get_next_order(self) -> int:
        """Get next order number from existing chunks"""
        if not os.path.exists(self.output_file):
            return 1
        
        max_order = 0
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunk = json.loads(line)
                        order = chunk.get('metadata', {}).get('order', 0)
                        max_order = max(max_order, order)
        except:
            pass
        
        return max_order + 1
    
    def process_all(self):
        """Process all tables and append to chunks.jsonl"""
        mode = 'a' if os.path.exists(self.output_file) else 'w'
        next_order = self._get_next_order()
        
        with open(self.output_file, mode, encoding='utf-8') as jsonl_file:
            for folder_name in sorted(os.listdir(self.crawl_dir)):
                folder_path = os.path.join(self.crawl_dir, folder_name)
                
                if not os.path.isdir(folder_path):
                    continue
                
                tables_dir = os.path.join(folder_path, 'tables')
                if not os.path.isdir(tables_dir):
                    continue
                
                file_hash = folder_name.replace('.txt', '')
                url = self.metadata.get(file_hash, 'unknown')
                
                try:
                    table_files = sorted([f for f in os.listdir(tables_dir) 
                                         if f.lower().endswith(('.json', '.jso'))])
                except Exception as e:
                    print(f"  ❌ Error listing tables in {folder_name}: {e}")
                    continue
                
                if not table_files:
                    continue
                
                print(f"\n📁 {folder_name}")
                
                for table_file in table_files:
                    table_path = os.path.join(tables_dir, table_file)
                    
                    try:
                        with open(table_path, 'r', encoding='utf-8') as f:
                            table_data = json.load(f)
                        
                        # BƯỚC 1: Lấy nội dung đầy đủ
                        full_content = self._convert_table_to_text(table_data)
                        
                        if not full_content:
                            continue
                        
                        # BƯỚC 2: Tạo tóm tắt
                        summary_content = self._summarize_content(full_content)
                        
                        if not summary_content:
                            continue
                            
                        self.table_counter += 1
                        
                        # BƯỚC 3: Lưu chunk (Summary-RAG)
                        chunk_obj = {
                            'chunk_id': f"{file_hash}_table_{table_data.get('table_id', self.table_counter)}",
                            'url': url,
                            'content': summary_content, # Tóm tắt (cho embedding)
                            'metadata': {
                                'type': 'table',
                                'order': next_order,
                                'source_file': table_file,
                                'full_content': full_content # Nội dung đầy đủ (cho LLM)
                            }
                        }
                        
                        jsonl_file.write(json.dumps(chunk_obj, ensure_ascii=False) + '\n')
                        print(f"    ✅ {table_file} - {len(full_content)} chars - {len(summary_content)} chars (summary)")
                        
                        next_order += 1
                        time.sleep(2) # Rate limiting
                    
                    except Exception as e:
                        print(f"  ❌ Error processing {table_file}: {e}")
                        continue
                
        print(f"\n{'='*60}")
        print(f"📊 Total tables: {self.table_counter}")
        print(f"💾 Output: {self.output_file}")
        print(f"{'='*60}")


if __name__ == "__main__":
    # !!! QUAN TRỌNG: Hãy đảm bảo API key này là chính xác và còn hoạt động
    API_KEY = "AIzaSyClYVCbxN1B2IKsDeUmu7YS5EyF9923fqo" 
    
    CRAWL_DIR = r"C:\Users\nguye\OneDrive\Desktop\New folder (2)\raw_data\KQ"
    OUTPUT_FILE = r"C:\Users\nguye\OneDrive\Desktop\New folder (2)\chunk_data\chunks.jsonl"
    METADATA_FILE = r"C:\Users\nguye\OneDrive\Desktop\New folder (2)\raw_data\KQ\metadata.json"
    
    print(f"{'='*60}\n📋 Converting Tables to Text (Summary-RAG)\n{'='*60}\n")
    
    if "AIzaSy" not in API_KEY:
        print("❌ Vui lòng cập nhật API_KEY của bạn trong khối __main__!")
    else:
        try:
            converter = TableConverter(CRAWL_DIR, OUTPUT_FILE, METADATA_FILE, API_KEY)
            converter.process_all()
            print(f"\n✅ Done!")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()