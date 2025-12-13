import os
import json
import time
import re
import google.generativeai as genai
from pathlib import Path
from PIL import Image

class ImageOCR:
    def __init__(self, kq_dir: str, output_file: str, metadata_file: str, api_key: str):
        self.kq_dir = kq_dir
        self.output_file = output_file
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()
        self.image_counter = 0
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    def _extract_slug_from_url(self, url: str) -> str:
        """Extract slug from URL"""
        path = url.rstrip('/').split('/')[-1].replace('.html', '')
        return re.sub(r'-\d+$', '', path)[:100]
    
    def _load_metadata(self) -> dict:
        """Load metadata to map folder -> URL"""
        url_map = {}
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, dict) and "results" in data:
                    data = data["results"]
                elif isinstance(data, dict):
                    data = [data]
                
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    
                    url = item.get("url")
                    if not url:
                        continue
                    
                    # Multiple mapping strategies
                    slug = self._extract_slug_from_url(url)
                    url_map[slug] = url
                    
                    for key in ["file_hash", "filename", "file", "slug", "id"]:
                        if item.get(key):
                            clean_key = str(item[key]).replace(".txt", "")
                            url_map[clean_key] = url
        
        except Exception as e:
            print(f"⚠️  Warning: Could not load metadata - {e}")
        
        return url_map
    
    def _is_image(self, filename: str) -> bool:
        """Check if file is an image"""
        extensions = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'}
        return Path(filename).suffix.lower() in extensions
    
    def _detect_and_ocr(self, image_path: str) -> tuple:
        """
        Detect type, OCR image, and summarize for RAG.
        Returns: (img_type, full_content, summary_content)
        """
        try:
            img = Image.open(image_path)
            
            # === BƯỚC 1: PHÂN LOẠI ẢNH ===
            detect_prompt = """Phân loại ảnh này:
- Nếu có BẢNG (table) với nhiều hàng/cột → trả lời: "table"
- Nếu chỉ có TEXT thông thường (brochure, document) → trả lời: "text"
Chỉ trả lời 1 từ: "table" hoặc "text"
"""
            
            detection = self.model.generate_content([detect_prompt, img])
            img_type = detection.text.strip().lower()
            
            # === BƯỚC 2: TRÍCH XUẤT NỘI DUNG ĐẦY ĐỦ (FULL CONTENT) ===
            if 'table' in img_type:
                ocr_prompt = """Trích xuất BẢNG này thành text có cấu trúc rõ ràng.
Format yêu cầu:
- Dòng đầu: Tên cột 1 | Tên cột 2 | Tên cột 3
- Các dòng sau: Giá trị 1 | Giá trị 2 | Giá trị 3
QUAN TRỌNG: Giữ nguyên văn bản tiếng Việt. Không thêm giải thích. Chỉ xuất bảng."""
                result_type = "table"
            else:
                ocr_prompt = """Trích xuất TẤT CẢ văn bản từ ảnh này.
Yêu cầu:
- Giữ nguyên định dạng (tiêu đề, bullet points)
- Giữ nguyên tiếng Việt, không bỏ sót thông tin.
- Không thêm giải thích. Chỉ xuất văn bản gốc."""
                result_type = "text"
            
            response = self.model.generate_content([ocr_prompt, img])
            full_content = response.text.strip()
            
            # Clean AI-generated prefixes
            unwanted_prefixes = ["Dưới đây là", "Đây là", "Nội dung", "Bảng", "Here is"]
            for prefix in unwanted_prefixes:
                if full_content.lower().startswith(prefix.lower()):
                    newline_idx = full_content.find('\n')
                    if newline_idx > 0:
                        full_content = full_content[newline_idx+1:].strip()
                    break
            
            if not full_content:
                img.close()
                return result_type, None, None

            # === BƯỚC 3: TÓM TẮT NỘI DUNG (SUMMARY CONTENT) ===
            summary_content = ""
            if len(full_content) < 300: # Nếu quá ngắn, dùng luôn nội dung gốc
                summary_content = full_content
            else:
                try:
                    summarize_prompt = f"""Tóm tắt nội dung sau thành 1-2 câu mô tả ngắn gọn.
Mục đích là để tìm kiếm (embedding), không phải để trả lời.
Ví dụ: "Bảng học phí năm 2024" hoặc "Thông báo 7 bước nhập học".

Nội dung cần tóm tắt:
{full_content[:2000]}... 
"""
                    summary_response = self.model.generate_content(summarize_prompt)
                    summary_content = summary_response.text.strip()
                except Exception as e:
                    print(f"    ⚠️  Lỗi khi tóm tắt: {e}")
                    summary_content = full_content[:200] # Fallback

            # === BƯỚC 4: LÀM SẠCH BẢN TÓM TẮT ===
            # Loại bỏ các câu "chat" của AI mà bạn đã phát hiện
            ai_noise_patterns = [
                re.compile(r"Đây là một số lựa chọn tóm tắt.*?:", re.IGNORECASE),
                re.compile(r"Dưới đây là bản tóm tắt.*?:", re.IGNORECASE),
                re.compile(r"^\s*-\s*", re.MULTILINE), # Xóa các bullet point ở đầu
                re.compile(r"[\"']", re.MULTILINE) # Xóa dấu ngoặc kép/đơn
            ]
            
            for pattern in ai_noise_patterns:
                summary_content = pattern.sub("", summary_content).strip()

            # Nếu sau khi làm sạch mà tóm tắt bị rỗng, dùng fallback
            if not summary_content:
                summary_content = full_content[:200] # Fallback: lấy 200 ký tự đầu

            img.close()
            return result_type, full_content, summary_content
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return None, None, None
    
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
                        if not order:
                            order = chunk.get('order', 0)
                        max_order = max(max_order, order)
        except:
            pass
        
        return max_order + 1
    
    def process_all(self):
        """Scan and OCR all images"""
        mode = 'a' if os.path.exists(self.output_file) else 'w'
        next_order = self._get_next_order()
        
        print(f"📋 Loaded {len(self.metadata)} URL mappings\n")
        
        with open(self.output_file, mode, encoding='utf-8') as jsonl_file:
            for folder_name in sorted(os.listdir(self.kq_dir)):
                folder_path = os.path.join(self.kq_dir, folder_name)
                
                if not os.path.isdir(folder_path):
                    continue
                
                images_dir = os.path.join(folder_path, 'images')
                if not os.path.isdir(images_dir):
                    continue
                
                # Get URL for this folder
                file_hash = folder_name.replace('.txt', '')
                url = self.metadata.get(file_hash)
                
                # Fallback: fuzzy match
                if not url:
                    for key, val in self.metadata.items():
                        if file_hash in key or key in file_hash:
                            url = val
                            break
                
                if not url:
                    url = 'unknown'
                    print(f"⚠️  {folder_name} - No URL found")
                
                print(f"\n📁 {folder_name}")
                
                image_files = sorted([f for f in os.listdir(images_dir) 
                                     if self._is_image(f)])
                
                if not image_files:
                    print("    ⚠️  No images found")
                    continue
                
                for img_file in image_files:
                    img_path = os.path.join(images_dir, img_file)
                    print(f"    🔤 Processing: {img_file}")
                    
                    # Lấy cả 3 giá trị
                    img_type, full_content, summary_content = self._detect_and_ocr(img_path)
                    
                    if not full_content or not summary_content:
                        print("    ❌ Bỏ qua (Không có nội dung)")
                        continue
                    
                    self.image_counter += 1
                    
                    # One image = One chunk (Summary-RAG)
                    chunk_obj = {
                        'chunk_id': f"{file_hash}_image_{self.image_counter}",
                        'url': url,
                        'content': summary_content, # Tóm tắt (cho embedding)
                        'metadata': {
                            'type': f"image_{img_type}", # VD: image_table, image_text
                            'order': next_order,
                            'source_file': img_file,
                            'full_content': full_content # Nội dung đầy đủ (cho LLM)
                        }
                    }
                    
                    jsonl_file.write(json.dumps(chunk_obj, ensure_ascii=False) + '\n')
                    print(f"    ✅ {img_type} - {len(full_content)} chars - {len(summary_content)} chars (summary)")
                    
                    next_order += 1
                    time.sleep(2)  # Rate limiting for Gemini
                
                print(f"    📊 {len(image_files)} images processed")
        
        print(f"\n{'='*60}")
        print(f"✨ Total images: {self.image_counter}")
        print(f"💾 Output: {self.output_file}")
        print(f"{'='*60}")


if __name__ == "__main__":
    
    API_KEY = "AIzaSyClYVCbxN1B2IKsDeUmu7YS5EyF9923fqo" 
    
    KQ_DIR = r"C:\Users\nguye\OneDrive\Desktop\New folder (2)\raw_data\KQ"
    OUTPUT_FILE = r"C:\Users\nguye\OneDrive\Desktop\New folder (2)\chunk_data\chunks.jsonl"
    METADATA_FILE = r"C:\Users\nguye\OneDrive\Desktop\New folder (2)\raw_data\KQ\metadata.json"
    
    print(f"{'='*60}\n🖼️  OCR Images with Gemini 2.5 Flash (Summary-RAG)\n{'='*60}\n")
    
    if "AIzaSy" not in API_KEY:
        print("❌ Vui lòng cập nhật API_KEY")
    else:
        try:
            ocr = ImageOCR(KQ_DIR, OUTPUT_FILE, METADATA_FILE, API_KEY)
            ocr.process_all()
            print("\n✅ Done!")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()