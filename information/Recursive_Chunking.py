import os
import json
import re
from pathlib import Path

class RecursiveChunker:
    """Recursive text chunker optimized for Vietnamese content"""
    
    def __init__(self, cleaned_text_dir: str, output_file: str, metadata_file: str):
        self.cleaned_text_dir = cleaned_text_dir
        self.output_file = output_file
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()
        self.chunk_id_counter = 0
    
    def _extract_slug_from_url(self, url: str) -> str:
        """Extract slug from URL"""
        path = url.rstrip('/').split('/')[-1].replace('.html', '')
        return re.sub(r'-\d+$', '', path)[:100]
    
    def _load_metadata(self) -> dict:
        """Load URL metadata"""
        url_map = {}
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'results' in data:
                    data = data['results']
                for item in data:
                    if item.get('status') == 'success' and item.get('url'):
                        slug = self._extract_slug_from_url(item['url'])
                        url_map[slug] = item['url']
        except Exception as e:
            print(f"Warning: Could not load metadata - {e}")
        return url_map
    
    def _extract_title(self, text: str) -> tuple:
        """Extract title from first line. Returns: (title, remaining_text)"""
        lines = text.split('\n', 1)
        if len(lines) > 1:
            first_line = lines[0].strip()
            if len(first_line) < 150 and '.' not in first_line[-20:]:
                return first_line, lines[1]
        return None, text


    def _clean_source_text(self, text: str) -> str:
        """
        Làm sạch dựa trên Báo cáo Kiểm toán VÀ phản hồi của người dùng.
        1. Xóa nhiễu header (date/time) ở 3 dòng đầu.
        2. THAY THẾ (không xóa) các URL nội dung bằng placeholder [URL: domain.com].
        3. Xóa nhiễu toàn văn bản (icons, signatures, lines, long words).
        4. Giữ lại thông tin ngữ nghĩa (emails, phones, dates trong thân bài).
        5. Bảo toàn cấu trúc \n\n cho chunking.
        """
        
        # === 1. ĐỊNH NGHĨA CÁC MẪU (DỰA TRÊN BÁO CÁO) ===
        patterns = {
            # Chỉ dùng cho Header (3 dòng đầu)
            "header_date": re.compile(r'\b(Thứ \w+ - \d{2}/\d{2}/\d{4})|(\d{2}/\d{2}/\d{4})\b'),
            "header_time": re.compile(r'\b\d{2}:\d{2}(:\d{2})?\b'),
            
            # Regex để tìm và trích xuất URL nội dung
            "urls_in_content": re.compile(r'https?://(www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z0-9./-]+)'),
            
            # Dùng cho Toàn bộ văn bản
            "icons": re.compile(r'[⭐️◾◼️]'), 
            "signatures": re.compile(r'(GV HƯỚN[G DẪN])|(VIỆN TRƯỞG)', re.IGNORECASE),
            "decorative_lines": re.compile(r'(-{5,})|(={5,})|(\*{5,})'),
            "long_words": re.compile(r'\b\w{35,}\b'), 
            
            # Chuẩn hóa (luôn chạy cuối)
            "excessive_newlines": re.compile(r'(\n\s*){3,}'), # 3+ xuống dòng
            "excessive_spaces": re.compile(r'[ \t]+') # Nhiều dấu cách
        }
        
        # === 2. LÀM SẠCH HEADER ===
        lines = text.split('\n')
        header_lines = lines[:3]
        body_lines = lines[3:]
        
        cleaned_header_lines = []
        for line in header_lines:
            line = patterns["header_date"].sub("", line) # Xóa ngày
            line = patterns["header_time"].sub("", line) # Xóa giờ
            cleaned_header_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_header_lines) + '\n' + '\n'.join(body_lines)
        
        # === 3. LÀM SẠCH TOÀN VĂN BẢN ===
        # Thay thế URL bằng placeholder [URL: domain.com]
        cleaned_text = patterns["urls_in_content"].sub(r'[URL: \2]', cleaned_text)
        
        # Xóa các nhiễu còn lại
        cleaned_text = patterns["icons"].sub("", cleaned_text)
        cleaned_text = patterns["signatures"].sub("", cleaned_text)
        cleaned_text = patterns["decorative_lines"].sub("", cleaned_text)
        cleaned_text = patterns["long_words"].sub("", cleaned_text)
        
        # === 4. CHUẨN HÓA CUỐI CÙNG ===
        cleaned_text = patterns["excessive_newlines"].sub("\n\n", cleaned_text) # Giữ lại cấu trúc đoạn \n\n
        cleaned_text = patterns["excessive_spaces"].sub(" ", cleaned_text)
        
        return cleaned_text.strip()
    # ======================================================================
    # === KẾT THÚC HÀM LÀM SẠCH ===
    # ======================================================================

    def _split_text(self, text: str, separator: str) -> list:
        """Split text by separator type"""
        if separator == 'paragraph':
            return [p.strip() for p in re.split(r'\n\n+|\n(?=\d+\.\s)', text) if p.strip()]
        elif separator == 'sentence':
            sentences = re.split(
                r'(?<=[.!?:])\s+(?=[A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ0-9])|(?<=\n)(?=\S)', 
                text
            )
            return [s.strip() for s in sentences if len(s.strip()) > 15]
        elif separator == 'token':
            return [w for w in text.split() if w.strip()]
        else:
            return text.split(separator)
    
    def _recursive_split(self, text: str, separators: list, max_size: int) -> list:
        """Recursively split text using separator hierarchy"""
        if len(text) <= max_size:
            return [text]
        
        if not separators:
            print(f"  Warning: Hard cutting at {max_size} chars")
            return [text[i:i+max_size] for i in range(0, len(text), max_size)]
        
        parts = self._split_text(text, separators[0])
        good_splits = []
        
        for part in parts:
            if len(part) <= max_size:
                good_splits.append(part)
            elif len(separators) > 1:
                good_splits.extend(self._recursive_split(part, separators[1:], max_size))
            else:
                if len(part) > max_size * 1.2:
                    print(f"  Large chunk: {len(part)} chars")
                good_splits.append(part)
        
        return good_splits
    
    def _merge_chunks(self, splits: list, target_size: int, max_size: int, min_size: int) -> list:
        """Merge splits into balanced chunks"""
        if not splits:
            return []
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif len(current_chunk) + len(split) + 1 <= max_size:
                current_chunk += "\n" + split
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = split
        
        # Handle last chunk
        if current_chunk:
            if chunks and len(current_chunk) < min_size:
                chunks[-1] += "\n" + current_chunk
            else:
                chunks.append(current_chunk)
        
        return chunks
    
    def chunk_text(self, text: str, target_tokens: int = 200, max_tokens: int = 250, 
                   min_tokens: int = 100) -> list:
        """
        Main chunking method
        Params: target=180w (~900c), max=250w (~1250c), min=100w (~500c)
        """
        separators = ['paragraph', 'sentence', 'token']
        AVG_CHARS_PER_TOKEN = 5
        
        splits = self._recursive_split(text, separators, max_tokens * AVG_CHARS_PER_TOKEN)
        chunks = self._merge_chunks(splits, target_tokens * AVG_CHARS_PER_TOKEN, 
                                    max_tokens * AVG_CHARS_PER_TOKEN, 
                                    min_tokens * AVG_CHARS_PER_TOKEN)
        return chunks
    
    def process_all(self):
        """Process all files and generate chunks.jsonl"""
        if not os.path.exists(self.cleaned_text_dir):
            raise FileNotFoundError(f" Directory not found: {self.cleaned_text_dir}")
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        stats = {'total': 0, 'success': 0, 'failed': 0, 'empty': 0}
        
        with open(self.output_file, 'a', encoding='utf-8') as jsonl_file:
            for filename in sorted(os.listdir(self.cleaned_text_dir)):
                if not filename.endswith('.txt'):
                    continue
                
                stats['total'] += 1
                filepath = os.path.join(self.cleaned_text_dir, filename)
                file_hash = filename.replace('.txt', '')
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                   
                    text = self._clean_source_text(text)
                    
                    
                    if not text.strip():
                        print(f"  {file_hash} - Empty, skipped")
                        stats['empty'] += 1
                        continue
                    
                    url = self.metadata.get(file_hash, 'unknown')
                    
                    # Trích xuất tiêu đề từ văn bản ĐÃ SẠCH
                    title, content = self._extract_title(text)
                    
                    chunks = self.chunk_text(content)
                    
                    # Write chunks
                    for order, chunk in enumerate(chunks, 1):
                        self.chunk_id_counter += 1
                        chunk_content = f"{title}\n\n{chunk}" if title and order == 1 else chunk
                        
                        chunk_obj = {
                            'chunk_id': f"{file_hash}_chunk_{order}",
                            'url': url,
                            'content': chunk_content,
                            'metadata': {
                                'order': order,
                                'total_chunks': len(chunks),
                                'title': title
                            }
                        }
                        jsonl_file.write(json.dumps(chunk_obj, ensure_ascii=False) + '\n')
                    
                    stats['success'] += 1
                    token_counts = [len(c.split()) for c in chunks]
                    print(f"✓ {file_hash:<60} | {len(chunks):>2} chunks | "
                          f"{sum(token_counts)//len(token_counts):>3}w avg | "
                          f"{min(token_counts):>3}-{max(token_counts):>3}w")
                
                except Exception as e:
                    print(f" {file_hash} - Error: {e}")
                    stats['failed'] += 1
                    continue
        
        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY:")
        print(f"   Total files: {stats['total']}")
        print(f"   Success: {stats['success']} | Failed: {stats['failed']} |  Empty: {stats['empty']}")
        print(f"   Total chunks: {self.chunk_id_counter}")
        print(f"   Output: {self.output_file}")
        print(f"{'='*80}")


if __name__ == "__main__":
    CLEANED_TEXT_DIR = r"C:\Users\nguye\OneDrive\Desktop\New folder (2)\data"
    OUTPUT_FILE = r"C:\Users\nguye\OneDrive\Desktop\New folder (2)\chunk_data\chunks.jsonl"
    METADATA_FILE = r"C:\Users\nguye\OneDrive\Desktop\New folder (2)\bo_sung\metadata.json"
    
    print(f"{'='*80}")
    print(f"🔄 RECURSIVE CHUNKING - OPTIMIZED FOR EMBEDDING")
    print(f"   • No overlap • Target: 180w (~900c) • Range: 100-250w")
    print(f"{'='*80}\n")
    
    chunker = RecursiveChunker(CLEANED_TEXT_DIR, OUTPUT_FILE, METADATA_FILE)
    chunker.process_all()
    
    print(f"\n Done!")