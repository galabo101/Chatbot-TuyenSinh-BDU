import sqlite3
import uuid
import json  # Thêm import
from datetime import datetime

DB_FILE = "chat_history.db"

def init_db():
    """Khởi tạo database với schema mới hỗ trợ sources"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Bảng conversations giữ nguyên
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id TEXT PRIMARY KEY, 
                  user_id TEXT, 
                  title TEXT, 
                  created_at DATETIME)''')
    
    # Bảng messages - THÊM cột sources
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  conversation_id TEXT, 
                  role TEXT, 
                  content TEXT, 
                  sources TEXT,
                  created_at DATETIME)''')
    
    # Migration: Thêm cột sources nếu chưa có (cho DB cũ)
    try:
        c.execute("ALTER TABLE messages ADD COLUMN sources TEXT")
    except sqlite3.OperationalError:
        pass  # Cột đã tồn tại
    
    conn.commit()
    conn.close()


def save_message(conversation_id, role, content, sources=None):
    """
    Lưu tin nhắn KÈM sources
    
    Args:
        conversation_id: ID cuộc hội thoại
        role: "user" hoặc "assistant"
        content: Nội dung tin nhắn
        sources: List[Dict] nguồn tham khảo (chỉ cho assistant)
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Serialize sources thành JSON string
    sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
    
    c.execute(
        "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)", 
        (conversation_id, role, content, sources_json, created_at)
    )
    
    # Cập nhật tiêu đề nếu là tin nhắn user đầu tiên
    if role == "user":
        c.execute("SELECT count(*) FROM messages WHERE conversation_id = ?", (conversation_id,))
        if c.fetchone()[0] <= 1:
            new_title = content[:40] + "..." if len(content) > 40 else content
            c.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, conversation_id))
    
    conn.commit()
    conn.close()


def get_messages(conversation_id):
    """
    Lấy nội dung chat KÈM sources
    
    Returns:
        List[Dict] với keys: role, content, sources (nếu có)
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "SELECT role, content, sources FROM messages WHERE conversation_id = ? ORDER BY id", 
        (conversation_id,)
    )
    rows = c.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        msg = {
            "role": row[0], 
            "content": row[1]
        }
        # Deserialize sources từ JSON
        if row[2]:
            try:
                msg["sources"] = json.loads(row[2])
            except json.JSONDecodeError:
                msg["sources"] = []
        
        messages.append(msg)
    
    return messages


# Các hàm khác giữ nguyên...
def create_conversation(user_id, first_message=None):
    """Tạo cuộc hội thoại mới"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    conv_id = str(uuid.uuid4())
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = "Cuộc trò chuyện mới"
    if first_message:
        title = first_message[:30] + "..."
    c.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)", (conv_id, user_id, title, created_at))
    conn.commit()
    conn.close()
    return conv_id


def get_user_conversations(user_id):
    """Lấy danh sách chat của user"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows


def delete_conversation(conversation_id):
    """Xóa cuộc hội thoại"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    c.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


def delete_all_conversations(user_id):
    """Xóa tất cả"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM conversations WHERE user_id = ?", (user_id,))
    ids = [row[0] for row in c.fetchall()]
    for i in ids:
        c.execute("DELETE FROM messages WHERE conversation_id = ?", (i,))
    c.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
