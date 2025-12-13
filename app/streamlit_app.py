import streamlit as st
import extra_streamlit_components as stx
import uuid
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
ADMIN_PASS_HASH = hashlib.sha256("admin123".encode()).hexdigest()

# --- 1. SETUP & CONFIG ---
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from src.pipeline import RAGPipeline
from src.database import (
    init_db, create_conversation, save_message, 
    get_user_conversations, get_messages, delete_conversation, delete_all_conversations
)

# Initialize Database
init_db()

st.set_page_config(
    page_title="Chatbot Tuyển Sinh BDU", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# [MỚI] Khởi tạo trạng thái Admin
if "admin_mode" not in st.session_state:
    st.session_state.admin_mode = False

# --- 2. CUSTOM CSS STYLING ---
st.markdown("""
<style>
section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        font-size: 13px !important;
    }    
    section[data-testid="stSidebar"] {
    width: 20vw !important;     
</style>
""", unsafe_allow_html=True)

# --- 3. AUTHENTICATION & USER MANAGEMENT ---
cookie_manager = stx.CookieManager(key="bdu_cookie_mgr")

def get_stable_user_id():
    if "user_id" in st.session_state:
        return st.session_state.user_id

    cookie_user_id = cookie_manager.get("bdu_user_id")
    
    if not cookie_user_id:
        if "auth_retry_count" not in st.session_state:
            st.session_state.auth_retry_count = 0
        
        # ✅ FIX: Tăng retry count và delay để chờ cookie load xong
        # Cookie manager cần vài render cycles để đọc cookie từ browser
        if st.session_state.auth_retry_count < 5:  # Tăng từ 2 lên 5
            st.session_state.auth_retry_count += 1
            time.sleep(0.5)  # Tăng từ 0.3s lên 0.5s
            st.rerun()
            
        # Chỉ tạo user_id mới khi chắc chắn không có cookie
        new_id = str(uuid.uuid4())
        st.session_state.user_id = new_id
        cookie_manager.set("bdu_user_id", new_id, expires_at=datetime.now() + timedelta(days=365))
        return new_id
    
    st.session_state.user_id = cookie_user_id
    st.session_state.auth_retry_count = 0 
    return cookie_user_id

user_id = get_stable_user_id()

# --- 4. SESSION & CHAT STATE MANAGEMENT ---
def get_target_chat_id():
    url_chat_id = st.query_params.get("chat_id")
    user_chats = [c[0] for c in get_user_conversations(user_id)]
    
    if url_chat_id and url_chat_id in user_chats:
        return url_chat_id
    
    if user_chats:
        return user_chats[0]
    
    return create_conversation(user_id)

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id is None:
    target_id = get_target_chat_id()
    st.session_state.current_chat_id = target_id
    st.session_state.messages = get_messages(target_id)
    st.query_params["chat_id"] = target_id
# ✅ FIX: Đảm bảo messages luôn được load từ DB nếu session bị reset
elif "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = get_messages(st.session_state.current_chat_id)

url_id_check = st.query_params.get("chat_id")
if url_id_check and url_id_check != st.session_state.current_chat_id:
    user_chats = [c[0] for c in get_user_conversations(user_id)]
    if url_id_check in user_chats:
        st.session_state.current_chat_id = url_id_check
        st.session_state.messages = get_messages(url_id_check)

# --- 5. AI ENGINE LOADING ---
@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("google/embeddinggemma-300m")

@st.cache_resource
def load_pipeline(_embedding_model):
    return RAGPipeline(
        model_type="gemma", 
        verbose=False,
        preloaded_model=_embedding_model
    )

if "pipeline_loaded" not in st.session_state:
    with st.spinner("🔄 Đang khởi động hệ thống AI..."):
        embedding_model = load_embedding_model()
        pipeline = load_pipeline(embedding_model)
        st.session_state.pipeline_loaded = True
        st.session_state.pipeline = pipeline
    st.toast("✅ Hệ thống đã sẵn sàng!", icon="🚀")
    time.sleep(0.5)

# --- 6. SIDEBAR CONTROLLER ---
with st.sidebar:
    st.markdown("### 💬 Quản lý hội thoại")
    
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        new_id = create_conversation(user_id)
        st.session_state.current_chat_id = new_id
        st.session_state.messages = []
        st.query_params["chat_id"] = new_id
        st.rerun()
    
    st.divider()
    st.caption("LỊCH SỬ CHAT")
    
    history = get_user_conversations(user_id)
    
    if not history:
        st.info("Chưa có lịch sử hội thoại", icon="ℹ️")
    else:
        # ✅ FIX: Thêm current_chat_id vào session nếu chưa có
        if "current_chat_id" not in st.session_state and history:
            st.session_state.current_chat_id = history[0][0]
            st.session_state.messages = get_messages(history[0][0])
        
        for conv_id, title, _ in history:
            display_title = (title[:28] + '..') if title and len(title) > 28 else (title or "Hội thoại mới")
            is_active = conv_id == st.session_state.get("current_chat_id")
            
            col1, col2 = st.columns([0.85, 0.15])
            
            with col1:
                btn_type = "primary" if is_active else "secondary"
                if st.button(f"💬 {display_title}", key=f"btn_{conv_id}", type=btn_type, disabled=is_active, use_container_width=True):
                    st.session_state.current_chat_id = conv_id
                    st.session_state.messages = get_messages(conv_id)
                    st.query_params["chat_id"] = conv_id
                    st.rerun()
            
            with col2:
                if st.button("🗑", key=f"del_{conv_id}", help="Xóa hội thoại"):
                    delete_conversation(conv_id)
                    
                    # ✅ FIX: Nếu xóa chat đang active, chuyển sang chat khác
                    if is_active:
                        remaining_chats = [c[0] for c in get_user_conversations(user_id)]
                        if remaining_chats:
                            st.session_state.current_chat_id = remaining_chats[0]
                            st.session_state.messages = get_messages(remaining_chats[0])
                            st.query_params["chat_id"] = remaining_chats[0]
                        else:
                            # Không còn chat nào, tạo mới
                            new_id = create_conversation(user_id)
                            st.session_state.current_chat_id = new_id
                            st.session_state.messages = []
                            st.query_params["chat_id"] = new_id
                    st.rerun()
        
        if history:
            st.divider()
            if st.button("🗑️ Xóa tất cả lịch sử", use_container_width=True):
                delete_all_conversations(user_id)
                # ✅ FIX: Tạo chat mới sau khi xóa hết
                new_id = create_conversation(user_id)
                st.session_state.current_chat_id = new_id
                st.session_state.messages = []
                st.query_params["chat_id"] = new_id
                st.rerun()

    # --- [MỚI] KHU VỰC LOGIN ADMIN Ở SIDEBAR ---
    st.divider()
    if st.session_state.admin_mode:
        if st.session_state.admin_login_time:
            if (datetime.now() - st.session_state.admin_login_time).seconds > 600:
                st.session_state.admin_mode = False
                st.warning("Phiên quản trị hết hạn. Vui lòng đăng nhập lại.")
                st.rerun()
        if st.button("⬅️ Quay lại Chatbot", use_container_width=True):
            st.session_state.admin_mode = False
            st.rerun()
    else:
        with st.expander("🔐 Quản trị viên"):
            admin_pass = st.text_input("Mật khẩu", type="password")
            if st.button("Đăng nhập"):
                if hashlib.sha256(admin_pass.encode()).hexdigest() == ADMIN_PASS_HASH:
                    st.session_state.admin_mode = True
                    st.rerun()
                else:
                    st.error("Sai mật khẩu")

# --- 7. MESSAGE HANDLING LOGIC ---
def process_query(query_text: str):
    """Xử lý câu hỏi người dùng và sinh câu trả lời"""
    
    # A. Hiển thị & Lưu User Message
    st.session_state.messages.append({"role": "user", "content": query_text})
    save_message(st.session_state.current_chat_id, "user", query_text)  # Không có sources
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(query_text)
    
    # B. Xử lý Backend
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("🔍 Đang tìm kiếm thông tin..."):
            try:
                result = st.session_state.pipeline.run(query_text, user_id=user_id)
                
                if "error" in result:
                    answer = f"⚠️ {result['error']}"
                    st.error(answer)
                    sources = []
                else:
                    answer = result["answer"]
                    st.markdown(answer)
                    sources = result.get("sources", [])
                    
                    # Hiển thị sources
                    if sources:
                        with st.expander("📚 Nguồn tham khảo"):
                            for i, src in enumerate(sources[:5], 1):
                                s_type = src.get('type', 'text').upper()
                                s_url = src.get('url') or '#'        
                                s_title = src.get('title')
                                if not s_title or s_title == "None":
                                    s_title = src.get('chunk_id', '').replace('-', ' ').replace('_', ' ').title()
                                if not s_title:
                                    s_title = "Tài liệu tuyển sinh"
                                
                                st.markdown(f"**{i}. [{s_type}] {s_title}**\n🔗 [Xem chi tiết]({s_url})")
                    
            except Exception as e:
                answer = f"⚠️ Xin lỗi, hệ thống đang gặp sự cố."
                st.error(f"System Error: {str(e)}")
                sources = []
    
    # C. Lưu vào Session State VÀ Database (KÈM SOURCES)
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })
    
    # Lưu tin nhắn kèm sources vào DB
    save_message(
        st.session_state.current_chat_id, 
        "assistant", 
        answer, 
        sources=sources 
    )
    
    if len(st.session_state.messages) <= 2:
        st.rerun()


# --- 8. MAIN INTERFACE (CONTROLLER) ---

# [MỚI] Điều hướng: Nếu Admin Mode -> Render Dashboard, Ngược lại -> Render Chat
if st.session_state.admin_mode:
    from app.admin_page import render_admin_dashboard
    render_admin_dashboard()

else:
    # --- GIAO DIỆN CHAT CHÍNH (LOGIC CŨ GIỮ NGUYÊN) ---
    st.title("🎓 Chatbot Tư Vấn Tuyển Sinh BDU")
    st.caption("Hệ thống trả lời tự động dựa trên dữ liệu tuyển sinh chính thức.")

    # --- Gợi ý (Empty State) ---
    if not st.session_state.get("messages"):
        st.markdown("### 💡 Những câu hỏi thường gặp")
        
        sample_questions = [
            "📚 Học phí ngành Công nghệ Thông tin năm 2025?",
            "📊 Điểm chuẩn các ngành năm 2025?",
            "🎯 hồ sơ xét tuyển gồm những gì?",
            "💰 Trường có học bổng không?",
            "📍 Địa chỉ và cách liên hệ với trường?",
            "🏫 Trường có những ngành nào?"
        ]
        
        c1, c2, c3 = st.columns(3)
        for i, q in enumerate(sample_questions):
            with [c1, c2, c3][i % 3]:
                clean_q = q.split(" ", 1)[1] if " " in q else q
                if st.button(q, key=f"sample_{i}", use_container_width=True):
                    process_query(clean_q)
                    st.rerun()
        
        st.divider()

    # --- Render Chat History ---
    for msg in st.session_state.get("messages", []):
        avatar = "🎓" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            
            # Render sources nếu message có
            if msg.get("sources"):
                with st.expander("📚 Nguồn tham khảo"):
                    for i, src in enumerate(msg["sources"][:5], 1):
                        s_type = src.get('type', 'text').upper()
                        s_url = src.get('url', '#')
                        s_title = src.get('title', 'Tài liệu')
                        # Fallback cho title nếu rỗng
                        if not s_title or str(s_title).strip() == "":
                             s_title = src.get('chunk_id', 'N/A')
                        st.markdown(f"**{i}. [{s_type}] {s_title}**\n🔗 [Xem chi tiết]({s_url})")

    # --- INPUT BOX ---
    if prompt := st.chat_input("Hỏi gì về tuyển sinh nhé... 💬"):
        process_query(prompt)
        st.rerun()