import streamlit as st
import pandas as pd
import plotly.express as px
from src.admin_backend import get_chat_stats, get_top_keywords, process_uploaded_file

def render_admin_dashboard():
    st.header("🛠️ Trang Quản Trị Hệ Thống")
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 Thống kê & Xu hướng", "📚 Cập nhật Kiến thức"])
    
    # --- TAB 1: THỐNG KÊ ---
    with tab1:
        try:
            stats = get_chat_stats()
            
            # Metrics
            col1, col2 = st.columns(2)
            col1.metric("Tổng cuộc hội thoại", stats['total_conversations'])
            col2.metric("Tổng tin nhắn", stats['total_messages'])
            
            st.divider()
            
            # Biểu đồ từ khóa
            st.subheader("🔥 Chủ đề được quan tâm nhất")
            top_keywords = get_top_keywords()
            if top_keywords:
                df_kw = pd.DataFrame(top_keywords, columns=['Từ khóa', 'Số lần'])
                fig = px.bar(df_kw, x='Từ khóa', y='Số lần', color='Số lần')
                st.plotly_chart(fig, use_container_width=True)
            
            # Bảng câu hỏi gần đây
            st.subheader("💬 Câu hỏi gần đây")
            if not stats['recent_questions'].empty:
                st.dataframe(stats['recent_questions'], use_container_width=True, height=300)
            else:
                st.info("Chưa có dữ liệu câu hỏi.")
                
        except Exception as e:
            st.error(f"Không thể tải thống kê: {str(e)}")

    # --- TAB 2: CẬP NHẬT KIẾN THỨC ---
    with tab2:        
        
        uploaded_file = st.file_uploader("Upload tài liệu mới (PDF, Word, Excel, JSON, Ảnh (PNG/JPG))", type=['pdf', 'docx', 'txt', 'xlsx', 'json', 'png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Nút bấm đơn giản, không cần checkbox chọn model nữa
            if st.button("🚀 Bắt đầu Xử lý & Cập nhật", type="primary"):
                
                with st.status("Đang xử lý dữ liệu...", expanded=True) as status:
                    st.write("1. Đang tải và đọc cấu trúc file...")
                    st.write("2. Đang dùng AI (Llama 4 Scout) quét nội dung & bảng biểu...")
                    st.write("3. Đang cắt nhỏ dữ liệu (Chunking)...")
                    st.write("4. Đang mã hóa và lưu vào Qdrant...")
                    
                    try:
                        # Gọi hàm xử lý từ admin_backend
                        num_chunks = process_uploaded_file(uploaded_file)
                        
                        status.update(label="✅ Hoàn tất!", state="complete", expanded=False)
                        st.success(f"Thành công! Đã thêm **{num_chunks}** phân đoạn kiến thức mới vào bộ nhớ Chatbot.")
                        st.balloons()
                        
                    except Exception as e:
                        status.update(label="❌ Thất bại", state="error")
                        st.error(f"Lỗi xử lý: {str(e)}")