"""
Kịch bản 3: Đánh giá chất lượng hội thoại End-to-End
Đánh giá 3 nhóm câu hỏi: Đơn giản, Phức tạp, Thiếu dữ liệu
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import csv

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import RAGPipeline


class EndToEndEvaluator:
    """Đánh giá chất lượng hội thoại End-to-End"""
    
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.results = []
        
    def create_test_dataset(self) -> Dict[str, List[Dict]]:
        """
        Tạo bộ dữ liệu test cho 3 nhóm câu hỏi
        Mỗi câu hỏi có:
        - question: Câu hỏi
        - expected_keywords: Các từ khóa kỳ vọng trong câu trả lời
        - expected_source_type: Loại nguồn kỳ vọng (database/web_search)
        - difficulty: Độ khó
        """
        
        test_data = {
            # ============================================
            # NHÓM 1: CÂU HỎI ĐƠN GIẢN
            # Thông tin có sẵn trực tiếp trong database
            # ============================================
            "simple": [
                {
                    "id": "S01",
                    "question": "Học phí ngành Công nghệ thông tin là bao nhiêu?",
                    "expected_keywords": ["học phí", "CNTT", "công nghệ thông tin", "triệu", "đồng"],
                    "expected_source_type": "database",
                    "category": "học phí"
                },
                {
                    "id": "S02",
                    "question": "Địa chỉ trường Đại học Bình Dương ở đâu?",
                    "expected_keywords": ["504", "Bình Dương", "địa chỉ", "Phú Lợi"],
                    "expected_source_type": "database",
                    "category": "liên hệ"
                },
                {
                    "id": "S03",
                    "question": "Hotline tuyển sinh của trường là gì?",
                    "expected_keywords": ["0789", "hotline", "tuyển sinh", "điện thoại"],
                    "expected_source_type": "database",
                    "category": "liên hệ"
                },
                {
                    "id": "S04",
                    "question": "Trường có những ngành đào tạo nào?",
                    "expected_keywords": ["ngành", "đào tạo", "CNTT", "kinh tế", "luật"],
                    "expected_source_type": "database",
                    "category": "ngành học"
                },
                {
                    "id": "S05",
                    "question": "Điểm chuẩn ngành Luật năm 2024 là bao nhiêu?",
                    "expected_keywords": ["điểm", "chuẩn", "luật", "2024"],
                    "expected_source_type": "database",
                    "category": "điểm chuẩn"
                },
                {
                    "id": "S06",
                    "question": "Hồ sơ xét tuyển gồm những gì?",
                    "expected_keywords": ["hồ sơ", "xét tuyển", "giấy tờ", "đăng ký"],
                    "expected_source_type": "database",
                    "category": "hồ sơ"
                },
                {
                    "id": "S07",
                    "question": "Trường có học bổng không?",
                    "expected_keywords": ["học bổng", "hỗ trợ", "miễn giảm"],
                    "expected_source_type": "database",
                    "category": "học bổng"
                },
                {
                    "id": "S08",
                    "question": "Thời gian tuyển sinh năm 2025 khi nào?",
                    "expected_keywords": ["thời gian", "tuyển sinh", "2025", "đợt"],
                    "expected_source_type": "database",
                    "category": "lịch tuyển sinh"
                },
                {
                    "id": "S09",
                    "question": "Mã ngành Công nghệ thông tin là gì?",
                    "expected_keywords": ["mã ngành", "CNTT", "7480201"],
                    "expected_source_type": "database",
                    "category": "ngành học"
                },
                {
                    "id": "S10",
                    "question": "Email liên hệ tuyển sinh là gì?",
                    "expected_keywords": ["email", "tuyensinh", "bdu.edu.vn"],
                    "expected_source_type": "database",
                    "category": "liên hệ"
                }
            ],
            
            # ============================================
            # NHÓM 2: CÂU HỎI PHỨC TẠP
            # Cần tổng hợp nhiều nguồn hoặc so sánh
            # ============================================
            "complex": [
                {
                    "id": "C01",
                    "question": "So sánh điểm chuẩn ngành Luật và ngành Kinh tế năm 2024?",
                    "expected_keywords": ["luật", "kinh tế", "điểm", "so sánh"],
                    "expected_source_type": "database",
                    "category": "so sánh"
                },
                {
                    "id": "C02",
                    "question": "Học phí và điểm chuẩn ngành CNTT năm 2024?",
                    "expected_keywords": ["học phí", "điểm chuẩn", "CNTT", "2024"],
                    "expected_source_type": "database",
                    "category": "tổng hợp"
                },
                {
                    "id": "C03",
                    "question": "Tôi có 18 điểm thì đậu được những ngành nào và học phí ra sao?",
                    "expected_keywords": ["18 điểm", "ngành", "học phí", "đậu"],
                    "expected_source_type": "database",
                    "category": "tư vấn"
                },
                {
                    "id": "C04",
                    "question": "Ngành nào có điểm chuẩn thấp nhất và cao nhất năm 2024?",
                    "expected_keywords": ["điểm chuẩn", "thấp nhất", "cao nhất", "2024"],
                    "expected_source_type": "database",
                    "category": "so sánh"
                },
                {
                    "id": "C05",
                    "question": "Trường có những phương thức xét tuyển nào và điều kiện từng phương thức?",
                    "expected_keywords": ["phương thức", "xét tuyển", "điều kiện", "học bạ", "thi"],
                    "expected_source_type": "database",
                    "category": "tổng hợp"
                },
                {
                    "id": "C06",
                    "question": "So sánh học phí giữa các ngành khối kinh tế?",
                    "expected_keywords": ["học phí", "so sánh", "kinh tế", "ngành"],
                    "expected_source_type": "database",
                    "category": "so sánh"
                },
                {
                    "id": "C07",
                    "question": "Nếu tôi thích công nghệ, nên chọn ngành CNTT hay Trí tuệ nhân tạo?",
                    "expected_keywords": ["CNTT", "trí tuệ nhân tạo", "tư vấn", "công nghệ"],
                    "expected_source_type": "database",
                    "category": "tư vấn"
                },
                {
                    "id": "C08",
                    "question": "Liệt kê các ngành có mức học phí dưới 20 triệu/năm?",
                    "expected_keywords": ["ngành", "học phí", "triệu", "dưới"],
                    "expected_source_type": "database",
                    "category": "lọc"
                },
                {
                    "id": "C09",
                    "question": "Quy trình nộp hồ sơ online và offline khác nhau như thế nào?",
                    "expected_keywords": ["hồ sơ", "online", "offline", "quy trình"],
                    "expected_source_type": "database",
                    "category": "so sánh"
                },
                {
                    "id": "C10",
                    "question": "Điểm chuẩn năm 2024 thay đổi như thế nào so với 2023?",
                    "expected_keywords": ["điểm chuẩn", "2024", "2023", "thay đổi"],
                    "expected_source_type": "database",
                    "category": "so sánh"
                }
            ],
            
            # ============================================
            # NHÓM 3: CÂU HỎI THIẾU DỮ LIỆU
            # Kiểm tra khả năng tự động tìm kiếm Web
            # ============================================
            "out_of_scope": [
                {
                    "id": "O01",
                    "question": "Tin tức tuyển sinh mới nhất hôm nay của trường?",
                    "expected_keywords": ["tin tức", "tuyển sinh", "mới"],
                    "expected_source_type": "web_search",
                    "category": "tin tức"
                },
                {
                    "id": "O02",
                    "question": "Trường Đại học Bình Dương xếp hạng thứ mấy ở Việt Nam?",
                    "expected_keywords": ["xếp hạng", "Việt Nam", "đại học"],
                    "expected_source_type": "web_search",
                    "category": "xếp hạng"
                },
                {
                    "id": "O03",
                    "question": "Cơ hội việc làm sau khi tốt nghiệp ngành CNTT tại BDU?",
                    "expected_keywords": ["việc làm", "tốt nghiệp", "cơ hội"],
                    "expected_source_type": "web_search",
                    "category": "việc làm"
                },
                {
                    "id": "O04",
                    "question": "Đánh giá của sinh viên về trường Đại học Bình Dương?",
                    "expected_keywords": ["đánh giá", "sinh viên", "review"],
                    "expected_source_type": "web_search",
                    "category": "đánh giá"
                },
                {
                    "id": "O05",
                    "question": "Ký túc xá trường có giá bao nhiêu một tháng?",
                    "expected_keywords": ["ký túc xá", "giá", "tháng"],
                    "expected_source_type": "web_search",
                    "category": "sinh hoạt"
                },
                {
                    "id": "O06",
                    "question": "Trường có chương trình trao đổi sinh viên quốc tế không?",
                    "expected_keywords": ["trao đổi", "quốc tế", "sinh viên"],
                    "expected_source_type": "web_search",
                    "category": "quốc tế"
                },
                {
                    "id": "O07",
                    "question": "Lương trung bình của sinh viên BDU sau khi ra trường?",
                    "expected_keywords": ["lương", "trung bình", "ra trường"],
                    "expected_source_type": "web_search",
                    "category": "việc làm"
                },
                {
                    "id": "O08",
                    "question": "Các câu lạc bộ sinh viên tại trường?",
                    "expected_keywords": ["câu lạc bộ", "sinh viên", "hoạt động"],
                    "expected_source_type": "web_search",
                    "category": "sinh hoạt"
                },
                {
                    "id": "O09",
                    "question": "Thời tiết ở Bình Dương hôm nay thế nào?",
                    "expected_keywords": [],  # Ngoài phạm vi hoàn toàn
                    "expected_source_type": "reject",
                    "category": "ngoài phạm vi"
                },
                {
                    "id": "O10",
                    "question": "Cho tôi công thức nấu phở?",
                    "expected_keywords": [],  # Ngoài phạm vi hoàn toàn
                    "expected_source_type": "reject",
                    "category": "ngoài phạm vi"
                }
            ]
        }
        
        return test_data
    
    def evaluate_single_response(
        self, 
        question_data: Dict, 
        response: Dict,
        group: str
    ) -> Dict[str, Any]:
        """
        Đánh giá một response đơn lẻ
        
        Metrics:
        - keyword_coverage: % từ khóa kỳ vọng xuất hiện trong câu trả lời
        - source_accuracy: Nguồn có đúng loại kỳ vọng không
        - response_time: Thời gian phản hồi
        - has_sources: Có trích dẫn nguồn không
        - answer_length: Độ dài câu trả lời
        """
        
        answer = response.get("answer", "").lower()
        sources = response.get("sources", [])
        timing = response.get("timing", {})
        
        # 1. Keyword Coverage
        expected_keywords = question_data.get("expected_keywords", [])
        if expected_keywords:
            matched = sum(1 for kw in expected_keywords if kw.lower() in answer)
            keyword_coverage = matched / len(expected_keywords)
        else:
            keyword_coverage = 1.0 if question_data["expected_source_type"] == "reject" else 0.0
        
        # 2. Source Type Accuracy
        expected_source = question_data.get("expected_source_type", "database")
        
        if expected_source == "reject":
            # Kỳ vọng chatbot từ chối trả lời
            reject_phrases = [
                "không thể", "ngoài phạm vi", "chỉ có thể tư vấn", 
                "không tìm thấy", "không có thông tin"
            ]
            source_accuracy = 1.0 if any(p in answer for p in reject_phrases) else 0.0
        elif expected_source == "web_search":
            # Kiểm tra có dùng web search không
            web_sources = [s for s in sources if s.get("type") == "web_search"]
            source_accuracy = 1.0 if web_sources else 0.5
        else:
            # Database sources
            db_sources = [s for s in sources if s.get("type") != "web_search"]
            source_accuracy = 1.0 if db_sources else 0.0
        
        # 3. Response Time
        response_time = timing.get("total", 0)
        
        # 4. Has Sources
        has_sources = len(sources) > 0
        
        # 5. Answer Length (đánh giá độ chi tiết)
        answer_length = len(answer.split())
        
        # 6. Overall Score (weighted)
        if group == "simple":
            # Câu đơn giản: ưu tiên chính xác và nhanh
            overall_score = (
                keyword_coverage * 0.5 +
                source_accuracy * 0.3 +
                (1.0 if response_time < 5 else 0.5) * 0.2
            )
        elif group == "complex":
            # Câu phức tạp: ưu tiên đầy đủ thông tin
            overall_score = (
                keyword_coverage * 0.4 +
                source_accuracy * 0.2 +
                (1.0 if answer_length > 50 else 0.5) * 0.2 +
                (1.0 if has_sources else 0.0) * 0.2
            )
        else:  # out_of_scope
            # Câu ngoài phạm vi: ưu tiên xử lý đúng
            overall_score = (
                source_accuracy * 0.6 +
                keyword_coverage * 0.4
            )
        
        return {
            "question_id": question_data["id"],
            "question": question_data["question"],
            "category": question_data["category"],
            "group": group,
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
            "metrics": {
                "keyword_coverage": round(keyword_coverage, 3),
                "source_accuracy": round(source_accuracy, 3),
                "response_time": round(response_time, 3),
                "has_sources": has_sources,
                "num_sources": len(sources),
                "answer_length": answer_length,
                "overall_score": round(overall_score, 3)
            },
            "sources_used": [
                {"type": s.get("type"), "title": s.get("title", "N/A")[:50]} 
                for s in sources[:3]
            ]
        }
    
    def run_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Chạy đánh giá toàn bộ
        """
        print("=" * 70)
        print("KỊCH BẢN 3: ĐÁNH GIÁ CHẤT LƯỢNG HỘI THOẠI END-TO-END")
        print("=" * 70)
        
        test_data = self.create_test_dataset()
        all_results = []
        group_stats = {}
        
        for group_name, questions in test_data.items():
            print(f"\n{'='*60}")
            print(f"NHÓM: {group_name.upper()} ({len(questions)} câu hỏi)")
            print("=" * 60)
            
            group_results = []
            
            for i, q_data in enumerate(questions, 1):
                print(f"\n[{i}/{len(questions)}] {q_data['id']}: {q_data['question'][:50]}...")
                
                try:
                    # Chạy pipeline
                    start_time = time.time()
                    response = self.pipeline.run(q_data["question"], user_id="evaluator")
                    
                    # Đánh giá response
                    eval_result = self.evaluate_single_response(q_data, response, group_name)
                    group_results.append(eval_result)
                    
                    # In kết quả nhanh
                    metrics = eval_result["metrics"]
                    print(f"   ✓ Score: {metrics['overall_score']:.2f} | "
                          f"Keywords: {metrics['keyword_coverage']:.2f} | "
                          f"Time: {metrics['response_time']:.2f}s | "
                          f"Sources: {metrics['num_sources']}")
                    
                except Exception as e:
                    print(f"   ✗ ERROR: {str(e)}")
                    group_results.append({
                        "question_id": q_data["id"],
                        "question": q_data["question"],
                        "group": group_name,
                        "error": str(e),
                        "metrics": {
                            "keyword_coverage": 0,
                            "source_accuracy": 0,
                            "response_time": 0,
                            "overall_score": 0
                        }
                    })
                
                # Delay để tránh rate limit
                time.sleep(1)
            
            # Tính thống kê nhóm
            valid_results = [r for r in group_results if "error" not in r]
            
            if valid_results:
                group_stats[group_name] = {
                    "total_questions": len(questions),
                    "successful": len(valid_results),
                    "failed": len(questions) - len(valid_results),
                    "avg_overall_score": sum(r["metrics"]["overall_score"] for r in valid_results) / len(valid_results),
                    "avg_keyword_coverage": sum(r["metrics"]["keyword_coverage"] for r in valid_results) / len(valid_results),
                    "avg_source_accuracy": sum(r["metrics"]["source_accuracy"] for r in valid_results) / len(valid_results),
                    "avg_response_time": sum(r["metrics"]["response_time"] for r in valid_results) / len(valid_results),
                    "with_sources_rate": sum(1 for r in valid_results if r["metrics"]["has_sources"]) / len(valid_results)
                }
            
            all_results.extend(group_results)
        
        # Tổng hợp kết quả
        final_report = {
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(all_results),
            "group_statistics": group_stats,
            "detailed_results": all_results,
            "summary": self._generate_summary(group_stats)
        }
        
        # Lưu kết quả
        if save_results:
            self._save_results(final_report)
        
        # In báo cáo
        self._print_report(final_report)
        
        return final_report
    
    def _generate_summary(self, group_stats: Dict) -> Dict:
        """Tạo tóm tắt đánh giá"""
        
        if not group_stats:
            return {"status": "No data"}
        
        overall_score = sum(
            g["avg_overall_score"] * g["successful"] 
            for g in group_stats.values()
        ) / sum(g["successful"] for g in group_stats.values())
        
        return {
            "overall_score": round(overall_score, 3),
            "best_group": max(group_stats.items(), key=lambda x: x[1]["avg_overall_score"])[0],
            "worst_group": min(group_stats.items(), key=lambda x: x[1]["avg_overall_score"])[0],
            "avg_response_time": round(
                sum(g["avg_response_time"] for g in group_stats.values()) / len(group_stats), 2
            ),
            "recommendation": self._get_recommendation(group_stats)
        }
    
    def _get_recommendation(self, group_stats: Dict) -> str:
        """Đưa ra khuyến nghị dựa trên kết quả"""
        
        recommendations = []
        
        if group_stats.get("simple", {}).get("avg_overall_score", 0) < 0.7:
            recommendations.append("Cần cải thiện độ chính xác cho câu hỏi đơn giản - kiểm tra lại indexing")
        
        if group_stats.get("complex", {}).get("avg_overall_score", 0) < 0.6:
            recommendations.append("Cần cải thiện khả năng tổng hợp thông tin - xem xét Query Decomposition")
        
        if group_stats.get("out_of_scope", {}).get("avg_source_accuracy", 0) < 0.5:
            recommendations.append("Cần cải thiện Web Search fallback hoặc xử lý câu hỏi ngoài phạm vi")
        
        avg_time = sum(g.get("avg_response_time", 0) for g in group_stats.values()) / len(group_stats)
        if avg_time > 10:
            recommendations.append("Thời gian phản hồi chậm - xem xét tối ưu hóa hoặc caching")
        
        return " | ".join(recommendations) if recommendations else "Hệ thống hoạt động tốt"
    
    def _save_results(self, report: Dict):
        """Lưu kết quả ra file"""
        
        # Tạo thư mục output
        output_dir = Path("evaluation/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Lưu JSON đầy đủ
        json_path = output_dir / f"e2e_evaluation_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📁 Saved JSON report: {json_path}")
        
        # Lưu CSV tóm tắt
        csv_path = output_dir / f"e2e_evaluation_{timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ID", "Group", "Category", "Question", 
                "Overall Score", "Keyword Coverage", "Source Accuracy",
                "Response Time", "Num Sources"
            ])
            
            for r in report["detailed_results"]:
                m = r.get("metrics", {})
                writer.writerow([
                    r.get("question_id", ""),
                    r.get("group", ""),
                    r.get("category", ""),
                    r.get("question", "")[:100],
                    m.get("overall_score", 0),
                    m.get("keyword_coverage", 0),
                    m.get("source_accuracy", 0),
                    m.get("response_time", 0),
                    m.get("num_sources", 0)
                ])
        
        print(f"📁 Saved CSV report: {csv_path}")
    
    def _print_report(self, report: Dict):
        """In báo cáo đánh giá"""
        
        print("\n" + "=" * 70)
        print("BÁO CÁO ĐÁNH GIÁ END-TO-END")
        print("=" * 70)
        
        print(f"\n📅 Thời gian: {report['evaluation_date']}")
        print(f"📊 Tổng số câu hỏi: {report['total_questions']}")
        
        print("\n" + "-" * 50)
        print("THỐNG KÊ THEO NHÓM")
        print("-" * 50)
        
        for group_name, stats in report["group_statistics"].items():
            group_label = {
                "simple": "Câu hỏi ĐƠN GIẢN",
                "complex": "Câu hỏi PHỨC TẠP",
                "out_of_scope": "Câu hỏi THIẾU DỮ LIỆU"
            }.get(group_name, group_name)
            
            print(f"\n📌 {group_label}")
            print(f"   • Số câu: {stats['successful']}/{stats['total_questions']}")
            print(f"   • Điểm TB: {stats['avg_overall_score']:.2%}")
            print(f"   • Keyword Coverage: {stats['avg_keyword_coverage']:.2%}")
            print(f"   • Source Accuracy: {stats['avg_source_accuracy']:.2%}")
            print(f"   • Thời gian TB: {stats['avg_response_time']:.2f}s")
            print(f"   • Có nguồn: {stats['with_sources_rate']:.2%}")
        
        summary = report["summary"]
        print("\n" + "-" * 50)
        print("TÓM TẮT")
        print("-" * 50)
        print(f"🎯 Điểm tổng thể: {summary['overall_score']:.2%}")
        print(f"✅ Nhóm tốt nhất: {summary['best_group']}")
        print(f"⚠️  Nhóm cần cải thiện: {summary['worst_group']}")
        print(f"⏱️  Thời gian TB: {summary['avg_response_time']}s")
        print(f"\n💡 Khuyến nghị: {summary['recommendation']}")
        
        print("\n" + "=" * 70)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("🚀 Khởi tạo hệ thống đánh giá...")
    
    # Load pipeline
    from sentence_transformers import SentenceTransformer
    
    print("📦 Loading embedding model...")
    embedding_model = SentenceTransformer("google/embeddinggemma-300m")
    
    print("🔧 Loading RAG pipeline...")
    pipeline = RAGPipeline(
        model_type="gemma",
        verbose=False,
        preloaded_model=embedding_model
    )
    
    # Chạy đánh giá
    evaluator = EndToEndEvaluator(pipeline)
    results = evaluator.run_evaluation(save_results=True)
    
    print("\n✅ Đánh giá hoàn tất!")
