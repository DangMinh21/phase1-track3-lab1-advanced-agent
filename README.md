# Lab 16 - Reflexion Agent

## Thông tin học viên
- Họ và tên: Đặng Văn Minh
- MSHV: 2A202600027

## Giới thiệu chung
Đây là bài lab xây dựng và đánh giá Reflexion Agent trên bài toán QA đa bước (multi-hop QA) với dataset HotpotQA.  
Hệ thống gồm 2 agent để so sánh:
- `ReAct`: 1 lần trả lời.
- `Reflexion`: vòng lặp `answer -> evaluate -> reflect` với nhiều attempt.

Mục tiêu:
- Chạy benchmark trên >=100 mẫu dữ liệu thật.
- Xuất báo cáo đúng schema (`report.json`, `report.md`).
- Tích hợp runtime LLM thật (OpenAI-compatible API) và ghi nhận token/latency thực tế.

## Hướng dẫn chạy và chấm bài cho giảng viên

### 1) Cài đặt môi trường
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Chạy benchmark (mock, kiểm tra nhanh)
```bash
python run_benchmark.py --dataset data/hotpot_120.json --out-dir outputs/score100_mock --mode mock
python autograde.py --report-path outputs/score100_mock/report.json
```

### 3) Chạy benchmark (real, dùng LLM thật)
```bash
export OPENAI_API_KEY="your_api_key"
export LLM_MODEL="gpt-4o-mini"
# Nếu dùng endpoint custom OpenAI-compatible thì mở dòng dưới:
# export LLM_BASE_URL="https://api.openai.com/v1"

python run_benchmark.py --dataset data/hotpot_120.json --out-dir outputs/sample_run --mode real
python autograde.py --report-path outputs/sample_run/report.json
```

### 4) File kết quả cần kiểm tra
- `outputs/.../react_runs.jsonl`
- `outputs/.../reflexion_runs.jsonl`
- `outputs/.../report.json`
- `outputs/.../report.md`

## Những gì đã hoàn thành
- Đã xây dựng runtime LLM thật trong `src/reflexion_lab/llm_runtime.py`.
- Đã hỗ trợ evaluator JSON có retry parser và fallback an toàn.
- Đã tích hợp vòng lặp Reflexion trong `src/reflexion_lab/agents.py`:
  - reflection memory
  - adaptive max attempts
  - memory compression
- Đã chạy benchmark trên `data/hotpot_120.json` (đủ điều kiện >=100 mẫu).
- Đã sinh report đúng schema và pass autograde.
- Đã mở rộng phần `failure_modes` để tăng độ sâu phân tích trong báo cáo.

## Bonus đã làm
### Bonus trong rubric autograde
- `structured_evaluator`
- `reflection_memory`
- `adaptive_max_attempts`
- `memory_compression`
- `benchmark_report_json`

### Bonus mở rộng thêm
- `plan_then_execute`: planner tạo kế hoạch ngắn trước khi actor trả lời.
- `mini_lats_branching`: thêm nhánh suy luận thay thế (light branch) ở attempt tiếp theo.
- `self_consistency_vote`: chọn đáp án theo bỏ phiếu sau khi normalize.
- `reflection_overfit_guard`: dừng sớm khi lặp lại chiến lược và đáp án sai không tiến bộ.

## Cấu trúc mã nguồn chính
- `src/reflexion_lab/agents.py`: logic ReAct/Reflexion và loop attempts.
- `src/reflexion_lab/llm_runtime.py`: runtime gọi OpenAI-compatible API.
- `src/reflexion_lab/prompts.py`: system prompts cho actor/evaluator/reflector/planner.
- `src/reflexion_lab/reporting.py`: tổng hợp metric và xuất report.
- `src/reflexion_lab/schemas.py`: schema dữ liệu và report.
- `run_benchmark.py`: script chạy benchmark.
- `autograde.py`: script chấm điểm tự động.

## Lưu ý cần thiết
- Trong `--mode real`, bắt buộc có `OPENAI_API_KEY`.
- `LLM_BASE_URL` phải có schema `http://` hoặc `https://` nếu được khai báo.
- Điểm autograde là tham chiếu tự động; giảng viên có thể đánh giá thêm chất lượng reasoning và code quality.
