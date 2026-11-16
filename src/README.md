# Code

Toàn bộ bài làm nằm trong [day7.py](day7.py), dùng lại starter: `agentic.run_agent` (vòng lặp agent), `baseline.hit` / `pack_context` / `TfidfIndex`, `convert_day6.convert_chunk`, `agent_metrics.py`. Embedding model và vector store được nạp thẳng từ repo Day 6, không chép sang đây (tài liệu có bản quyền).

| Phần trong `day7.py` | Nội dung |
|---|---|
| Experiment configs | 8 cấu hình retrieval, 4 cấu hình agent (A3 thêm sau phân tích lỗi dev), giả thuyết, tiêu chí chọn, `FROZEN` (R5 + A1) |
| Day 6 | Nạp `starter/src` của Day 6 (alias `day6src`), chunk chiến lược C, dense search, kiểm tra sha của index |
| Benchmark | 20 câu Day 6 (chuyển sang dev), 6 câu test mới, 4 câu thiếu bằng chứng; evidence chép nguyên văn từ chunk |
| Retrievers | TF-IDF (starter), BM25, hybrid RRF, cross-encoder rerank, chunk lân cận |
| Agent | `ClaudeController.decide` (Controller của starter) + `AgentRetriever.search` (Retriever của starter) |

## Chạy

```bash
pip install anthropic==2.6.0 python-dotenv chromadb==1.5.9 sentence-transformers==6.3.0 torch==2.15.0 transformers==5.23.0
```

Tạo `.env` ở thư mục gốc (đã bị gitignore): `ANTHROPIC_API_KEY=...`; tùy chọn `ANTHROPIC_BASE_URL` (proxy), `AGENT_MODEL` (mặc định `claude-opus-5-5`; lần chạy ghi nhận dùng `claude-opus-5`), `DAY6_DIR` (mặc định: repo Day 6 nằm cạnh repo này), `DAY6_VECTOR_STORE` (`chroma` mặc định, hoặc `pgvector`). Repo Day 6 cần có `data/processed/chunks_C.jsonl` và `data/index/chroma`: chạy `python -m src.run --stage ingest` rồi `--stage index` trong `starter/` của Day 6 với `VECTOR_STORE=chroma`.

Chạy từ thư mục gốc. Trên Windows đặt `PYTHONUTF8=1`; khi model đã có trong cache, đặt `HF_HUB_OFFLINE=1` để khỏi chờ mạng.

```bash
python -m unittest discover -s tests -v
python src/day7.py prepare                    # data/questions.json, data/agent_cases.json, starter/data/converted/chunks.json
python starter/baseline.py --chunks starter/data/converted/chunks.json --questions data/questions.json --split dev --output starter/results/dev-tfidf
python src/day7.py retrieval --split dev      # starter/results/retrieval-dev/
python src/day7.py agent --split dev --configs A0_dense,A1_selected,A2_selected_single_search,A3_selected_citefix
# chấm tay: copy reviews_template.json -> reviews.json trong từng thư mục agent, điền nhãn
python starter/agent_metrics.py --cases data/agent_cases.json --traces starter/results/agent-dev-A0_dense/traces.json --reviews starter/results/agent-dev-A0_dense/reviews.json --split dev --output starter/results/agent-dev-A0_dense/summary.json
# sau khi điền FROZEN["agent"] trong day7.py:
python src/day7.py retrieval --split test --configs R1_dense,R5_hybrid_rerank_nb --frozen
python src/day7.py agent --split test --configs A0_dense,A1_selected --frozen
```

## Kết quả

`starter/results/` bị gitignore theo mặc định. File được phép commit (chỉ có ID, điểm, trang và đoạn trích ≤160 ký tự) cần thêm bằng `git add -f`: `retrieval-*/`, `agent-*/traces.json`, `reviews.json`, `summary.json`, `auto_summary.json`, `dev-tfidf/summary.json`. **Không commit** `agent-*/traces_full.json`, `dev-tfidf/traces.json` hay `starter/data/` (chứa toàn văn sách).
