# Báo cáo Day 7

> Nhãn chấm câu trả lời (`reviews.json`) và nhãn context 0/1/2 do **Claude (AI) soạn và kiểm tra lại 2 lượt** (lượt 2 sửa citation của A0 dev q20 thành `false`), có lý do cho từng câu. Theo BENCHMARK.md, nhãn do model kiểm tra là chưa đủ: học viên cần đọc và xác nhận trước khi nộp. Benchmark cũng được soạn với hỗ trợ của AI. Lệnh chạy: [src/README.md](../../src/README.md).

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** `harrypotter.pdf` (Pottermore), sách 1–3, 914 trang nội dung, sha256 khớp manifest của Day 6. Sách có bản quyền; giảng viên chấp nhận dùng cho lab; không commit PDF, text, chunk hay index. Dùng **1800 chunk của chiến lược C** đã chọn ở Day 6 (chia theo đoạn, ≤1000 ký tự; chunk có thể vắt trang, nên lưu `page..page_end`). Trong lúc chạy agent, các đoạn trích được gửi tới model qua proxy bên thứ ba `api.xah.io` (xem phần Giới hạn).
- **Vector store, embedding model, phiên bản và số chiều:** `intfloat/multilingual-e5-small`, revision `614241f6`, 384 chiều, tiền tố `query:`/`passage:`, chuẩn hóa L2. Chroma 1.5.9, HNSW cosine, `ef_search` 200, collection `hp_C_para1000_filter`. Code Day 6 được nạp trực tiếp từ repo Day 6, không chép sang. Lệnh `prepare` kiểm tra 1800/1800 vector có `text_sha1` khớp chunk. Day 6 báo cáo trên pgvector, còn Day 7 dùng Chroma vì máy không chạy Docker; dense top-3 trên 20 câu Day 6 cho **cùng kết quả với Day 6 (Hit@1 7/20, Hit@3 9/20)**, nên việc đổi store không làm lệch số liệu.
- **Agent model, cách chạy và phân chia dev/test:** `claude-opus-5` (qua proxy, `ANTHROPIC_BASE_URL`), adaptive thinking, effort `medium`, bật server-side refusal fallback. Dev gồm 20 câu Day 6 và 2 câu thiếu bằng chứng; test gồm 6 câu mới và 2 câu thiếu bằng chứng. **Lý do đổi split:** báo cáo Day 6 ghi rằng điểm test đã hiện ra lúc chọn chiến lược, nên test cũ không còn độc lập. Test mới lấy từ 6 chương mà không câu Day 6 nào dùng: B1 ch9, B1 ch14, B2 ch12, B3 ch9, B3 ch10, B3 ch11. Dữ liệu: [data/questions.json](../../data/questions.json), [data/agent_cases.json](../../data/agent_cases.json). sha256 của hai file là `483aef45…`, được chốt trước mọi lần chạy retrieval/agent.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Evidence** là nguyên văn một đoạn trong một chunk. **Hit** = cùng nguồn, trang gold nằm trong `page..page_end`, evidence là substring của chunk (`starter/baseline.py:hit`, đã mở rộng để hỗ trợ `page_end`).
  - **Sửa evidence Day 6:** 10 quote lệch dấu nháy cong/thẳng hoặc gạch ngang nên được tìm sau chuẩn hóa rồi chép lại nguyên văn từ chunk; quote q13 vắt qua hai chunk nên giữ nửa chứa đáp án.
  - **Câu thiếu bằng chứng** gần chủ đề nhưng nằm ngoài sách 1–3: Triwizard, em trai Dumbledore, Blast-Ended Skrewt, mật khẩu năm 4 (bẫy, vì mật khẩu năm 1–3 có trong tài liệu). Mỗi câu được kiểm tra bằng regex trên toàn bộ 1800 chunk (0 kết quả).
  - **Loại câu dev/test:** trực tiếp 5/1, diễn đạt lại 7/2, điều kiện 3/2, có filter 5/1, thiếu bằng chứng 2/2.
  - **Tiêu chí chọn** được ghi trong `SELECTION_RULE` của [src/day7.py](../../src/day7.py) trước khi chạy.

## Retrieval và thử nghiệm

Tất cả số liệu tính trên **20 câu dev có đáp án** (câu thiếu bằng chứng không tính vào mẫu số), cùng corpus, cùng index, cùng nhãn, dùng **query gốc**. Filter `book_no` chỉ dùng khi câu hỏi có sẵn ngữ cảnh do người dùng cung cấp (5/20 câu). Context budget tính bằng ký tự của khối context (kể cả header), chọn bằng `starter/baseline.py:pack_context` (loại trùng, đoạn không vừa thì bỏ). Thời gian là median mỗi query trên CPU; retrieval không có chi phí API. Trace từng câu: `starter/results/retrieval-dev/`.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline R1 | Dense Day 6, top-3, filter, 4000 ký tự | Cấu hình C của Day 6 | dev | 7/20 (35%) | 9/20 (45%) | 9/20 (45%) | 60 ms / $0 |
| R1b (đối chứng) | Như R1, không filter | Filter đóng góp bao nhiêu? | dev | 7/20 | 9/20 | 9/20 | 65 ms / $0 |
| R0 | TF-IDF của starter, top-3, filter | Lexical đơn giản | dev | 6/20 (30%) | 6/20 (30%) | 6/20 (30%) | 22 ms / $0 |
| Phương pháp 2: R2 | BM25 (k1 1.2, b 0.75, bỏ stopword) | Tên riêng hiếm khớp từ khóa tốt hơn | dev | 2/20 (10%) | 10/20 (50%) | 10/20 (50%) | 7 ms / $0 |
| Phương pháp 3: R3 | Hybrid RRF dense + BM25 (k 60, độ sâu 20) | Hai phương pháp sai ở các câu khác nhau | dev | 5/20 (25%) | 10/20 (50%) | 10/20 (50%) | 92 ms / $0 |
| Phương pháp 4 / Cải thiện 1: R4 | R3 + rerank cross-encoder `ms-marco-MiniLM-L-6-v2` trên top-20 | Evidence có trong top-20 nhưng không ở top-3 | dev | **8/20 (40%)** | **12/20 (60%)** | 12/20 (60%) | 3965 ms / $0 |
| Cải thiện 2: R5 | R4 + chunk liền trước/sau, 8000 ký tự | Evidence nằm ở ranh giới chunk | dev | 8/20 | 12/20 | **17/20 (85%)** | 3918 ms / $0 · 6815 ký tự |
| Cải thiện 3: R6 | Dense top-5, 6000 ký tự | Rẻ nhất: chỉ tăng k | dev | 7/20 | 9/20 | 13/20 (65%) | 83 ms / $0 · 4522 ký tự |
| Cấu hình đã chọn: R5 | Như trên, chạy một lần sau khi chốt | | test | 3/6 (50%) | 4/6 (67%) | 4/6 (67%) | 3263 ms / $0 · 7524 ký tự |
| Baseline: R1 | Như trên | | test | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 180 ms / $0 · 2571 ký tự |

**Chấm tay context trên test** (0 = không có evidence, 1 = có nhưng thiếu điều kiện/nguồn, 2 = đủ; đọc cả 6 câu có đáp án; nhãn và lý do trong `starter/results/retrieval-test/context_labels.json`):

| Câu | R1 | R5 | Lý do |
|---|---|---|---|
| t01 Remembrall | 2 | 2 | Chunk vàng: "if it turns red… glowed scarlet" |
| t02 trứng rồng | 2 | 2 | Chunk vàng: "Won it… game o' cards with a stranger" |
| t03 Marauder's Map | 0 | 1 | R5 chỉ có "wiped the Map blank again", không nói cách xóa |
| t04 Firebolt | 1 | 2 | R1: "đang kiểm tra lời nguyền, xong sẽ báo"; R5 có câu "jinx-free" |
| t05 Quidditch (filter sách 3) | 0 | 2 | R1 không có đoạn về Hufflepuff |
| t06 Polyjuice | 0 | 0 | Cả hai chỉ lấy đoạn công thức pha chế, không có "cat hair" |
| **Số câu mức 2/1/0** | **2/1/3** | **4/1/1** | |

Starter `baseline.py` chạy nguyên bản (top-3, budget mặc định 800 ký tự) cho dev Hit@1 5/20, Hit@3 6/20, context 2/20, vì 800 ký tự không chứa nổi một chunk 1000 ký tự (`starter/results/dev-tfidf/summary.json`).

R4 vừa là phương pháp mới vừa là một cải thiện; R5 và R6 chỉ đổi cách chọn context nên Hit@k giữ nguyên. **Chọn R5** theo tiêu chí đã công bố: Context Evidence Rate dev cao nhất. Đánh đổi: context gấp khoảng 2.5 lần và độ trễ khoảng 3–4 giây/query (median dev 3.9 s, test 3.3 s; cross-encoder trên CPU).

**Phân tích lỗi retrieval trên dev** (trace `starter/results/retrieval-dev/*.json`):

1. **q04, q08, q11, q16, q17: đúng trang nhưng thiếu evidence.** Evidence nằm ở chunk liền kề với chunk top-3. *Giả thuyết:* thêm chunk lân cận. *Kết quả:* cả 5 câu có evidence trong context ở R5 (12/20 → 17/20; Hit@k không đổi).
2. **q02, q18: dense trượt.** Câu hỏi diễn đạt lại khiến e5-small không đưa evidence lên top-3. Hybrid + rerank đưa cả hai lên hạng 1. BM25 có Hit@3 10/20 nhưng Hit@1 chỉ 2/20: tìm đúng vùng nhưng xếp hạng kém; rerank sửa được điểm này.
3. **q06: rerank làm hại.** Hybrid R3 xếp evidence (Time-Turner, chunk p.905–906) hạng 1; cross-encoder đẩy nó ra khỏi top-3 để ưu tiên một đoạn gần nghĩa ở p.838 ("how she was managing to attend several classes"). Cái giá của reranker huấn luyện trên dữ liệu web.
4. **q13: lệch từ vựng.** Chunk vàng (*“Caput Draconis,” said Percy…*) không chứa "password" hay "Gryffindor Tower"; chữ "Password?" nằm ở chunk trước. Trong phạm vi sách 1, chunk này không có trong top-100 của cả dense lẫn BM25.
5. **q20:** evidence xếp hạng 14 ở dense và không có trong top-100 của BM25; "Mirror of Erised" xuất hiện ở rất nhiều đoạn.
6. **Filter cần nhưng chưa đủ** (R1 = R1b). Ở q13, q16, q17, top-3 khi không filter lẫn đoạn của cuốn khác (sách [3,2,2], [2,1,1], [3,1,3]). Filter đưa kết quả về đúng cuốn, nhưng evidence vẫn không vào top-3, nên lỗi còn lại là lỗi xếp hạng.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Vòng lặp:** dùng `agentic.run_agent` của starter. `ClaudeController.decide` (Controller của starter) gọi `claude-opus-5` một lần cho mỗi quyết định, với 2 tool `strict`: `search_book(query)` và `submit_answer(status, answer, citations)`. Mỗi lần gọi, model thấy câu hỏi, các query đã dùng và context tích lũy mới nhất; `run_agent` chỉ gửi bản mới nhất, nên budget không bị nhân lên.
  - **Retriever:** `AgentRetriever.search` (Retriever của starter) trả top-3 của retriever đã cấu hình, cộng chunk lân cận nếu cấu hình bật.
  - **Điều kiện dừng:** model gọi `submit_answer`, hoặc vượt `max_searches` (3; A2 là 1). Khi vượt, `run_agent` chuyển thành từ chối (`search_limit`).
  - **Context budget:** tính bằng ký tự, dùng chung cho mọi lượt search: A0 9000; A1–A3 18000, vì với chunk lân cận một lượt đã thêm khoảng 6800 ký tự, và cần chỗ cho 2–3 lượt. `pack_context` giữ ưu tiên cho đoạn của lượt trước.
  - **Filter:** chỉ lấy từ ngữ cảnh do người dùng cung cấp; model không tự chọn filter.
  - **Citation:** phải nằm trong context **và** xuất hiện dạng `[id]` trong câu trả lời (luật của starter). Không còn citation hợp lệ thì chuyển thành từ chối.
  - **Xử lý lỗi:** SDK tự thử lại; thêm chờ 10/30/60 giây cho lỗi mạng, 429 và 5xx. Refusal hoặc `max_tokens` được ghi `status: error`. Không bao giờ thay bằng kết quả giả. Lần chạy thử đầu tiên có 2 lỗi kết nối do DNS tới proxy chập chờn (đã ghi lại, không thuộc kết quả báo cáo); các lượt chạy chính có 0 lỗi provider.

Nhãn chấm tay: `reviews.json` trong từng thư mục `starter/results/agent-<split>-<config>/`, tổng hợp bằng `starter/agent_metrics.py` → `summary.json`. Đọc **toàn bộ** 22 câu dev × 4 cấu hình và 8 câu test × 2 cấu hình. Chi phí tính theo giá niêm yết `claude-opus-5` ($5/$25 mỗi triệu token) với số token proxy báo; tổng mọi lượt chạy khoảng **$1.80**.

| Cấu hình / split | Tool use | Số lượt search (1/2/3) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Context Evidence Rate | Chi phí/câu · median |
|---|---|---|---|---|---|---|---|---|---|
| A0_dense / dev | 22/22 | 17/0/5 | 3/1/16 · 1.65 | 19/21 | 18/18 | 2/2 | 2/20 | 16/20 | $0.012 · 11.3 s |
| A1_selected / dev | 22/22 | 17/3/2 | 1/0/19 · **1.90** | 20/20 | 19/19 | 2/2 | 1/20 | **20/20** | $0.019 · 8.8 s |
| A2_single_search / dev | 22/22 | 22/0/0 | 3/0/17 · 1.70 | 18/18 | 17/17 | 2/2 | 3/20 | 18/20 | $0.014 · 8.5 s |
| A3_citefix / dev | 22/22 | 17/3/2 | 1/0/19 · 1.90 | 19/20 | 20/20 | 2/2 | 0/20 | 19/20 | $0.019 · 12.5 s |
| A0_dense (baseline) / test | 8/8 | 4/0/4 | 0/0/6 · 2.00 | 7/7 | 6/6 | 2/2 | 0/6 | 6/6 | $0.017 · 19.0 s |
| **A1_selected (đã chọn) / test** | 8/8 | 5/0/3 | 0/0/6 · 2.00 | 8/8 | 6/6 | 2/2 | 0/6 | 6/6 | $0.028 · 18.1 s |

**Chọn agent** theo `SELECTION_RULE`: số câu đúng và có citation đầy đủ A1 = A3 = 19 > A2 = 17 > A0 = 16 (cả bốn từ chối đúng 2/2); hòa thì xét Context Evidence Rate, A1 20/20 > A3 19/20, nên **chọn A1**. Chốt trong `FROZEN` trước khi chạy test; test chỉ chạy một lần, cho A0 và A1.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t04 (test, A1) | Điều kiện | 1 lần search "McGonagall condition give Harry his Firebolt back…" → R5 + lân cận, có chunk vàng p.766 | "once it had been checked and they were sure it was jinx-free — that it had not been tampered with" [C-b3-c11-0037] [C-b3-c11-0038] | **2**: cả hai citation đều hỗ trợ |
| u04 (test, A0 và A1) | Thiếu bằng chứng (bẫy) | 3 lần search (mật khẩu, Fat Lady, năm 4) | Từ chối. A0 thấy "Flibbertigibbet" trong context nhưng nêu rõ không có gì xác nhận đó là mật khẩu đầu năm 4 | **Từ chối đúng** |
| q13 (dev, A3) | Có filter | 2 lần search, không lấy được chunk vàng; có C-b1-c09-0022 ("the **new** password… Pig snout") | "The password was 'Pig snout.'" [C-b1-c09-0022] | **0**: bỏ qua điều kiện thời gian; citation không hỗ trợ "đầu năm" |
| q14 (dev, A0) | Có filter | 1 lần search, chunk vàng có trong context | Model trả lời đúng (Áo choàng tàng hình + nội dung thiệp) nhưng chỉ liệt kê citation, không đặt `[id]` trong câu → `run_agent` chuyển thành từ chối | **0**: lỗi định dạng citation |
| q20 (dev, A0) | Điều kiện | 3 lần search, không lấy được p.267 | "wanting to find it, but not wanting to use it…" [p.259] [p.260] | **0**: điều kiện mấu chốt đến từ trí nhớ, không có trong context; cả 2 citation chỉ hỗ trợ ý phụ nên chấm `false` |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** `starter/results/agent-{dev,test}-<config>/traces.json` (ID, điểm, trang, đoạn trích ≤160 ký tự; context đầy đủ nằm trong `traces_full.json`, không commit vì bản quyền), `reviews.json`, `summary.json`, `auto_summary.json` (token, chi phí, thời gian). Lệnh chạy ở [src/README.md](../../src/README.md).

**Phân tích lỗi agent trên dev, trước/sau:**

1. **Lỗi định dạng citation** (A0 q14, A1 q03): câu trả lời đúng bị chuyển thành từ chối vì thiếu `[id]` trong câu. *Giả thuyết:* tự gắn các ID đã liệt kê (có trong context) vào câu trả lời (A3). *Kết quả:* cách sửa **không kích hoạt lần nào** trong lượt chạy A3 (0/22), vì lần này model luôn tự đặt `[id]`; q03 đúng chỉ do dao động. Hiệu quả **chưa đo được**, và theo tiêu chí đã chốt, A3 không được chọn.
2. **Trả lời theo trí nhớ khi context thiếu** (A0 q20; một phần A0 q16). *Giả thuyết:* context tốt hơn (R5 + lân cận) lấp được khoảng trống. *Kết quả:* A1 và A3 lấy được p.267 và trích đúng; q20 từ 0 lên 2, q16 từ 1 lên 2.
3. **Một lượt search không đủ** (A2 q16 hết lượt, q17, q20). *Kết quả:* cho phép tới 3 lượt (A1) tăng từ 17 lên 19/20, đổi lại chi phí cao hơn khoảng 40%.
4. **Dao động giữa các lần chạy:** A1 và A3 giống nhau về retrieval và prompt, chỉ khác cách sửa (không kích hoạt), vậy mà khác kết quả ở q03 và q13. Mỗi cấu hình chỉ chạy một lần, nên sai số khoảng ±1–2 câu trên 20.
5. **Kiến thức có sẵn của model lọt vào query:** "seven galleons" (q02), "Caput Draconis" (q13), "Harry Cedric tie" (u01), "Aberforth" (u02), "Goblet of Fire" (u03, u04). Nhờ đó, agent tìm kiếm tốt hơn hẳn retrieval một lượt dùng query gốc (A0 dev context 16/20 so với R1 9/20). Tuy nhiên, model không dùng kiến thức đó để trả lời câu thiếu bằng chứng: từ chối đúng 4/4 ở mọi cấu hình.

**Trước/sau và đánh đổi:**
- **Retrieval:** R1 → R5, context dev 9 → 17/20, test 2 → 4/6. Cái giá: độ trễ 60 ms → khoảng 3–4 giây, context 2.7k → 6.8k ký tự.
- **Agent:** A0 → A1, dev đúng 16 → 19/20. Trên test cả hai đạt 6/6 (test nhỏ, và agent bù cho retrieval yếu bằng cách search nhiều lượt: A0 search 3 lượt ở 4/8 câu). Chi phí mỗi câu khoảng 1.6 lần (dev $0.012 → $0.019, test $0.017 → $0.028), input token khoảng 1.8 lần, độ trễ tương đương.
- **Search một lượt** (A2) rẻ hơn khoảng 30% so với A1 nhưng mất 2 câu dev.

**Giới hạn:**
- Nhãn câu trả lời và context do AI soạn và kiểm tra lại 2 lượt; **học viên cần xác nhận** (BENCHMARK.md: model kiểm tra là chưa đủ).
- Benchmark nhỏ (20 dev / 6 test có đáp án). Cả baseline và cấu hình đã chọn đều đạt 6/6 trên test, nên test không phân biệt được hai cấu hình. Mỗi cấu hình chỉ chạy một lần.
- Test do chính người xây pipeline soạn (với hỗ trợ của AI), dù được chốt trước khi chạy.
- Model biết sẵn nội dung Harry Potter, nên số liệu retrieval của agent lạc quan hơn so với một tài liệu model chưa từng thấy.
- **Agent đi qua proxy bên thứ ba `api.xah.io`:** các đoạn trích sách được gửi qua dịch vụ ngoài, học viên cần xác nhận quyền theo RULES.md. Số token proxy báo không kiểm chứng được (lần thử đầu báo `input_tokens=1`); chi phí tính theo giá niêm yết Anthropic, hóa đơn proxy có thể khác.
- Dùng Chroma (HNSW) thay pgvector của Day 6; kết quả dense vẫn trùng Day 6.
- Hit dựa trên evidence vàng có thể bỏ sót evidence thay thế hợp lệ (q08, t02 trích đoạn khác nhưng vẫn đúng).
- Embedding và reranker chạy trên CPU.
- Test `test_baseline` của starter cần `PYTHONUTF8=1` trên Windows.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
