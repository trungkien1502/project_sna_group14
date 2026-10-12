# QUY CHUẨN THÍ NGHIỆM & BÀN GIAO — NHÓM 14

> **Trạng thái:** Bản ngắn gọn để nhóm thống nhất trước khi giao task. Các lựa chọn có ký hiệu **[CẦN CHỐT]** phải được Leader phê duyệt trước khi báo cáo kết quả chính thức.

## 1. Nguyên tắc chung

- **Đề tài:** Tái hiện và phân tích ảnh hưởng của semantic alignment từ LLM đến độ chính xác và popularity bias trong gợi ý sách; **không đề xuất kiến trúc mới**.
- **Mã nguồn:** [HKUDS/RLMRec](https://github.com/HKUDS/RLMRec), commit `22413752246de3dee8ab0d509f7f7a8889080f95`.
- **Dữ liệu:** Amazon-book tác giả phát hành: 11.000 users, 9.332 items; train **120.464**, validation **40.290**, test **40.106** tương tác. Giữ nguyên ba ma trận `trn_mat.pkl`, `val_mat.pkl`, `tst_mat.pkl`; không chia lại theo thời gian hoặc ngẫu nhiên.
- Train để dựng graph, học mô hình và tính popularity; validation để chọn checkpoint và tham số; test chỉ để báo cáo kết quả cuối.
- **Mô hình E1:** `lightgcn` (baseline) và `lightgcn_plus` (LightGCN + RLMRec-Con). Dùng YAML gốc từng mô hình; bật lưu checkpoint, ghi lại mọi thay đổi.
- **Seed [CẦN CHỐT]:** tạm dùng `[2023, 2024, 2025]` theo Proposal V2 (paper báo cáo 5 lần khởi tạo). Dùng cùng danh sách seed cho hai mô hình.
- **Checkpoint:** chọn theo **validation Recall@20** tốt nhất; early stopping theo trainer gốc; nạp best checkpoint rồi mới đánh giá test.

## 2. Quy tắc đánh giá bắt buộc

- Báo cáo **Recall@10/20, NDCG@10/20**: tính theo từng user có positive ở split, sau đó lấy trung bình theo user như RLMRec gốc.
- Khi tạo Top-K test: xếp điểm tất cả item, **loại các item user đã tương tác trong train**; **không tự loại validation items**, vì evaluator gốc không làm vậy. Không dùng test positives để tạo điểm hay xếp hạng.
- `pop(item)` = số tương tác với item trong **train**.
- **Head/Tail [CẦN CHỐT]:** head = top 20% item phổ biến nhất (hòa điểm xếp item ID tăng dần), tail = phần còn lại.
- **ARP@K:** trung bình popularity của K item được gợi ý (dùng train degree). **APLT@K:** tỷ lệ item thuộc tail trong Top-K.
- **Head/Tail Recall [CẦN CHỐT]:** `số hit ở head/tail ÷ số test positives thuộc head/tail` (micro aggregation), báo cáo rõ cách tính khác Recall chung.
- **E3:** so sánh full alignment, user-only, item-only và `kd_weight ∈ {0.001, 0.01, 0.1}`; các cấu hình phải dùng cùng protocol và lưu đầy đủ code thay đổi. Xác nhận chính xác thành phần loss trước khi chạy.
- **E4:** so sánh hai mô hình trên **cùng seed, cùng user và cùng test ground truth**; không lựa chọn ví dụ theo cảm tính mà không nêu tiêu chí.

## 3. Định dạng đầu ra thống nhất

**Google Drive gốc:** `MyDrive/Data_SNA_Group14/`

| Phần | Đầu ra tối thiểu trên Drive | Code cần có trên GitHub |
|---|---|---|
| **E1** | `experiments/E1/<model>/seed_<seed>/` gồm `run_config.json`, `metrics.json`, `verification.json`, log và checkpoint | `notebooks/03_E1_Full_Training.ipynb` |
| **Top-K** (Leader) | `experiments/predictions/<model>/seed_<seed>/topk_test.npz`, `export_meta.json`, `export_verification.json` | `src/inference/export_topk.py` |
| **E2** | `experiments/E2/metrics/<model>/seed_<seed>.json` | `src/evaluation/popularity_metrics.py`, tests |
| **E3** | `experiments/E3/<variant>/seed_<seed>/` gồm config, metrics, verification, checkpoint, log | `src/ablation/` và configs/patch |
| **E4** | `experiments/E4/per_user_comparison_<seed>.csv`, `summary_<seed>.json` | `src/analysis/`, tests, biểu đồ nhỏ |

**Top-K `.npz` phải chứa:**

| Key | Shape | Mô tả |
|---|---|---|
| `user_ids` | `(n_eval_users,)` | User có positive trong test, ID tăng dần, không trùng |
| `top_items` | `(n_eval_users, 20)` | 20 item đề xuất theo thứ tự điểm giảm dần; item ID nguyên hợp lệ |
| `top_scores` | `(n_eval_users, 20)` | Tùy chọn, điểm tương ứng |

`export_meta.json` tối thiểu ghi `model`, `seed`, `split=test`, `top_k=20`, `n_eval_users`, `masking=train_positives_only`, RLMRec commit và đường dẫn checkpoint. Không dùng JSON ví dụ/placeholder làm output chính thức.

**Quy tắc lưu trữ:** GitHub chứa code, notebook, configs, tests, hướng dẫn và bảng/hình nhỏ; Drive chứa `.pkl`, embeddings, checkpoints, logs và predictions lớn. Không push dữ liệu lớn hoặc thông tin truy cập Drive lên GitHub.

## 4. Điều kiện nghiệm thu

- [ ] E1: `verification.json` báo `passed`; checkpoint đọc được; Recall/NDCG hữu hạn, đúng model/seed.
- [ ] Leader xuất Top-20 từ **best checkpoint**; ID hợp lệ, không trùng trong một hàng, không có train positives trong Top-K.
- [ ] Tính lại Recall/NDCG@10/20 từ Top-K và test matrix **khớp E1 `metrics.json` cùng model/seed** (mục tiêu sai lệch tuyệt đối ≤ `1e-6`; nếu khác phải tìm nguyên nhân).
- [ ] E2: tính ARP/APLT/Head-Tail Recall theo cùng định nghĩa; có test tay trên dữ liệu nhỏ.
- [ ] E3: nộp mã nguồn sửa đổi, config, test thành phần loss và kết quả từng variant.
- [ ] E4: nộp bảng so sánh từng user và quy tắc chọn các ví dụ để phân tích.
- [ ] Mỗi task nộp **input, lệnh chạy, output, đường dẫn GitHub/Drive và kiểm thử** để cập nhật README cuối kỳ.

**Lưu ý phụ thuộc:** E1 có thể tiếp tục chạy các seeds sau khi chốt seed. E2/E4 có thể viết code và unit test trước, nhưng **không báo cáo kết quả thật** cho đến khi Top-K được kiểm tra khớp Recall/NDCG của E1.
