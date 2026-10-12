# QUY CHUẨN THÍ NGHIỆM & ĐỊNH DẠNG BÀN GIAO — Nhóm 14

**Tệp duy nhất:** `docs/EXPERIMENT_SPEC.md`  
**Gộp từ:** `EXPERIMENT_PROTOCOL.md` (quy tắc nghiên cứu) và `OUTPUT_CONTRACT.md` (giao diện dữ liệu). **Không cần lưu hai file cũ.**  
**Đề tài:** Đánh giá tác động của thông tin ngữ nghĩa từ LLM đến độ chính xác và thiên lệch phổ biến trong gợi ý sách dựa trên đồ thị.  
**Mã nguồn gốc:** [HKUDS/RLMRec](https://github.com/HKUDS/RLMRec), commit `22413752246de3dee8ab0d509f7f7a8889080f95`.

> **Trạng thái: DỰ THẢO CHỜ LEADER CHỐT.** Các hạng mục seed, ngưỡng head/tail, cách tính head/tail recall và ablation loss phải được duyệt trước khi ghi kết quả chính thức. Các đoạn JSON có `null` hoặc `FILL_...` chỉ là ví dụ schema, không phải kết quả đã chạy.

## Cách sử dụng tài liệu

- **Leader:** Đọc toàn bộ và ký duyệt các quyết định ở phần C; phụ trách chuẩn hóa Top-K và kiểm thử tính khớp E1.
- **Thành viên E1/E3:** Đọc phần A về splits, masking, seed, early stopping và phần B về thư mục run.
- **Thành viên E2/E4:** Đọc phần A về popularity bias, error analysis và phần B về `topk_test.npz`, schemas JSON/CSV.
- **GitHub** lưu code, cấu hình, tests, tài liệu và hình/bảng nhỏ; **Google Drive** lưu dữ liệu `.pkl`, checkpoints, logs, dự đoán lớn.

---

## PHẦN A — Quy tắc thí nghiệm (Experiment Protocol)

### 1. Nguyên tắc thí nghiệm

- **Một protocol chung** cho LightGCN, RLMRec-Con và các biến thể E3; những khác biệt trong YAML gốc giữa các mô hình phải được ghi nhận, không âm thầm ép tất cả tham số thành một bộ giống nhau.
- **Không thay đổi** 3 ma trận chia tập tác giả phát hành. Không chia lại ngẫu nhiên, không chia theo timestamp cho các thí nghiệm chính.
- **Tập train** dùng để dựng graph và tính độ phổ biến item. **Validation** dùng để chọn checkpoint; **test** chỉ báo cáo kết quả cuối cùng, không chọn cấu hình dựa trên test.
- Định dạng và quy tắc xuất kết quả nằm ở Phần B tài liệu này. Mỗi thay đổi protocol cần cập nhật tài liệu và thông báo cho các task liên quan.

### 2. Dữ liệu và phân chia tập

Nguồn: Amazon-book theo gói dữ liệu công khai của RLMRec.

| Thuộc tính | Giá trị đã kiểm tra |
|---|---:|
| Số người dùng | 11.000 |
| Số sách | 9.332 |
| Số tương tác train | 120.464 |
| Số tương tác validation | 40.290 |
| Số tương tác test | 40.106 |
| Kích thước ma trận | `(11000, 9332)` |
| Giao nhau của từng cặp split | 0 tương tác |

Các file gốc: `trn_mat.pkl`, `val_mat.pkl`, `tst_mat.pkl`, `usr_prf.pkl`, `itm_prf.pkl`, `usr_emb_np.pkl`, `itm_emb_np.pkl`. Dữ liệu được đọc từ `MyDrive/Data_SNA_Group14/data/raw/amazon/` khi chạy trong Colab nhóm. Hướng dẫn nạp dữ liệu công khai cho người ngoài thuộc README; không bắt buộc họ có Drive của nhóm.

Dữ liệu semantic đã kiểm tra: `usr_emb_np.pkl` `(11000,1536)` và `itm_emb_np.pkl` `(9332,1536)`; dùng embeddings tác giả phát hành, không tự tạo lại bằng prompt không xác minh được.

### 3. Môi trường và khả năng tái hiện

- Chạy trên Google Colab có GPU. Lần kiểm thử của Leader: Python `3.13.15`, PyTorch `2.11.0+cu130`, GPU Tesla T4.
- Clone đúng commit RLMRec nêu trên; ghi lại **Git commit của repo nhóm** và phiên bản Python/PyTorch/GPU từng run.
- Không mặc định môi trường Colab bất biến: bản `requirements.txt` hoặc script kiểm tra dependencies cần được kiểm thử trên runtime mới trước hạn nộp.
- Không đẩy lên GitHub checkpoint, profile/embedding `.pkl`, token hoặc dữ liệu lớn. Tài liệu, code, config, test và bảng tổng hợp nhỏ nằm trên GitHub; checkpoint/log/predictions lớn nằm trên Drive.

### 4. Models, seeds và cấu hình E1

| Mã nguồn | Tên trong báo cáo | Vai trò |
|---|---|---|
| `lightgcn` | LightGCN | Baseline, chỉ học từ graph tương tác |
| `lightgcn_plus` | LightGCN + RLMRec-Con | Graph + contrastive semantic alignment từ LLM profiles |

**Bộ seed hiện có trong kế hoạch:** `[2023, 2024, 2025]`. Chạy từng seed **cho cả hai mô hình**, cùng tập chia và cách chọn checkpoint. Paper dùng 5 lần khởi tạo; quyết định có mở rộng thành 5 seeds không phải chốt **trước khi** triển khai toàn bộ E1, nếu đổi phải cập nhật tài liệu và phân công.

Cấu hình gốc tại commit đã khóa:

| Thuộc tính | LightGCN | `lightgcn_plus` |
|---|---:|---:|
| Embedding dim | 32 | 32 |
| Graph layers | 3 | 3 |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Train batch size | 4096 | 4096 |
| Max epochs | 3000 | 3000 |
| Validation interval | 3 epochs (code: epoch index 0, 3, ...) | Tương tự |
| Early stopping patience | 5 validation checks | 5 checks |
| Regularization (`reg_weight`, Amazon) | `1e-6` | `1e-7` |
| Semantic weight (`kd_weight`, Amazon) | — | `0.01` |
| Semantic temperature | — | `0.2` |

`save_model` mặc định của LightGCN là `false`; notebook nhóm **bật lưu checkpoint cho cả hai**. Không đổi tham số khác nếu không ghi rõ và mở một nhánh thí nghiệm khác.

**Kiểm thử Leader đã xác nhận:** LightGCN seed 2023, 247 epochs, best epoch theo số thứ tự bắt đầu từ 1 là 232; test Recall@10 `0.091574`, Recall@20 `0.143436`, NDCG@10 `0.069997`, NDCG@20 `0.086940`. Đây là kết quả **một lần chạy**, không phải trung bình nhiều seeds. Với `lightgcn_plus` seed 2023, Leader cho biết đã chạy xong; cần dẫn tới JSON và verification thực tế, không tự ghi số nếu chưa trích xuất.

### 5. Protocol đánh giá — theo đúng mã nguồn RLMRec

**Quan trọng: Đây là quy tắc quan sát được từ code ở commit đã khóa, không phải quy tắc suy đoán.**

1. Graph cộng tác được dựng từ **train** bằng `DataHandlerGeneralCF.load_data()`.
2. Với validation: ground truth = positive items của `val_mat.pkl`; với test: ground truth = positive items của `tst_mat.pkl`.
3. Tập user dùng để tính Recall/NDCG trên một split gồm **những user có ít nhất một positive item trong split đó** (`AllRankTstData.test_users`); không được giả định luôn bằng 11.000 mà phải ghi số lượng thực tế.
4. Điểm xếp hạng: mô hình tính scores cho mọi sách, sau đó **mask tất cả train-positive items của user** bằng `-1e8` theo `train_mask` trong `model.full_predict()`.
5. **Validation-positive items không tự động bị mask khi đánh giá test trong code gốc.** Không âm thầm bổ sung masking validation khi tái hiện E1/E2/E4; nếu muốn thử, phải xem đó là thí nghiệm phụ có nhãn riêng.
6. Chọn Top-K lớn nhất (`K=20`), các chỉ số K=10 sử dụng 10 phần tử đầu của chính ranking đó.
7. Các chỉ số Recall/NDCG của tác giả **lấy trung bình trên users có positive ở split**, không phải gộp tổng hits trên tất cả positives.
8. Checkpoint tốt nhất là lần validation có **Recall@20 cao nhất**. Trainer xét kết quả sau những epoch index chia hết cho `test_step=3`; nếu không cải thiện trong 5 lượt validation liên tiếp thì dừng. Trainer sau đó nạp lại best state và mới test.
9. Sử dụng test **chỉ sau khi chọn** checkpoint theo validation. Không dùng test để chỉnh weight, seed, cấu hình hoặc chọn ablation tốt nhất.

Định nghĩa cho user `u`, tập positive test `T_u`, top-k recommendations `R_u^k`:

- `Recall@K = mean_{u thuộc U_test} ( |R_u^K ∩ T_u| / |T_u| )`.
- `NDCG@K = mean_{u thuộc U_test} ( DCG_u@K / IDCG_u@K )`; binary relevance, `DCG = sum_{r=1..K} hit_u(r)/log2(r+1)` và `IDCG` tương ứng `min(K, |T_u|)` kết quả đúng ở đầu ranking.

Các metric chính được trình bày cho `K=10,20`; source YAML có cả `K=5`, có thể tiếp tục tính nhưng không bắt buộc đưa vào bảng chính.

### 6. E2 — Popularity bias (định nghĩa đề xuất để duyệt)

Mọi thống kê `pop(i)` dùng **train**: `pop(i) = số tương tác train với item i` (từ ma trận nhị phân train, không dùng val/test).

- **Head:** Sắp xếp item theo `(−pop(i), item_id)`; lấy `ceil(0.20 * 9332) = 1867` item đầu. **Tail:** 7465 item còn lại. Item có `pop=0` vẫn thuộc tập item và xếp theo quy tắc trên. Đây là *quy ước nghiên cứu của nhóm*, không khẳng định là phân hoạch tác giả RLMRec dùng.
- **ARP@K:** `mean_u [(1/K) * sum_{i thuộc R_u^K} pop(i)]` — độ phổ biến trung bình của recommendations, **train degree thô**.
- **APLT@K:** `mean_u [ (# item tail trong R_u^K)/K ]`.
- **HeadRecall@K (micro):** `sum_u |R_u^K ∩ T_u ∩ Head| / sum_u |T_u ∩ Head|`.
- **TailRecall@K (micro):** `sum_u |R_u^K ∩ T_u ∩ Tail| / sum_u |T_u ∩ Tail|`.

**Chú ý:** Head/TailRecall ở đây là **micro** để tránh chia cho zero theo từng user và khác phép mean-over-user của Recall gốc. **Phải ghi rõ cách gộp này trong bảng/README**, không gọi chung cả hai là cùng một kiểu Recall. Báo cáo cả hai mẫu số (`head_test_positives`, `tail_test_positives`). Nếu mẫu số toàn tập bằng 0, metric là không xác định, không gán 0.

Diễn giải popularity bias cần xem đồng thời độ chính xác và cơ cấu gợi ý; ARP giảm/APLT tăng không tự động có nghĩa toàn hệ thống tốt hơn.

### 7. E3 — Ablation (chỉ đổi thành phần được thử nghiệm)

- LightGCN (không semantic alignment) và RLMRec-Con full alignment.
- RLMRec-Con **user-only**: giữ thành phần InfoNCE user; loại bỏ **cả hai** thành phần InfoNCE item (positive + negative).
- RLMRec-Con **item-only**: loại bỏ thành phần InfoNCE user; giữ cả positive + negative item InfoNCE.
- Weight full alignment dự kiến: `kd_weight ∈ {0.001, 0.01, 0.1}`; `0.01` là cấu hình mặc định Amazon. Giữ nguyên temperature `0.2` nếu không có thí nghiệm temperature riêng.
- Giữ nguyên graph, splits, batch size, optimizer, quy trình early stopping và seed; code ablation phải nằm trong repo nhóm (patch/wrapper/module) và có kiểm tra loss, không sửa trực tiếp commit baseline mà không lưu thay đổi.
- Khi chọn kết quả E3 chính, dùng validation để quyết định cấu hình theo quy tắc đã công bố; test của mỗi cấu hình chỉ dùng để phân tích/báo cáo, không chọn tùy ý kết quả tốt nhất.

### 8. E4 — Error analysis

- Cùng danh sách user, cùng Top-20, cùng test ground truth và cùng seed nếu thực hiện so sánh seed-paired.
- Nhóm user theo train-degree (ngưỡng phân nhóm cần chốt và ghi nhận), và nhóm item theo head/tail E2.
- Phân tích các user cải thiện/giảm từ LightGCN sang RLMRec-Con bằng `hits@20`, Recall@20 và ví dụ có quy tắc lựa chọn minh bạch; không dùng ví dụ riêng lẻ để suy ra quan hệ nhân quả.
- Mọi ví dụ/biểu đồ phải tái tạo được từ `topk_test.npz` và ma trận gốc.

---

## PHẦN B — Định dạng input/output và bàn giao (Output Contract)

**Quy ước chung:** ID user và item là số nguyên nội bộ của Amazon-book (`user: 0..10999`, `item: 0..9331`). Đối với E1, `model` là `lightgcn` hoặc `lightgcn_plus`. Mọi đầu ra phải cho biết model, seed, dataset, split, phiên bản mã nguồn, và dấu vết input thích hợp. Không đưa test positives vào tính điểm/xếp hạng; chỉ được dùng để xác định tập user đánh giá và tính metric cuối.

### 2. Đầu ra E1 — đã được notebook nhóm tạo

Vị trí trên Google Drive:

```text
MyDrive/Data_SNA_Group14/experiments/E1/<model>/seed_<seed>/
├── run_config.json
├── metrics.json
├── verification.json
├── validation_history.json
├── training_console.log
├── checkpoints/
│   └── <model>-amazon-<seed>.pth
└── logs/
```

Đây là **định dạng có thật của notebook E1**, không yêu cầu sửa tên output đang dùng. Trước khi inference, kiểm tra:

- `verification.json`: `status="passed"`.
- `metrics.json`: `status="completed"`, đúng `model`, `seed`, source commit, và chứa `test_metrics` với `recall@10`, `recall@20`, `ndcg@10`, `ndcg@20`.
- Checkpoint load được bằng PyTorch, đúng mô hình và đúng seed.
- `run_config.json` và log ghi đầy đủ môi trường và cấu hình. File checkpoint không được ghi đè trong cùng model/seed.

**Nếu hiện các seed ngoài 2023 chưa được chạy**, chỉ tạo folder khi run thực sự hoàn thành; không tạo JSON giả.

### 3. Giao diện quan trọng nhất: Top-K export E1 → E2/E4

**Người phát triển/kiểm tra:** Leader.  
**Đề xuất code:** `src/inference/export_topk.py` (sẽ lập trình).  
**Input:** checkpoint E1 + 3 ma trận dữ liệu, mã nguồn RLMRec cố định, model/seed cần xuất.  
**Output trên Drive:**

```text
MyDrive/Data_SNA_Group14/experiments/predictions/<model>/seed_<seed>/
├── topk_test.npz
├── export_meta.json
└── export_verification.json
```

#### 3.1. Schema của `topk_test.npz`

File `.npz` nén chứa:

| Key | Kiểu dữ liệu | Shape | Nội dung |
|---|---|---|---|
| `user_ids` | `np.int32` hoặc `np.int64` | `(n_eval_users,)` | Danh sách user có ít nhất 1 positive trong test; **sắp tăng dần, không trùng** |
| `top_items` | `np.int32` hoặc `np.int64` | `(n_eval_users, 20)` | ID của 20 sách được gợi ý, mỗi hàng xếp từ điểm cao xuống thấp |
| `top_scores` | `np.float32` hoặc `np.float64` | `(n_eval_users, 20)` | **Tùy chọn**: scores của các item tương ứng; hữu ích cho debug, không dùng để tính Recall/NDCG |

`n_eval_users` = số user có positive tại `tst_mat.pkl`, **không được tự mặc định bằng 11.000**. Kiểm tra thực tế rồi ghi con số này trong metadata.

Ví dụ **minh họa cấu trúc, không phải kết quả thật**:

```python
import numpy as np
with np.load('topk_test.npz') as arr:
    user_ids  = arr['user_ids']     # (n_eval_users,)
    top_items = arr['top_items']    # (n_eval_users, 20)
    scores = arr['top_scores'] if 'top_scores' in arr.files else None
```

#### 3.2. Cách tạo ranking phải khớp trainer gốc

1. Khởi tạo RLMRec tại source commit đã khóa, đọc `run_config.json`, dựng graph **từ train** và nạp **đúng best checkpoint** đã lưu.
2. Chuyển mô hình sang chế độ đánh giá; tính scores bằng đường gọi tương đương `model.full_predict()`.
3. **Mask train-positive items của user bằng `−1e8`**, không mask validation-positive khi xuất `test` theo protocol tái hiện của trainer gốc. `full_predict()` đã áp dụng train mask qua `_mask_predict`; **không mask trái protocol lần thứ hai** hoặc thêm validation mask.
4. Sử dụng `torch.topk(..., k=20)` hoặc cách tương đương cho kết quả cùng thứ tự. Lưu `user_ids` được sắp tăng và sắp lại `top_items`/`top_scores` theo hàng tương ứng.
5. Chỉ dùng test positives để xác định **tập user đánh giá và tính metric cuối**, không dùng để tạo scores, lọc items hoặc sắp ranking.

#### 3.3. Schema `export_meta.json` — bắt buộc

```json
{
  "schema_version": "1.0",
  "experiment": "E1_Reproduction",
  "model": "lightgcn",
  "dataset": "amazon",
  "seed": 2023,
  "split": "test",
  "top_k": 20,
  "n_users_dataset": 11000,
  "n_items_dataset": 9332,
  "n_eval_users": null,
  "masking": "train_positives_only",
  "rlmrec_commit": "22413752246de3dee8ab0d509f7f7a8889080f95",
  "group_repo_commit": "FILL_FROM_GIT_REV_PARSE_HEAD",
  "checkpoint_path": "FILL_ACTUAL_CHECKPOINT_PATH",
  "checkpoint_sha256": "FILL_IF_AVAILABLE",
  "exported_at_utc": "FILL_ISO_8601_TIMESTAMP"
}
```

Các trường `null`/`FILL_...` chỉ là **placeholder trong tài liệu**; khi chương trình chạy thật **phải ghi giá trị thực**, đặc biệt `n_eval_users`, commit và checkpoint. Nếu không tính được SHA256 thì ghi `null` và lý do vào log; đường dẫn thực tế bắt buộc phải có.

#### 3.4. Quy tắc kiểm thử `export_verification.json`

- Shape chính xác, integer ID hợp lệ, `user_ids` duy nhất và tăng dần.
- Mỗi hàng có đủ 20 `top_items`, item hợp lệ, **không có ID trùng trong cùng hàng**.
- Không có item nào trong Top-20 thuộc **train positives của đúng user**.
- `user_ids` khớp tập user có positive của `tst_mat.pkl` (không thiếu/thừa).
- Nếu có `top_scores`, mọi điểm hữu hạn, sắp không tăng theo rank; kiểm tra trên ví dụ mẫu rằng ranking khớp `full_predict`.
- **Kiểm thử tích hợp quyết định:** Từ `topk_test.npz` và `tst_mat.pkl`, tính lại **Recall@10/20 và NDCG@10/20** theo *macro user averaging* của RLMRec. Kết quả khớp `metrics.json` của **cùng model/seed/checkpoint**, độ lệch tuyệt đối dự kiến tối đa `1e-6` (nếu không, phải điều tra, không coi là PASS).
- Chỉ đánh dấu `"status": "passed"` sau khi hoàn tất toàn bộ kiểm tra; ghi `max_abs_metric_diff`, số user, lỗi và thời điểm nghiệm thu.

**Điều kiện chặn:** E2/E4 không báo cáo kết quả chính thức nếu chưa có ít nhất 1 file Top-K được nghiệm thu bằng bước tính lại Recall/NDCG.

### 4. Đầu ra E2 — Popularity Bias

**Đầu vào:** `topk_test.npz`, `trn_mat.pkl`, `tst_mat.pkl`, `export_meta.json`; theo các định nghĩa metrics ở Phần A.

**Code GitHub:** `src/evaluation/popularity_metrics.py`, `tests/test_popularity_metrics.py`; notebook phân tích nếu cần nằm trong `notebooks/`.

**Kết quả Drive (mỗi model/seed):**

```text
experiments/E2/metrics/<model>/seed_<seed>.json
```

Schema tối thiểu (giá trị dưới đây là ký hiệu **không phải số đo thật**):

```json
{
  "schema_version": "1.0",
  "model": "lightgcn",
  "dataset": "amazon",
  "seed": 2023,
  "split": "test",
  "popularity_source": "train_only",
  "head_fraction": 0.2,
  "head_count": 1867,
  "tail_count": 7465,
  "head_tail_tiebreaker": "pop_desc_item_id_asc",
  "head_tail_recall_aggregation": "micro_positive_weighted",
  "n_eval_users": null,
  "head_test_positives": null,
  "tail_test_positives": null,
  "metrics": {
    "arp@10": null,
    "arp@20": null,
    "aplt@10": null,
    "aplt@20": null,
    "head_recall@10": null,
    "head_recall@20": null,
    "tail_recall@10": null,
    "tail_recall@20": null
  },
  "source_topk_path": "FILL_ACTUAL_PATH",
  "status": "completed"
}
```

**Yêu cầu kiểm thử:** có unit test trên dataset nhỏ với phép tính tay; một item test có thể ở head hoặc tail nhưng không cả hai; tất cả scores hữu hạn và trong miền hợp lệ (ARP không giới hạn trong `[0,1]`); metric undefined dùng `null` với giải thích, không tự gán 0.

**GitHub:** chỉ commit code/tests/docs/bảng tổng hợp nhỏ. **Drive:** kết quả chi tiết và mọi predictions lớn.

### 5. Đầu ra E3 — Ablation

**Input:** commit RLMRec, source patch nhóm quản lý, `trn/val/tst_mat.pkl`, các semantic embeddings, danh sách seeds.

**Các ID variant đề xuất:** `full`, `user_only`, `item_only`, `weight_0p001`, `weight_0p01`, `weight_0p1`. Cách biến thể tác động đến InfoNCE phải được xác nhận bằng code và ghi trong mô tả PR/config của variant.

**Drive:**

```text
experiments/E3/<variant_id>/seed_<seed>/
├── run_config.json
├── metrics.json
├── verification.json
├── checkpoints/
└── logs/
```

Đầu ra chứa tối thiểu `variant_id`, `semantic_components` (user/item/both), `kd_weight`, `kd_temperature`, model, seed, `rlmrec_commit`, `group_repo_commit`, số epoch, best validation recall@20 và kết quả test. Nếu sử dụng lại checkpoint full E1 thì phải trỏ đến đúng run gốc, không sao chép rồi đổi nhãn.

**GitHub:** `src/ablation/`, notebook nếu có, code patch/config/README ngắn, unit/smoke tests. Mọi khác biệt ngoài alignment phải được ghi nhận để giữ tính so sánh công bằng.

### 6. Đầu ra E4 — Error Analysis

**Input:** predictions của hai mô hình (ưu tiên so sánh **cùng seed**), test positives, train degrees, head/tail membership theo E2.

**Drive:**

```text
experiments/E4/per_user_comparison_<seed>.csv
experiments/E4/summary_<seed>.json
```

CSV tối thiểu: `user_id`, `seed`, `lightgcn_hits20`, `rlmrec_hits20`, `lightgcn_recall20`, `rlmrec_recall20`, `delta_recall20`, `train_user_degree`, `user_degree_bucket`. Mọi bảng có thể mở rộng, nhưng tên cột tối thiểu cần giữ ổn định để ghép kết quả.

GitHub: `src/analysis/`, unit tests cho phân nhóm, biểu đồ nhỏ trong `results/figures/` và phần mô tả phân tích trong notebook/báo cáo trình bày tiêu chí chọn ví dụ, số mẫu từng nhóm, hạn chế kết luận.

### 7. Bảng trách nhiệm và quy trình nhận-bàn giao

| Vai trò | Phải tạo/kiểm tra | Giao cho ai |
|---|---|---|
| Leader (inference) | `topk_test.npz`, `export_meta.json`, `export_verification.json` | Thành viên E2/E4 |
| Thành viên E1 LightGCN | E1 folders đúng seed, `verification.json=passed` | Leader |
| Thành viên E1 RLMRec-Con | E1 folders đúng seed, `verification.json=passed` | Leader |
| Thành viên E2 | `src/evaluation/`, tests, `experiments/E2/metrics/...` | Leader + E4 |
| Thành viên E3 | Ablation patch + cấu hình + các run và báo cáo | Leader |
| Thành viên E4 | CSV so sánh user, JSON thống kê, biểu đồ + cách tái tạo | Leader |

---

## PHẦN C — Checklist chốt trước khi bàn giao

- [ ] **Seed:** Chọn chính thức **3 seeds `[2023, 2024, 2025]`** (theo proposal hiện tại) **hoặc 5 seeds** (gần paper hơn), ghi rõ bộ seed cố định. Không thay đổi sau khi so sánh kết quả.
- [ ] **E2:** Chốt head = top **20% item** xếp theo `(popularity giảm dần, item_id tăng dần)`, tail = phần còn lại; popularity chỉ từ train.
- [ ] **E2:** Chốt Head/Tail Recall dạng **micro-positive-weighted** và ghi rõ khác với Recall@K macro theo user của RLMRec.
- [ ] **E3:** Đối chiếu đúng từng thành phần InfoNCE loss khi tắt user-only / item-only; lưu mọi mã nguồn sửa đổi trên GitHub nhóm.
- [ ] **Top-K schema:** Tạo file `.npz` mẫu, để thành viên E2/E4 xác nhận đọc được `user_ids` và `top_items`.
- [ ] **Top-K consistency:** Leader xuất ranking từ **checkpoint thật**, tính lại Recall/NDCG@10,20 và xác nhận khớp `metrics.json` cùng model/seed (sai số tuyệt đối <= `1e-6`, hoặc điều tra nếu lệch).
- [ ] **Đường dẫn:** Dùng thống nhất tên thư mục model/seed/variant và không ghi đè run đã hoàn tất.
- [ ] **Trách nhiệm:** Người phụ trách task phải bàn giao code, test, output, và hướng dẫn chạy lại cho Leader trước khi báo hoàn thành.

**Được phép bắt đầu ngay:** E1 chạy các seed đã chốt; E2/E4 code và unit test với dữ liệu minh họa. **Chỉ chạy và báo cáo kết quả E2/E4 thật sau khi Top-K consistency test đạt.**

### Các file source RLMRec cần đối chiếu

- `encoder/trainer/trainer.py` — training/validation/test, early stopping, checkpoint.
- `encoder/trainer/metrics.py` — Recall/NDCG.
- `encoder/data_utils/data_handler_general_cf.py`, `encoder/data_utils/datasets_general_cf.py`, `encoder/models/base_model.py` — train graph, đánh giá và train masking.
- `encoder/models/general_cf/lightgcn.py`, `lightgcn_plus.py` — mô hình và semantic alignment.
- `encoder/config/modelconf/lightgcn.yml`, `lightgcn_plus.yml` — hyperparameters gốc.

**Quy tắc thay đổi:** Nếu thực hiện thử nghiệm khác protocol, đặt tên variant riêng, lưu config và thay đổi nguồn code, không âm thầm sửa baseline.
