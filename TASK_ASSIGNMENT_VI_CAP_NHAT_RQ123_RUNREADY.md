# Nhóm 14 — Phân công nhiệm vụ, tiến độ và hướng dẫn bàn giao thí nghiệm

**Đề tài:** Đánh giá tác động của thông tin ngữ nghĩa từ LLM đến độ chính xác và thiên lệch phổ biến trong gợi ý sách dựa trên đồ thị.  
**Phạm vi:** Tái hiện kết quả và phân tích thực nghiệm có kiểm soát giữa LightGCN và RLMRec-Con (tên trong mã nguồn RLMRec là `lightgcn_plus`) trên Amazon-book; tập trung vào **(RQ1) accuracy, popularity bias và popularity dependence, (RQ2) semantic alignment và semantic–collaborative agreement, (RQ3) propagation depth × semantic alignment**; **không xây dựng kiến trúc gợi ý mới**.  
**GitHub nhóm:** https://github.com/trungkien1502/project_sna_group14  
**Mã nguồn gốc:** https://github.com/HKUDS/RLMRec — cố định tại commit `22413752246de3dee8ab0d509f7f7a8889080f95`  
**Google Drive dùng chung:** `MyDrive/Data_SNA_Group14/`  
**Trạng thái:** Cập nhật 14/10/2026 — **đang ở tuần 1 của tổng cộng 6 tuần (12/10–22/11/2026)**. Mục tiêu hoàn thành thí nghiệm cốt lõi **trước hoặc trong tuần 3 (01/11)**, khóa toàn bộ kết quả muộn nhất **tuần 4 (08/11)**, và có bản nộp hoàn chỉnh **tuần 5 (15/11)**; **tuần 6 là dự phòng**, không bắt buộc chờ đến tuần 6 mới nộp. Đây là mốc nội bộ đề xuất, không phải deadline chính thức của giảng viên. Thay “Thành viên 1–5” bằng tên thật khi chốt nhóm.

## 1. Quy ước chung cho toàn nhóm

- **GitHub:** Lưu mã nguồn, notebook, mã kiểm thử, tệp cấu hình không chứa thông tin bí mật, tài liệu và các tệp CSV/JSON tổng hợp nhỏ. Thành viên làm việc trên nhánh riêng và tạo **Pull Request (PR)** để Leader kiểm tra trước khi gộp. **Không đẩy lên GitHub** dataset đầy đủ, embedding của user/item, checkpoint lớn, log dài hoặc thông tin đăng nhập Colab/Google Drive.
- **Google Drive:** Lưu các tệp `.pkl` gốc, danh sách gợi ý lớn, kết quả chi tiết theo user, checkpoint, log dài và dữ liệu trung gian dung lượng lớn. **Không sửa các tệp trong** `data/raw/amazon/`.
- **Vị trí notebook:** Hiện các notebook có thể vẫn nằm ở **thư mục gốc repository**. Nhóm sẽ chuyển sang `notebooks/` theo hướng dẫn tại Mục 2. Từ lúc hoàn tất commit chuyển file, các task và README sử dụng đường dẫn `notebooks/...`.
- **Dataset:** Amazon-book gồm **11.000 users**, **9.332 items**; train **120.464**, validation **40.290**, test **40.106** tương tác. Giữ nguyên các tập chia cho mọi mô hình và seed. Chỉ dùng **train** để xây dựng đồ thị cộng tác và xác định độ phổ biến của item. Chọn checkpoint bằng validation; chỉ đánh giá test cuối cùng sau khi đã chọn checkpoint.
- **Random seeds:** Kế hoạch ban đầu là `[2023, 2024, 2025]`. Paper gốc sử dụng 5 lần khởi tạo ngẫu nhiên nên có thể cân nhắc mở rộng lên 5 seeds nếu nhóm thống nhất; **không tự ý thay đổi protocol đã chốt**.
- **Chỉ số đánh giá:** Sử dụng `K = 10, 20` cho Recall, NDCG, ARP, APLT và Head/Tail Recall. RQ1 bổ sung mức phụ thuộc giữa **train item degree** và **recommendation frequency/exposure**; RQ2 bổ sung chỉ số **semantic–collaborative agreement** đã chốt trước khi xem test; RQ3 so sánh các propagation depth cố định. Trước khi mở rộng đánh giá, phải đối chiếu masking và nhãn dương trong mã nguồn gốc. Mọi kết quả phải ghi rõ thuộc **validation** hay **test**.
- **Môi trường:** Google Colab có GPU. Ghi nhận phiên bản Python/PyTorch và loại GPU của từng lần chạy. Thành viên cần được cấp quyền thư mục Drive chung và kiểm tra đường dẫn `MyDrive/Data_SNA_Group14/`; nếu dùng shortcut Drive thì phải xác nhận đường dẫn hoạt động.
- **Phân quyền chạy:** Mỗi thư mục kết quả ứng với một cặp **model/seed** chỉ có một người phụ trách. Không chạy hai tiến trình đồng thời ghi vào cùng thư mục. Không tùy tiện chạy lại để ghi đè kết quả test đã được nghiệm thu.
- **Điều kiện hoàn thành chung:** Mã nguồn được commit và PR được review; tệp yêu cầu trên Drive tồn tại; có ghi chú bàn giao ngắn về cách chạy, giả định, đường dẫn, kết quả kiểm tra, lỗi (nếu có) và kết quả thực tế. **Mỗi thành viên phải cung cấp phần hướng dẫn tái hiện để Leader cập nhật `README.md`.**

## 2. Cấu trúc thư mục chuẩn cho GitHub và Google Drive

Mục tiêu của cách tổ chức này là tách **mã nguồn có thể tái hiện** khỏi **dữ liệu và kết quả có dung lượng lớn**. Đây là **cấu trúc mục tiêu**, không có nghĩa mọi file/thư mục đều đã tồn tại; chỉ tạo file khi task tương ứng bắt đầu. Các notebook E1 đã có cần được chuyển từ thư mục gốc vào `notebooks/` **một lần**, trước khi nhóm làm việc trên các nhánh riêng.

### 2.1. GitHub — mã nguồn, cấu hình, tài liệu và kết quả báo cáo nhỏ

```text
project_sna_group14/
├── README.md                        # Hướng dẫn tái hiện từ đầu đến cuối
├── requirements.txt                 # Phụ thuộc và phiên bản phần mềm
├── .gitignore                       # Loại trừ dữ liệu, checkpoint và file lớn
│
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_E1_LightGCN_Setup.ipynb
│   ├── 03_E1_Full_Training.ipynb
│   ├── E2_popularity_analysis.ipynb       # RQ1: bias + popularity dependence
│   ├── E3_semantic_alignment.ipynb       # RQ2: ablation + agreement analysis
│   └── E4_propagation_analysis.ipynb     # RQ3: depth × semantic alignment
│
├── src/
│   ├── inference/
│   │   └── export_topk.py                 # Leader: xuất Top-K từ checkpoint
│   ├── evaluation/
│   │   └── popularity_metrics.py          # Leader setup; TV3 chỉ chạy/kiểm tra output E2
│   ├── ablation/                          # Leader setup switch/config; TV4 chỉ chạy E3-A
│   └── analysis/
│       ├── semantic_collab_agreement.py   # Leader setup; TV5 chỉ chạy E3-B
│       └── propagation_analysis.py        # Leader setup; TV5 chỉ chạy/tổng hợp E4
│
├── configs/
│   └── experiment_protocol.yaml          # Protocol/cấu hình chung khi đã chốt
│
├── tests/
│   ├── test_topk_export.py
│   ├── test_popularity_metrics.py
│   └── test_agreement_metrics.py
│
├── docs/
│   ├── TASK_ASSIGNMENT.md          # Phân công và tiêu chí nghiệm thu
│   ├── EXPERIMENT_SPEC.md          # Protocol và định dạng output chung
│   └── runs/                       # Ghi nhận các lượt chạy E1 (khi cần)
│
└── results/
    ├── tables/                     # CSV/Markdown tổng hợp nhỏ
    └── figures/                    # Biểu đồ dùng trong báo cáo
```

**Quy định:**

- `README.md` ở **thư mục gốc** để người chấm nhìn thấy ngay khi mở repository. Phải ghi thứ tự thực hiện, nguồn dataset, đường dẫn, bước cài đặt, lệnh/notebook chạy E1–E4, cách tạo bảng kết quả và các hạn chế của việc tái hiện.
- `notebooks/` chứa phần điều phối và thao tác Colab; `src/` chứa **hàm xử lý có thể gọi lại**, giúp E2–E4 không phải sao chép code giữa các notebook.
- `configs/` lưu giá trị tham số đã chốt, không ghi token, tài khoản hoặc đường dẫn cá nhân. `tests/` chứa test độc lập có ví dụ nhỏ để xác minh cách tính toán.
- `results/` chỉ chứa **bảng tổng hợp và hình nhỏ phục vụ báo cáo**; dữ liệu dự đoán lớn, log dài và checkpoint đưa lên Drive.
- **Không sao chép nguyên repository RLMRec của tác giả** vào repo nhóm. README/notebook phải nêu nguồn và **commit RLMRec cố định**; riêng thay đổi ablation ở E3 phải có mã/patch do nhóm quản lý để tái hiện được.

### 2.2. Google Drive — dữ liệu gốc, dự đoán và kết quả huấn luyện

Giữ nguyên các tệp E1 đã có, **không di chuyển/đổi tên** thư mục train hiện hành.

```text
MyDrive/Data_SNA_Group14/
├── data/
│   ├── raw/
│   │   └── amazon/
│   │       ├── trn_mat.pkl
│   │       ├── val_mat.pkl
│   │       ├── tst_mat.pkl
│   │       ├── usr_prf.pkl
│   │       ├── itm_prf.pkl
│   │       ├── usr_emb_np.pkl
│   │       └── itm_emb_np.pkl
│   └── manifests/                 # Kiểm tra dữ liệu và audit
│
└── experiments/
    ├── E1/
    │   ├── lightgcn/
    │   │   └── seed_2023/         # Các seed khác tạo khi chạy
    │   └── lightgcn_plus/
    │       └── seed_2023/
    ├── predictions/
    │   └── <model>/seed_<seed>/  # Top-20 và metadata export
    ├── E2/                         # RQ1: bias + popularity dependence
    ├── E3/                         # RQ2: ablation + agreement
    └── E4/                         # RQ3: model/depth/seed + Top-K
```

**Mỗi lượt E1** giữ `run_config.json`, `metrics.json`, `verification.json`, `validation_history.json`, `training_console.log`, `checkpoints/`, `logs/` trong `experiments/E1/<model>/seed_<seed>/`. E2–E4 sử dụng thư mục riêng tương ứng; không ghi đè output của E1. **Không dùng Drive cá nhân làm điều kiện bắt buộc duy nhất cho người ngoài tái hiện:** README phải chỉ cách tải dữ liệu từ nguồn công khai, tổ chức lại đường dẫn tương đương và chạy lại với Colab riêng.


### 2.4. Ai chịu trách nhiệm cập nhật tài liệu tái hiện?

- **Leader:** Sở hữu `README.md`, `.gitignore`, cấu trúc thư mục và **setup sẵn notebook/code/config/verification cho E1–E4**. Trước khi giao task, Leader phải chạy smoke test để bảo đảm thành viên chỉ cần đổi đúng các biến được cho phép, Run All, kiểm tra PASS và bàn giao output.
- **Thành viên 1–2 (E1 + E4 runs):** Chỉ chạy notebook/config đã khóa cho model/seed/depth được giao; cung cấp môi trường, checkpoint/metrics/Top-K, PASS/FAIL và lỗi thực tế. Không tự đổi protocol.
- **Thành viên 3–5 (E2–E4 analysis):** Không tự thiết kế metric hay sửa logic nghiên cứu. Chỉ chạy notebook/code đã được Leader chuẩn bị, kiểm tra verification, lưu đúng output và ghi lại cách chạy/tham số thực tế. Leader tích hợp vào README chính.
- **Mỗi lần thay đổi cấu trúc hoặc lệnh chạy:** Cập nhật README và tài liệu task trong **cùng PR**, tránh đến tuần nộp mới ghi lại quy trình.

---

## 3. Phân công cho 6 người (đề xuất)

| Người phụ trách | Trách nhiệm chính | Thành viên thực hiện gì | Leader phải setup trước | Sản phẩm chính |
|---|---|---|---|---|
| **Leader (bạn)** | Bao quát toàn bộ RQ1–RQ3, setup pipeline, review và tổng hợp | Chốt protocol, chuẩn bị notebook/code/config/verification; smoke test trước khi giao; nghiệm thu output | Toàn bộ E1–E4, exporter Top-K, metric, ablation switch, agreement analysis, depth protocol | `docs/EXPERIMENT_SPEC.md`, notebook/code chạy được, bảng tổng hợp E1–E4 |
| **Thành viên 1** | E1 LightGCN + E4 LightGCN | Chọn `SEED` ở E1; sau đó chọn `DEPTH` + `SEED` ở E4; Run All; kiểm tra PASS; xuất Top-K | Notebook E1/E4 LightGCN, output path, exporter, verification | E1 LightGCN seeds còn lại + E4 LightGCN theo depth |
| **Thành viên 2** | E1 RLMRec-Con + E4 RLMRec-Con | Chọn `SEED` ở E1; sau đó chọn `DEPTH` + `SEED` ở E4; Run All; kiểm tra PASS; xuất Top-K | Notebook E1/E4 RLMRec-Con, output path, exporter, verification | E1 RLMRec-Con seeds còn lại + E4 RLMRec-Con theo depth |
| **Thành viên 3** | E2 — RQ1 | Chọn `MODEL` + `SEED` hoặc chạy batch đã giao; Run All; kiểm tra `E2_VERIFICATION: PASS`; bàn giao CSV/JSON/figure | `popularity_metrics.py`, unit test, notebook E2, head/tail rule, Spearman, verification | Bảng accuracy/bias + popularity dependence |
| **Thành viên 4** | E3-A — Ablation cho RQ2 | Chọn `ALIGN_MODE`, `ALIGN_WEIGHT`, `SEED`; Run All; không sửa loss; kiểm tra đúng mode/weight và PASS | Patch/switch alignment, config hợp lệ, gradient/smoke verification, notebook E3-A | Graph-only/user-only/item-only/full/weight runs |
| **Thành viên 5** | E3-B Agreement + E4 analysis | Chọn `SEED`/analysis mode theo bảng giao; Run All notebook đã khóa; kiểm tra PASS; bàn giao bảng/figure | Agreement metric, group rule, analysis notebook, depth-curve notebook, verification | Agreement groups + performance delta; depth curves; case minh họa |

**Nguyên tắc cốt lõi:** Leader chịu trách nhiệm phần khó và phần quyết định nghiên cứu. Thành viên **không tự nghĩ protocol, không tự sửa metric/loss, không tự chọn threshold/depth sau khi xem test**. Task chuẩn là: `mở notebook → đổi đúng biến được phép → Run All → PASS → lưu output → báo Leader`. Mục 4 (E1) là mẫu chuẩn để áp dụng cho E2–E4.

### 3.1. Thời gian thực hiện và deadline theo từng người

**Khung thời gian:** Tuần 1 là **12–18/10/2026**; mục tiêu vẫn là có thí nghiệm cốt lõi trước **01/11**, khóa kết quả kỹ thuật muộn nhất **08/11**, hoàn thiện gói nộp trước **15/11**.

| Người | Tuần 1 — trước 18/10 | Tuần 2 — trước 25/10 | Tuần 3 — trước 01/11 | Tuần 4–5 |
|---|---|---|---|---|
| **Leader** | Khóa protocol; setup notebook/code E2–E4; smoke test từng pipeline | Review E1; khóa E2; duyệt E3 switch; khóa E4 depth/config | Nghiệm thu E2–E4; tổng hợp bảng trả lời RQ1–RQ3 | **08/11** khóa số liệu; **15/11** report/slide/README |
| **TV1 – LightGCN** | Chạy E1 seed 2024/2025 + Top-K | Khi Leader giao notebook E4: chạy LightGCN theo depth/seed được phân | Hoàn tất run E4 LightGCN, bàn giao PASS + output | Chỉ chạy bổ sung nếu Leader yêu cầu |
| **TV2 – RLMRec-Con** | Chạy E1 seed 2024/2025 + Top-K | Khi Leader giao notebook E4: chạy RLMRec-Con theo depth/seed được phân | Hoàn tất run E4 RLMRec-Con, bàn giao PASS + output | Chỉ chạy bổ sung nếu Leader yêu cầu |
| **TV3 – E2** | Chờ Leader khóa notebook E2; kiểm tra quyền Drive/input | Chạy các model/seed được giao; bàn giao bảng E2 + PASS | Chạy nốt seed còn thiếu; tạo figure/tổng hợp từ notebook sẵn | Chỉnh báo cáo theo review |
| **TV4 – E3-A** | Chờ Leader khóa switch/config; chạy smoke test theo hướng dẫn | Chạy user-only/item-only/full/weight theo ma trận giao | Hoàn tất E3-A core runs; bàn giao PASS + output | Chạy bổ sung nếu Leader yêu cầu |
| **TV5 – E3-B + E4 analysis** | Chờ Leader khóa agreement metric/notebook | Chạy agreement analysis trên output đã có; kiểm tra PASS | Chạy notebook tổng hợp E4; xuất curves/case theo rule sẵn | Hoàn thiện diễn giải theo review |

**Phụ thuộc:** E2 chỉ bắt đầu chính thức khi notebook E2 đã được Leader smoke-test. E3 chỉ chạy khi switch alignment và agreement metric đã khóa. E4 chỉ chạy khi depth set/output path đã khóa. Thành viên không tự xử lý bằng cách đổi protocol nếu notebook lỗi; báo `Done / Next / Blocked` kèm log để Leader sửa pipeline.


## 4. E1 — Hướng dẫn giao việc chính xác cho Thành viên 1 và 2

### Đầu vào (Input)

- Notebook trên GitHub sau khi chuẩn hóa cấu trúc: `notebooks/03_E1_Full_Training.ipynb` (trước khi commit di chuyển, tệp vẫn có thể ở thư mục gốc).
- Bảy tệp dataset trong `MyDrive/Data_SNA_Group14/data/raw/amazon/`.
- Mã nguồn RLMRec ở commit đã khóa và môi trường Google Colab GPU.

### Cách thực hiện

1. Mở notebook từ GitHub bằng **một phiên Colab mới**. Kiểm tra biến `ROOT` trỏ đúng đến thư mục Drive mà mình có quyền truy cập.
2. Chỉ sửa hai biến `MODEL` và `SEED` ở **Cell 1** theo nhiệm vụ. Thành viên 1 dùng `MODEL="lightgcn"`; Thành viên 2 dùng `MODEL="lightgcn_plus"`.
3. Nếu trước đó đã import RLMRec cho lần chạy khác, khởi động lại phiên Colab. Chạy lần lượt **Cell 1 → Cell 5**. Không tự ý đổi hyperparameters, chia dữ liệu, early stopping hay cách đánh giá.
4. Cell 5 cần báo `CELL 5: SUCCESS`. Xác nhận checkpoint đọc được, test cuối cùng chạy thành công, đủ JSON và log. Nếu lỗi kiểm tra, **ghi lại lỗi**; không sửa dataset hoặc dùng test để điều chỉnh tham số.
5. Viết báo cáo ngắn gồm: model, seed, số epoch đã chạy, best epoch, validation Recall@20, test Recall@10/20 và NDCG@10/20, source commit, môi trường và đường dẫn Drive.
6. Sau khi Cell 5 đạt, chạy `src/inference/export_topk.py` cho đúng model/seed; xác nhận `export_verification.json` báo `passed` và lưu 3 tệp Top-K dưới `experiments/predictions/`.

### Đầu ra mong đợi (Output trên Drive)

```text
MyDrive/Data_SNA_Group14/experiments/E1/<model>/seed_<seed>/
  run_config.json
  metrics.json
  verification.json
  validation_history.json
  training_console.log
  checkpoints/...
  logs/...
```

### Tiêu chí nghiệm thu

Cell 5 thành công; bốn chỉ số test là số hữu hạn trong khoảng `[0, 1]`; `recall@10 <= recall@20`; checkpoint nạp lại được; model/seed/source commit đúng; output nằm đúng thư mục; PR chứa báo cáo chạy đã được Leader review. **Có đủ tệp chỉ chứng minh quy trình kỹ thuật chạy được, không tự động chứng minh kết quả khớp paper.**

### Việc Leader thực hiện khi thu đủ E1

Thu thập đủ **3 seeds cho mỗi mô hình**; tính trung bình (*mean*) và độ lệch chuẩn mẫu (*sample standard deviation*) cho từng metric. Đối chiếu với paper trên cùng dataset, K và protocol, ghi rõ chênh lệch. **Không chọn seed dựa trên điểm test cao nhất.**

## 5. Chuẩn đầu ra dự đoán dùng chung — Leader, TV1–TV3 và TV5

E2 cần Top-K từ E1; E4 cần Top-K cho **từng model/depth/seed** để áp lại cùng bộ metric bias. Notebook training không tự xuất đầy đủ loại dữ liệu này; nhóm **đã xây dựng và kiểm thử `src/inference/export_topk.py` với kết quả E1 seed 2023**. Các run mới phải dùng cùng exporter/masking, chỉ đổi checkpoint và output directory.

### Định dạng đầu ra đã sử dụng trên Drive

```text
experiments/predictions/<model>/seed_<seed>/topk_test.npz
experiments/predictions/<model>/seed_<seed>/export_meta.json
experiments/predictions/<model>/seed_<seed>/export_verification.json
```

- **`topk_test.npz`:** Chứa mảng `user_ids` kích thước `(11000,)`, `item_ids` và `scores` kích thước `(11000,20)`. ID của user/item là số nguyên; mỗi hàng item được sắp từ điểm cao nhất xuống thấp nhất. Mảng `scores` chứa điểm dự đoán tương ứng. Không cần ghi toàn bộ ma trận dày `11000 × 9332` lên Drive.
- **`export_meta.json`:** Ghi model, seed, đường dẫn checkpoint, source commit, tập đánh giá (`test`), K (`20`), quy tắc masking, tổng số user/item, thời điểm xuất và commit code tạo output.

### Kiểm tra bắt buộc

Tính lại **Recall@20 và NDCG@20 từ danh sách Top-K đã xuất và ma trận test**, rồi đối chiếu với cách đánh giá của trainer gốc trong sai số số học mà nhóm thống nhất. Quy tắc masking đã đối chiếu với evaluator gốc: loại train positives, không tự loại validation positives khi đánh giá test. Kiểm tra từng danh sách không có item trùng, ID hợp lệ, không chứa train positives bị cấm và không sử dụng kết quả test để quyết định thứ tự xếp hạng.

## 6. E2 — Popularity Bias & Popularity Dependence (RQ1, Thành viên 3)

### Leader phải chuẩn bị trước khi giao

- `src/evaluation/popularity_metrics.py` đã cài sẵn Recall/NDCG/ARP/APLT/Head-Tail Recall, `freq@K` và Spearman `pop ↔ freq`.
- `tests/test_popularity_metrics.py` chạy PASS trên dữ liệu giả lập có đáp án tính tay.
- `notebooks/E2_popularity_analysis.ipynb` đã có sẵn load dữ liệu, load Top-K, tính metric, xuất bảng/figure và verification.
- Head/Tail rule, K, cách xử lý zero-exposure và output path đã khóa trong `EXPERIMENT_SPEC.md`.
- Leader chạy thử ít nhất **1 model/1 seed** và xác nhận notebook kết thúc bằng `E2_VERIFICATION: PASS`.

### Đầu vào (Input)

- `trn_mat.pkl`, `tst_mat.pkl`.
- Top-K đã verified từ `experiments/predictions/<model>/seed_<seed>/`.
- Notebook `notebooks/E2_popularity_analysis.ipynb`.

### Biến Thành viên 3 được phép sửa

Chỉ sửa các biến ở cell cấu hình đầu notebook theo bảng Leader giao:

```python
MODEL = "lightgcn"        # hoặc "lightgcn_plus"
SEED = 2023               # 2023 / 2024 / 2025
K_LIST = [10, 20]         # đã khóa; không tự đổi
```

Không sửa công thức metric, head/tail rule, Spearman, masking, test split hoặc code trong `src/evaluation/`.

### Cách thực hiện

1. Mở notebook bằng phiên Colab mới, mount đúng `Data_SNA_Group14`.
2. Chọn đúng `MODEL` và `SEED` được giao.
3. Run All từ đầu đến cuối.
4. Notebook phải báo `E2_VERIFICATION: PASS`.
5. Kiểm tra output có đủ accuracy metrics, bias metrics, `freq@K`, Spearman và figure degree↔exposure.
6. Nếu FAIL hoặc thiếu file: lưu log/screenshot lỗi và báo Leader; **không tự sửa công thức hay protocol**.
7. Gửi Leader một dòng bàn giao: `MODEL / SEED / PASS-FAIL / đường dẫn output`.

### Đầu ra mong đợi

```text
MyDrive/Data_SNA_Group14/experiments/E2/<model>/seed_<seed>/
  metrics.json
  popularity_dependence.json
  item_exposure.csv
  verification.json
  figures/...
```

### Tiêu chí nghiệm thu

`E2_VERIFICATION: PASS`; popularity chỉ dùng train; output model/seed đúng; Top-K input đã passed; metric hữu hạn; không có thay đổi code/protocol ngoài biến cho phép. Leader tổng hợp mean±SD và viết kết luận RQ1.

---

## 7. E3 — Semantic Alignment Analysis (RQ2)

### 7.1. E3-A — Alignment Ablation (Thành viên 4)

### Leader phải chuẩn bị trước khi giao

- Xác định chính xác user/item alignment loss trong `lightgcn_plus`.
- Tạo switch/config trong code nhóm cho `graph_only`, `user_only`, `item_only`, `full`; không yêu cầu TV4 tự sửa source RLMRec.
- Khóa các alignment weight cần chạy và mapping parameter thật sự tới loss.
- Có smoke/gradient verification xác nhận mỗi mode bật/tắt đúng thành phần.
- Chuẩn bị `notebooks/E3_semantic_alignment.ipynb` với cell cấu hình duy nhất và output path tự động.
- Leader chạy thử ít nhất 1 cấu hình và xác nhận `E3A_VERIFICATION: PASS`.

### Đầu vào

- Dataset/semantic embeddings đã khóa.
- Notebook E3-A do Leader chuẩn bị.
- Ma trận cấu hình `ALIGN_MODE × ALIGN_WEIGHT × SEED` do Leader giao.

### Biến Thành viên 4 được phép sửa

```python
ALIGN_MODE = "user_only"   # graph_only / user_only / item_only / full
ALIGN_WEIGHT = 0.01        # chỉ dùng giá trị Leader đã liệt kê
SEED = 2024                # theo phân công
```

Không sửa loss code, optimizer, backbone, split, early stopping, metric hoặc output naming.

### Cách thực hiện

1. Mở notebook E3 bằng phiên Colab mới.
2. Điền đúng 3 biến được giao.
3. Run All.
4. Kiểm tra `E3A_VERIFICATION: PASS`, trong đó verification phải ghi đúng `ALIGN_MODE`, `ALIGN_WEIGHT`, `SEED`.
5. Sau training, chạy exporter Top-K đã setup sẵn nếu notebook không tự gọi.
6. Nếu verification FAIL: báo Leader; **không tự sửa loss**.
7. Bàn giao `mode / weight / seed / PASS-FAIL / output path`.

### Đầu ra mong đợi

```text
MyDrive/Data_SNA_Group14/experiments/E3/runs/<align_mode>/weight_<w>/seed_<seed>/
  run_config.json
  metrics.json
  verification.json
  checkpoints/...
  topk/... hoặc đường dẫn prediction tương ứng
```

### Tiêu chí nghiệm thu

Mode/weight/seed đúng; verification PASS; checkpoint chọn bằng validation; không thay đổi ngoài cấu hình được giao; Top-K vượt kiểm tra masking. Leader tổng hợp ablation để trả lời phần “user-side hay item-side đóng góp nhiều hơn”.

### 7.2. E3-B — Semantic–Collaborative Agreement (Thành viên 5)

### Leader phải chuẩn bị trước khi giao

- Chốt representation nào là **collaborative view** và **semantic view**.
- Chốt agreement metric chính, `M`, cách chia low/medium/high và cách tính performance delta **trước khi xem test result của analysis này**.
- Cài sẵn `src/analysis/semantic_collab_agreement.py`, unit test và `notebooks/E3_semantic_alignment.ipynb`/analysis cell.
- Notebook tự load representation/output cần thiết, tự nhóm agreement và xuất bảng/figure.
- Leader smoke-test để notebook báo `E3B_VERIFICATION: PASS`.

### Biến Thành viên 5 được phép sửa

```python
AGREEMENT_SIDE = "user"    # hoặc "item"
SEED = 2023                # theo phân công
```

Không tự đổi similarity, Top-M, ngưỡng group, representation source hay cách tính delta.

### Cách thực hiện

1. Mở notebook đã khóa, chọn `AGREEMENT_SIDE` và `SEED`.
2. Run All.
3. Kiểm tra `E3B_VERIFICATION: PASS`.
4. Xác nhận output có số lượng mẫu của từng agreement group và performance delta tương ứng.
5. Nếu một group quá ít mẫu hoặc notebook cảnh báo, báo Leader; không tự đổi threshold.
6. Bàn giao `side / seed / PASS-FAIL / output path`.

### Đầu ra mong đợi

```text
MyDrive/Data_SNA_Group14/experiments/E3/agreement/<side>/seed_<seed>/
  agreement_scores.csv
  group_summary.csv
  performance_delta.csv
  verification.json
  figures/...
```

### Tiêu chí nghiệm thu

Agreement protocol đúng bản Leader đã khóa; verification PASS; không cherry-pick case; có số lượng mẫu từng group; output đủ để Leader kết luận alignment thay đổi thế nào khi graph và semantic đồng thuận/không đồng thuận.

---

## 8. E4 — Propagation–Semantic Interaction (RQ3)

**TV1 chạy LightGCN; TV2 chạy RLMRec-Con; TV5 chỉ chạy notebook tổng hợp sau khi có output.**

### Leader phải chuẩn bị trước khi giao

- Khóa core depth set, mặc định `DEPTH ∈ {1, 2, 3}`; chỉ Leader mới được đổi.
- Chuẩn bị notebook/config sao cho **chỉ thay `MODEL`, `DEPTH`, `SEED`** là đủ chạy.
- Giữ toàn bộ hyperparameter khác theo protocol; với RLMRec-Con khóa full-alignment weight đại diện bằng validation.
- Output path tự chứa model/depth/seed; không ghi đè E1.
- Tích hợp exporter Top-K và verification sau mỗi run.
- Chuẩn bị `notebooks/E4_propagation_analysis.ipynb` cho TV5 để tự đọc toàn bộ run và vẽ curve.
- Chốt tiêu chí `comparable accuracy` trong `EXPERIMENT_SPEC.md` trước khi xem test của E4.

### 8.1. Chạy model theo depth — TV1 và TV2

#### Biến được phép sửa

TV1:
```python
MODEL = "lightgcn"
DEPTH = 1                 # 1 / 2 / 3 theo bảng Leader giao
SEED = 2024
```

TV2:
```python
MODEL = "lightgcn_plus"
DEPTH = 1                 # 1 / 2 / 3 theo bảng Leader giao
SEED = 2024
```

Không sửa bất kỳ hyperparameter nào khác.

#### Cách thực hiện

1. Mở notebook E4 bằng phiên Colab mới.
2. Điền đúng `MODEL`, `DEPTH`, `SEED`.
3. Run All.
4. Kiểm tra `E4_RUN_VERIFICATION: PASS`.
5. Xác nhận checkpoint, metrics và Top-K đều tồn tại; exporter verification phải `passed`.
6. Nếu FAIL: gửi log cho Leader; không tự đổi depth set, learning rate, early stopping hoặc weight.
7. Bàn giao `model / depth / seed / PASS-FAIL / output path`.

#### Đầu ra mong đợi

```text
MyDrive/Data_SNA_Group14/experiments/E4/<model>/layers_<depth>/seed_<seed>/
  run_config.json
  metrics.json
  verification.json
  checkpoints/...
  predictions/...
```

### 8.2. Tổng hợp propagation curves — TV5

#### Biến được phép sửa

TV5 không tự chọn depth hay metric. Notebook đọc toàn bộ ma trận run đã khóa. Nếu cần chỉ chọn:

```python
RUN_SET = "core"
```

#### Cách thực hiện

1. Chỉ bắt đầu khi Leader xác nhận đủ các run cốt lõi cần so sánh.
2. Mở `notebooks/E4_propagation_analysis.ipynb` và Run All.
3. Kiểm tra `E4_ANALYSIS_VERIFICATION: PASS`.
4. Notebook phải xuất curve `Recall/NDCG vs depth`, `ARP/APLT/Tail Recall vs depth` và bảng cặp depth có accuracy comparable theo rule đã khóa.
5. Bàn giao PASS/FAIL và output path; không tự chọn cặp đẹp để báo cáo.

#### Đầu ra mong đợi

```text
MyDrive/Data_SNA_Group14/experiments/E4/analysis/
  depth_summary.csv
  comparable_accuracy_pairs.csv
  verification.json
  figures/...
```

### Tiêu chí nghiệm thu

Tất cả run đúng model/depth/seed; ngoài depth không có thay đổi không được phép; checkpoint chọn bằng validation; Top-K pass masking; analysis dùng đúng comparable-accuracy rule. Leader là người kết luận RQ3/H3, không giao thành viên tự diễn giải theo cảm tính.

### 8.3. Supporting Error/Case Analysis — chỉ chạy sau cùng

Leader chuẩn bị sẵn rule chọn case. TV5 chỉ chạy notebook để xuất ví dụ minh họa cho: tail item tăng/giảm, semantic–collaborative disagreement, và thay depth làm ranking đổi. Case chỉ dùng minh họa, không thay kết luận định lượng.


## 9. Kế hoạch 6 tuần — ưu tiên hoàn thành sớm

| Tuần | Khoảng ngày | Mốc kết quả cần đạt |
|---|---|---|
| **Tuần 1** | **12–18/10** | Khóa protocol RQ1–RQ3; TV1–TV2 chạy E1 seed còn lại; **Leader setup + smoke-test E2/E3/E4** |
| **Tuần 2** | **19–25/10** | **Khóa E1** + Top-K; TV3 chạy E2; TV4 chạy E3-A; TV5 chạy E3-B khi đủ input; TV1–TV2 bắt đầu E4 theo depth đã giao |
| **Tuần 3** | **26/10–01/11** | Hoàn thành các run E2–E4 cốt lõi; TV5 chạy notebook tổng hợp E4; Leader có bảng/curve để bước đầu trả lời **RQ1–RQ3** |
| **Tuần 4** | **02–08/11** | Bổ sung seed/run còn thiếu, review chéo, **khóa kết quả kỹ thuật**; hoàn thiện bảng/hình và giới hạn kết luận |
| **Tuần 5** | **09–15/11** | Hoàn chỉnh report, slides, README, clean-run và gói nộp |
| **Tuần 6** | **16–22/11** | Dự phòng: sửa lỗi, diễn tập, nộp/bảo vệ theo lịch chính thức |

**Ưu tiên GPU khi thiếu tài nguyên:** E1 hoàn tất trước; sau đó ưu tiên **E3 core ablation** và **E4 depth core**. Không mở rộng thêm weight/depth nếu làm trễ bộ thí nghiệm cốt lõi đã chốt.


## 10. Checklist Pull Request cho mọi thành viên

- [ ] Nhánh và PR ghi rõ mã task, người phụ trách, cách chạy lại.
- [ ] Code ghi rõ commit RLMRec và sử dụng đúng chia tập dữ liệu cố định.
- [ ] Không commit dataset `.pkl` gốc, checkpoint lớn, token/credentials của Drive hoặc dữ liệu chi tiết quá lớn.
- [ ] Có bằng chứng kiểm thử/smoke test và output thực tế.
- [ ] Báo cáo nêu metric, phân biệt **validation/test**, ghi rõ đường dẫn kết quả.
- [ ] PR cập nhật hướng dẫn chạy/đầu vào/đầu ra tương ứng để README phục vụ tái hiện luôn chính xác.
- [ ] Leader review PR trước khi merge vào `main`.

## 11. Việc thực hiện ngay

1. **Leader:** Ưu tiên setup trước, chưa giao thành viên tự phát triển E2–E4. Hoàn thiện lần lượt: (a) notebook E2 + verification, (b) E3 alignment switch + verification, (c) E3 agreement notebook + verification, (d) E4 depth notebook + exporter + analysis notebook. Mỗi pipeline phải có ít nhất một smoke run trước khi giao.
2. **TV1–TV2:** Trong lúc Leader setup E2–E4, chỉ tập trung hoàn tất E1 seeds 2024/2025 + Top-K đúng hướng dẫn Mục 4. Sau đó chờ bảng `DEPTH × SEED` của Leader để chạy E4.
3. **TV3:** Chưa tự code metric mới. Khi nhận `E2_popularity_analysis.ipynb` đã khóa, chỉ chạy các `MODEL × SEED` được giao, kiểm tra `E2_VERIFICATION: PASS` và bàn giao output.
4. **TV4:** Chưa tự sửa RLMRec loss. Khi Leader bàn giao switch/config E3-A đã smoke-test, chỉ chạy `ALIGN_MODE × ALIGN_WEIGHT × SEED` theo bảng giao.
5. **TV5:** Chưa tự định nghĩa agreement metric/depth analysis. Khi Leader khóa notebook E3-B/E4-analysis, chỉ chạy theo `AGREEMENT_SIDE/SEED` hoặc `RUN_SET` được giao và bàn giao output/figures.
6. **Cả nhóm:** Nếu notebook FAIL, dừng và báo Leader. Không tự sửa metric/loss/protocol để “làm cho chạy được”. Mọi thay đổi logic phải quay về Leader và được cập nhật trong `EXPERIMENT_SPEC.md` trước khi chạy lại.

## 12. Hướng dẫn sử dụng Google Drive chung trên Google Colab (bắt buộc trước khi chạy task)

**Thư mục duy nhất của nhóm:** `Data_SNA_Group14` do Leader tạo. Mục đích là để **mỗi thành viên dùng tài khoản Google riêng** nhưng đọc dữ liệu và lưu kết quả vào **cùng một thư mục**, không phải gửi checkpoint hoặc `.pkl` riêng cho Leader.

**A. Leader cấp quyền (chỉ làm một lần):**

1. Vào Google Drive, tìm thư mục **`Data_SNA_Group14`** → **Chia sẻ / Share**.
2. Thêm email Google của 5 thành viên với quyền **Người chỉnh sửa / Editor**; thông báo cho nhóm sau khi chia sẻ.
3. Nhắc mọi người **không sửa, xóa, đổi tên** bất kỳ tệp nào trong `data/raw/amazon/` và không ghi đè kết quả model/seed do người khác phụ trách. Leader nên giữ một bản sao lưu kết quả đã nghiệm thu.

**B. Mỗi thành viên thiết lập trên Google Drive cá nhân:**

1. Mở mục **Được chia sẻ với tôi / Shared with me**, tìm `Data_SNA_Group14`.
2. Chọn **Sắp xếp / Organize → Thêm lối tắt / Add shortcut** vào **My Drive** (thư mục gốc của My Drive); **không chọn tạo bản sao / Make a copy**.
3. Giữ đúng tên lối tắt `Data_SNA_Group14`. Nếu đã có thư mục trùng tên trong My Drive, xử lý trùng tên trước khi chạy Colab.
4. Mở Colab bằng tài khoản Google của chính mình, bật GPU, chạy `drive.mount('/content/drive')` và xác thực tài khoản vừa được chia sẻ.

**C. Kiểm tra bắt buộc quyền đọc/ghi từ Colab (mỗi thành viên chạy một lần):**

```python
from google.colab import drive
from pathlib import Path
import uuid

drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/Data_SNA_Group14')
assert root.is_dir(), 'Không thấy folder Drive nhóm: kiểm tra quyền và lối tắt trong My Drive'
assert (root / 'data/raw/amazon/trn_mat.pkl').is_file(), 'Không đọc được dataset chung'

# Chỉ tạo và xóa file test của chính mình, không chạm vào file chung.
test_file = root / f'colab_access_test_{uuid.uuid4().hex}.txt'
try:
    test_file.write_text('ok', encoding='utf-8')
    assert test_file.read_text(encoding='utf-8') == 'ok'
    print('PASS: Đọc dataset và ghi Drive nhóm thành công')
finally:
    if test_file.exists():
        test_file.unlink()
```

Nếu lệnh trên không báo `PASS`, **dừng trước khi training**: kiểm tra lại quyền Editor, shortcut có nằm ở gốc My Drive không, tài khoản Google dùng để mount, và tên thư mục/đường dẫn. Không tự thay toàn bộ notebook sang đường dẫn Drive cá nhân rồi nộp kết quả ở nơi khác mà chưa thông báo Leader.

**D. Quy tắc lưu kết quả cho từng thành viên:**

- TV1: `experiments/E1/lightgcn/seed_2024/`, `seed_2025/`; TV2: `experiments/E1/lightgcn_plus/seed_2024/`, `seed_2025/`.
- E4: TV1 lưu LightGCN và TV2 lưu RLMRec-Con theo `experiments/E4/<model>/layers_<L>/seed_<seed>/`; không ghi đè E1.
- TV3 lưu E2 tại `experiments/E2/`; TV4 lưu E3 runs tại `experiments/E3/runs/`; TV5 lưu agreement/analysis tại `experiments/E3/agreement/` và `experiments/E4/analysis/`.
- **GitHub chứa code và hướng dẫn chạy; Drive chứa dataset, logs, checkpoint và output lớn.** Mỗi PR cần ghi đúng đường dẫn Drive để Leader tìm kết quả.
- Không chia sẻ mật khẩu/tài khoản Google cho nhau. Quyền Editor có thể xóa/ghi đè file, nên tránh đổi tên/xóa dữ liệu chung và giữ bản sao lưu những lượt chạy đã đạt.

**Ghi chú tái hiện cho người ngoài nhóm:** Quyền truy cập Drive chung chỉ phục vụ nhóm. `README.md` vẫn phải hướng dẫn tải Amazon-book từ nguồn công khai và tạo cấu trúc Drive tương đương trên tài khoản riêng; **không yêu cầu người chấm phải được cấp quyền Drive cá nhân của Leader**.
