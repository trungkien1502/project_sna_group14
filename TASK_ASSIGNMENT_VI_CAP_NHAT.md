# Nhóm 14 — Phân công nhiệm vụ, tiến độ và hướng dẫn bàn giao thí nghiệm

**Đề tài:** Đánh giá tác động của thông tin ngữ nghĩa từ LLM đến độ chính xác và thiên lệch phổ biến trong gợi ý sách dựa trên đồ thị.  
**Phạm vi:** Tái hiện kết quả và phân tích thực nghiệm có kiểm soát giữa LightGCN và RLMRec-Con (tên trong mã nguồn RLMRec là `lightgcn_plus`) trên Amazon-book; **không xây dựng kiến trúc gợi ý mới**.  
**GitHub nhóm:** https://github.com/trungkien1502/project_sna_group14  
**Mã nguồn gốc:** https://github.com/HKUDS/RLMRec — cố định tại commit `22413752246de3dee8ab0d509f7f7a8889080f95`  
**Google Drive dùng chung:** `MyDrive/Data_SNA_Group14/`  
**Trạng thái:** Cập nhật 12/10/2026 — **đang ở tuần 1 của tổng cộng 6 tuần (12/10–22/11/2026)**. Mục tiêu hoàn thành thí nghiệm cốt lõi **trước hoặc trong tuần 3 (01/11)**, khóa toàn bộ kết quả muộn nhất **tuần 4 (08/11)**, và có bản nộp hoàn chỉnh **tuần 5 (15/11)**; **tuần 6 là dự phòng**, không bắt buộc chờ đến tuần 6 mới nộp. Đây là mốc nội bộ đề xuất, không phải deadline chính thức của giảng viên. Thay “Thành viên 1–5” bằng tên thật khi chốt nhóm.

## 1. Quy ước chung cho toàn nhóm

- **GitHub:** Lưu mã nguồn, notebook, mã kiểm thử, tệp cấu hình không chứa thông tin bí mật, tài liệu và các tệp CSV/JSON tổng hợp nhỏ. Thành viên làm việc trên nhánh riêng và tạo **Pull Request (PR)** để Leader kiểm tra trước khi gộp. **Không đẩy lên GitHub** dataset đầy đủ, embedding của user/item, checkpoint lớn, log dài hoặc thông tin đăng nhập Colab/Google Drive.
- **Google Drive:** Lưu các tệp `.pkl` gốc, danh sách gợi ý lớn, kết quả chi tiết theo user, checkpoint, log dài và dữ liệu trung gian dung lượng lớn. **Không sửa các tệp trong** `data/raw/amazon/`.
- **Vị trí notebook:** Hiện các notebook có thể vẫn nằm ở **thư mục gốc repository**. Nhóm sẽ chuyển sang `notebooks/` theo hướng dẫn tại Mục 2. Từ lúc hoàn tất commit chuyển file, các task và README sử dụng đường dẫn `notebooks/...`.
- **Dataset:** Amazon-book gồm **11.000 users**, **9.332 items**; train **120.464**, validation **40.290**, test **40.106** tương tác. Giữ nguyên các tập chia cho mọi mô hình và seed. Chỉ dùng **train** để xây dựng đồ thị cộng tác và xác định độ phổ biến của item. Chọn checkpoint bằng validation; chỉ đánh giá test cuối cùng sau khi đã chọn checkpoint.
- **Random seeds:** Kế hoạch ban đầu là `[2023, 2024, 2025]`. Paper gốc sử dụng 5 lần khởi tạo ngẫu nhiên nên có thể cân nhắc mở rộng lên 5 seeds nếu nhóm thống nhất; **không tự ý thay đổi protocol đã chốt**.
- **Chỉ số đánh giá:** Sử dụng `K = 10, 20` cho Recall, NDCG, ARP, APLT và Head/Tail Recall. Trước khi mở rộng đánh giá, phải đối chiếu cách loại trừ item đã tương tác (*masking*) và cách xác định nhãn dương trong mã nguồn gốc. Mọi kết quả phải ghi rõ thuộc **validation** hay **test**.
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
│   ├── E2_popularity_analysis.ipynb # Sẽ bổ sung ở E2
│   ├── E3_ablation.ipynb            # Sẽ bổ sung ở E3
│   └── E4_error_analysis.ipynb      # Sẽ bổ sung ở E4
│
├── src/
│   ├── inference/
│   │   └── export_topk.py           # Leader: xuất Top-K từ checkpoint
│   ├── evaluation/
│   │   └── popularity_metrics.py    # Thành viên 3: metric E2
│   ├── ablation/                    # Thành viên 4: code điều chỉnh E3
│   └── analysis/                    # Thành viên 5: code phân tích E4
│
├── configs/
│   └── experiment_protocol.yaml    # Protocol/cấu hình chung khi đã chốt
│
├── tests/
│   ├── test_topk_export.py
│   └── test_popularity_metrics.py
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
    ├── E2/
    ├── E3/
    └── E4/
```

**Mỗi lượt E1** giữ `run_config.json`, `metrics.json`, `verification.json`, `validation_history.json`, `training_console.log`, `checkpoints/`, `logs/` trong `experiments/E1/<model>/seed_<seed>/`. E2–E4 sử dụng thư mục riêng tương ứng; không ghi đè output của E1. **Không dùng Drive cá nhân làm điều kiện bắt buộc duy nhất cho người ngoài tái hiện:** README phải chỉ cách tải dữ liệu từ nguồn công khai, tổ chức lại đường dẫn tương đương và chạy lại với Colab riêng.


### 2.4. Ai chịu trách nhiệm cập nhật tài liệu tái hiện?

- **Leader:** Sở hữu `README.md`, `.gitignore` và cấu trúc thư mục; kiểm tra một lượt chạy lại bằng hướng dẫn viết ra, không dựa vào các biến đã tồn tại trong notebook Colab cũ.
- **Thành viên 1–2 (E1):** Cung cấp model/seed, môi trường, cách chạy 5 cell, checkpoint/metrics trên Drive, các lỗi và cách khắc phục đã xác nhận.
- **Thành viên 3–5 (E2–E4):** Mỗi PR phải có hướng dẫn cách tạo output phần mình từ input, đường dẫn file, lệnh/notebook, tham số và test nghiệm thu. Leader tích hợp vào README chính.
- **Mỗi lần thay đổi cấu trúc hoặc lệnh chạy:** Cập nhật README và tài liệu task trong **cùng PR**, tránh đến tuần nộp mới ghi lại quy trình.

---

## 3. Phân công cho 6 người (đề xuất)

| Người phụ trách | Trách nhiệm chính | Công việc kỹ thuật | Sản phẩm trên GitHub | Sản phẩm trên Google Drive |
|---|---|---|---|---|
| **Leader (bạn)** | Chốt quy trình thí nghiệm, tích hợp kỹ thuật, review code, phụ trách đối chiếu E1 | Rà soát training, masking validation/test, nạp checkpoint; xây dựng/duyệt giao diện xuất dự đoán dùng chung; tổng hợp bảng kết quả | `docs/EXPERIMENT_SPEC.md`, `results/tables/E1_summary.csv`, các PR đã review | Kết quả E1 seed 2023 đã kiểm chứng; bảng tổng hợp |
| **Thành viên 1** | E1 — Tái hiện mô hình baseline | Chạy LightGCN với seeds 2024 và 2025; kiểm tra output và ghi nhận môi trường | `docs/runs/E1_lightgcn.md` | `experiments/E1/lightgcn/seed_{2024,2025}/` |
| **Thành viên 2** | E1 — Tái hiện mô hình có ngữ nghĩa | Chạy RLMRec-Con (`lightgcn_plus`) với seeds 2024 và 2025; ghi nhận khả năng so sánh và lỗi phát sinh | `docs/runs/E1_rlmrec_con.md` | `experiments/E1/lightgcn_plus/seed_{2024,2025}/` |
| **Thành viên 3** | E2 — Đánh giá thiên lệch phổ biến | Lập trình item popularity chỉ từ train, phân chia head/tail, ARP@K, APLT@K, Head/Tail Recall@K; viết unit test | `src/evaluation/popularity_metrics.py`, `tests/test_popularity_metrics.py`, `README.md` (mục E2) | `experiments/E2/metrics/` |
| **Thành viên 4** | E3 — Thực nghiệm loại bỏ từng thành phần (ablation) | Kiểm tra alignment loss trong `lightgcn_plus` ở commit đã khóa; triển khai user-only, item-only, full alignment và thay đổi trọng số; giữ nguyên baseline | `src/ablation/`, `notebooks/E3_ablation.ipynb`, `README.md` (mục E3) | `experiments/E3/` |
| **Thành viên 5** | E4 — Phân tích lỗi và từng nhóm đối tượng | So sánh Top-K của hai mô hình, số gợi ý trúng, bậc tương tác user và độ phổ biến item; tạo ví dụ/biểu đồ có thể tái lập | `src/analysis/`, `notebooks/E4_error_analysis.ipynb`, `README.md` (mục E4) | `experiments/E4/` |

**Nguyên tắc cân bằng:** Leader không chỉ làm việc quản lý mà trực tiếp xây dựng giao diện dùng chung, tích hợp và kiểm thử/review kỹ thuật. Leader nên phối hợp sát với Thành viên 4 nếu E3 cần can thiệp sâu vào loss. Khi E4 còn chờ dữ liệu dự đoán Top-K, Thành viên 5 có thể hỗ trợ bảng tổng hợp E1.

### 3.1. Thời gian thực hiện và deadline theo từng người

**Khung thời gian:** Tuần 1 là **12–18/10/2026** (đang thực hiện); còn **5 tuần** đến hết **22/11/2026**. **Không cần dùng hết 6 tuần:** nếu thí nghiệm, tài liệu và kiểm tra tái hiện đạt sớm, nhóm có thể khóa kết quả/nộp sớm theo lịch giảng viên. Bảng dưới là **hạn muộn nhất để quản lý nội bộ**, khuyến khích nộp ngay khi xong. Không chờ đến hạn mới push code hoặc báo tiến độ.

| Người | Tuần 1 — trước 18/10 | Tuần 2 — trước 25/10 | Tuần 3 — trước 01/11 | Tuần 4–5 (dự phòng và báo cáo) |
|---|---|---|---|---|
| **Leader** | Chốt `EXPERIMENT_SPEC.md`, xác nhận Drive dùng chung; khóa công cụ Top-K; giao task | Review đủ E1 và kết quả E2; kiểm tra PR, output, masking/metrics | Duyệt kết quả E3/E4 cốt lõi; có bảng kết quả và bản thảo phân tích đầu tiên | **08/11:** khóa số liệu cuối (nếu chưa khóa); **15/11:** hoàn thiện báo cáo, slide, README kiểm thử lại |
| **TV1 – LightGCN** | Chạy seed 2024/2025, lưu checkpoint/metrics và xác minh Cell 5 | Muộn nhất **21/10** xuất Top-K + xác minh; trước **25/10** nộp báo cáo chạy/PR | Hỗ trợ tổng hợp mean ± SD, đối chiếu paper, hoàn thiện phần baseline | Review tài liệu và hỗ trợ kiểm tra tái hiện khi cần |
| **TV2 – RLMRec-Con** | Chạy seed 2024/2025, lưu checkpoint/metrics và xác minh Cell 5 | Muộn nhất **21/10** xuất Top-K + xác minh; trước **25/10** nộp báo cáo chạy/PR | Hỗ trợ đối chiếu mô hình và giải thích khác biệt | Review tài liệu và hỗ trợ kiểm tra tái hiện khi cần |
| **TV3 – E2 Popularity Bias** | Hoàn thành code ARP/APLT/Head-Tail Recall và unit tests trên dữ liệu nhỏ | **25/10:** chạy đủ hai mô hình/seed đã có, nộp bảng E2 và PR | **01/11:** biểu đồ, mô tả định nghĩa/diễn giải chỉ số | Chỉnh báo cáo theo review, không tự ý đổi metric |
| **TV4 – E3 Ablation** | Xác định chính xác loss và cách bật/tắt user/item alignment | **25/10:** code, config, unit/smoke test và ít nhất một run pilot | **01/11:** kết quả **các cấu hình E3 cốt lõi**; nộp PR và bảng phân tích | Đến **08/11** để chạy bổ sung seed/cấu hình nếu GPU cho phép; sau đó chốt kết quả |
| **TV5 – E4 Error Analysis** | Chốt cách chọn case và phân nhóm user/item | **25/10:** code đối chiếu Top-K chạy được và có kết quả mẫu | **01/11:** phân tích định lượng, case, bảng và biểu đồ E4 | Hoàn thiện diễn giải và các giới hạn trước **08/11** |

**Mục tiêu sớm:** Nếu E1, E2, E3 và E4 đều đạt nghiệm thu vào **01/11**, nhóm chuyển ngay sang tổng hợp, không cần đợi 08/11. Nếu GPU/ablation chậm, **08/11** là mốc khóa kết quả kỹ thuật dự phòng. **15/11** là hạn nội bộ để có gói nộp hoàn chỉnh; tuần 6 (16–22/11) chỉ dành cho sửa lỗi, chạy clean-run/README, diễn tập hoặc nộp theo lịch chính thức.

**Phụ thuộc:** E2/E4 bắt đầu với Top-K seed 2023 đã có; chỉ chốt kết quả đa seed khi TV1–TV2 bàn giao đủ. E3 chỉ chạy full sau khi Leader duyệt thay đổi loss. Nếu công việc bị chặn hoặc trễ **trên 1 ngày**, thành viên báo `Done / Next / Blocked` kèm PR và log để nhóm điều phối ngay.


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

## 5. Chuẩn đầu ra dự đoán dùng chung — Leader, Thành viên 3 và 5

E2 và E4 cần **danh sách sách được xếp hạng theo từng user**, chứ không chỉ `metrics.json` tổng hợp. Notebook E1 không tự xuất loại dữ liệu này; nhóm **đã xây dựng và kiểm thử `src/inference/export_topk.py` với kết quả E1 seed 2023**. Các seed mới phải chạy lại công cụ này sau khi huấn luyện.

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

## 6. E2 — Phân tích thiên lệch phổ biến (Thành viên 3)

### Input

Ma trận train dạng CSR (`trn_mat.pkl`), ma trận test dạng CSR (`tst_mat.pkl`), danh sách Top-20 theo user từ E1 cho hai mô hình và các seed.

### Các định nghĩa cần lập trình và ghi rõ trong tài liệu

- **`pop(i)`** = số tương tác thuộc **train** liên quan đến item `i` (không dùng validation/test).
- **Head/Tail:** Quy tắc đề xuất: sắp item giảm dần theo `pop(i)`; nếu bằng nhau, phân định thứ tự theo item ID để kết quả không thay đổi. **20% item có độ phổ biến cao nhất = head**, **80% còn lại = tail**. Cố định quy tắc cho mọi lượt chạy; ghi rõ cách xử lý item chưa xuất hiện trong train (nếu có).
- **`ARP@K`** = độ phổ biến trung bình của các item được gợi ý trong Top-K, tính trên tất cả users. Nêu rõ dùng degree gốc hay chuẩn hóa; mặc định dùng **raw train degree**.
- **`APLT@K`** = tỷ lệ item thuộc tail trong Top-K, tính trung bình trên các users (tương đương tính gộp khi mọi danh sách có cùng K).
- **`HeadRecall@K`** = tổng số item head thuộc test positives được gợi ý đúng / tổng số head test positives. **`TailRecall@K`** tính tương tự cho tail. Cần báo cáo mẫu số và quy định rõ trường hợp nhóm không có mẫu dương.
- Có thể tính lại Recall@K/NDCG@K để đối chiếu với pipeline, nhưng nên viết các metric thành hàm độc lập, dễ kiểm thử.

### Output

- **GitHub:** `src/evaluation/popularity_metrics.py`, `tests/test_popularity_metrics.py`, `README.md` (mục E2).
- **Drive:** `experiments/E2/metrics/<model>/seed_<seed>.json` cùng bảng tổng hợp/so sánh.

### Tiêu chí nghiệm thu

Có unit test trên một đồ thị giả lập nhỏ với đáp án tính tay; quy tắc head/tail xác định được duy nhất; item popularity không đọc test; tên model/seed/K được giữ đầy đủ; bảng và biểu đồ có thể tạo lại bằng lệnh được tài liệu hóa.

## 7. E3 — Ablation cho semantic alignment (Thành viên 4, Leader review)

### Input

Mã nguồn và config `lightgcn_plus` tại commit RLMRec đã khóa; train/validation/test; embedding ngữ nghĩa cố định: `usr_emb_np.pkl` `(11000,1536)` và `itm_emb_np.pkl` `(9332,1536)`; pipeline E1 đã kiểm chứng.

### Các biến thể thí nghiệm có kiểm soát

- **LightGCN baseline:** Không có alignment.
- **RLMRec-Con full alignment:** Dùng cả user và item alignment.
- **User-only alignment:** Chỉ giữ phần alignment của user.
- **Item-only alignment:** Chỉ giữ phần alignment của item.
- **Thử trọng số full alignment:** `0.001`, `0.01`, `0.1` — chỉ thực hiện sau khi đọc code và xác nhận tham số đang điều chỉnh đúng loss mong muốn.

Có thể dùng lại kết quả E1 cho cấu hình full/default **nếu mọi thiết lập hoàn toàn giống nhau**, tránh chạy lại không cần thiết.

### Cách thực hiện

1. Xác định chính xác các thành phần loss của user/item và vị trí sử dụng chúng trong mã nguồn tác giả.
2. Tạo các tùy chọn bật/tắt rõ ràng trong **mã nguồn do nhóm quản lý**, không âm thầm sửa repository gốc đang được dùng để tái hiện E1.
3. Kiểm tra bằng smoke test nhỏ hoặc một epoch trước khi chạy đủ các seed đã thống nhất.
4. Lưu cấu hình, log, metrics và checkpoint từng biến thể. Không suy luận rằng việc đặt embedding bằng 0 chắc chắn tương đương tắt loss nếu chưa kiểm tra công thức và gradient.

### Output

- **GitHub:** Code có thể chạy lại, hướng dẫn áp dụng thay đổi, `src/ablation/`, `notebooks/E3_ablation.ipynb`, `README.md` (mục E3).
- **Drive:** `experiments/E3/` chứa từng run, config, checkpoint, logs và kết quả.
- **Báo cáo:** Bảng ablation về Recall/NDCG và các chỉ số thiên lệch phổ biến khi có thể.

### Tiêu chí nghiệm thu

Đúng thành phần loss được vô hiệu hóa (được xác nhận qua review code hoặc kiểm tra gradient); dataset split, optimizer và kiến trúc được giữ nguyên ngoài thay đổi chủ đích; có seed/trọng số cụ thể; chọn checkpoint chỉ dựa trên validation.

## 8. E4 — Phân tích lỗi và phân tích theo nhóm (Thành viên 5)

### Input

Danh sách Top-K từ E1; ma trận train/test dạng CSR; item degree và nhãn head/tail do E2 tạo; có thể bổ sung Top-K của E3 nếu sẵn sàng. Hồ sơ user/item dùng để minh họa phải được xử lý cẩn thận, không công bố thông tin nhạy cảm/bí mật.

### Công việc

- So sánh user có số lượt gợi ý trúng trong test tăng/giảm khi chuyển từ LightGCN sang RLMRec-Con ở `K=20`.
- Phân nhóm theo số tương tác train của user và theo head/tail của item.
- Tính số lượng mẫu từng nhóm, mức thay đổi Recall và chọn ví dụ minh họa theo quy tắc có thể lặp lại.
- Giải thích bằng dữ liệu và profile quan sát được; không khẳng định chắc chắn sở thích không được ghi nhận hoặc suy ra quan hệ nhân quả từ tương quan.

### Output

- **GitHub:** `src/analysis/`, `notebooks/E4_error_analysis.ipynb`, biểu đồ và `README.md` (mục E4) trình bày cách chọn trường hợp và các giới hạn.
- **Drive:** Kết quả chi tiết theo user và dữ liệu trung gian lớn trong `experiments/E4/`.

### Tiêu chí nghiệm thu

Cùng quy tắc Top-K và masking; lựa chọn trường hợp minh họa có thể lặp lại; không chỉ lựa chọn ví dụ có lợi cho một mô hình; có số lượng mẫu và mẫu số khi so sánh.

## 9. Kế hoạch 6 tuần — ưu tiên hoàn thành sớm

**Đang ở tuần 1**, tổng thời gian của project là **6 tuần: 12/10–22/11/2026**. Lịch dưới đây được thiết kế theo nguyên tắc **hoàn thành kỹ thuật sớm ở tuần 3–4, hoàn tất gói nộp ở tuần 5; tuần 6 chỉ là dự phòng**. Đây là lịch nội bộ, cần đối chiếu deadline chính thức của môn học.

| Tuần | Khoảng ngày | Mốc kết quả cần đạt (hoàn thành sớm hơn càng tốt) |
|---|---|---|
| **Tuần 1 – hiện tại** | **12–18/10** | Hoàn tất phân quyền Drive/GitHub; E1 chạy các seed còn lại; E2 code + tests; E3 hiểu rõ alignment loss; E4 có phương pháp và code khởi đầu |
| **Tuần 2** | **19–25/10** | **Khóa E1** và xuất Top-K đầy đủ; **E2 có bảng metrics hợp lệ**; E3 code + pilot; E4 so sánh Top-K được |
| **Tuần 3** | **26/10–01/11** | **Ưu tiên hoàn thành E3/E4 cốt lõi**, có bảng E1–E4, bước đầu trả lời RQ1/RQ2 và soạn report |
| **Tuần 4** | **02–08/11** | **Mốc dự phòng để khóa kết quả kỹ thuật**, xử lý review, bổ sung seed/ablation nếu cần; hoàn thành hình/bảng và bản thảo report |
| **Tuần 5** | **09–15/11** | **Hoàn chỉnh report, slides, README, code/data tái hiện và gói nộp**; tập trình bày; có thể nộp ngay khi đạt yêu cầu |
| **Tuần 6 – dự phòng** | **16–22/11** | Chỉ rà soát, sửa lỗi, kiểm tra clean-run và bảo vệ/nộp theo lịch giảng viên; **không phải tuần bắt buộc mới hoàn thành** |

**Quy tắc theo dõi:** Thành viên push/PR ngay khi một phần chạy được, cập nhật `Done / Next / Blocked` theo từng mốc; Leader nghiệm thu liên tục. Nếu thiếu GPU, ưu tiên E1 và E3 cốt lõi trước phần mở rộng. Mọi kết quả trễ hơn tuần 4 phải được báo ngay, tránh dồn vào tuần báo cáo.


## 10. Checklist Pull Request cho mọi thành viên

- [ ] Nhánh và PR ghi rõ mã task, người phụ trách, cách chạy lại.
- [ ] Code ghi rõ commit RLMRec và sử dụng đúng chia tập dữ liệu cố định.
- [ ] Không commit dataset `.pkl` gốc, checkpoint lớn, token/credentials của Drive hoặc dữ liệu chi tiết quá lớn.
- [ ] Có bằng chứng kiểm thử/smoke test và output thực tế.
- [ ] Báo cáo nêu metric, phân biệt **validation/test**, ghi rõ đường dẫn kết quả.
- [ ] PR cập nhật hướng dẫn chạy/đầu vào/đầu ra tương ứng để README phục vụ tái hiện luôn chính xác.
- [ ] Leader review PR trước khi merge vào `main`.

## 11. Việc thực hiện ngay

1. **Leader:** Chuyển ba notebook sang `notebooks/` theo Mục 2.3 (nếu chưa chuyển), rồi đưa tài liệu này lên GitHub theo đường dẫn `docs/TASK_ASSIGNMENT.md`. Sau khi họp nhóm, cập nhật tên thành viên và đối chiếu mốc nội bộ với hạn chính thức của giảng viên.
2. **Leader:** Giữ `README.md` là tài liệu tái hiện duy nhất; sử dụng `docs/EXPERIMENT_SPEC.md` để chốt masking, seeds, metrics và định dạng output. Kiểm tra `.gitignore` trước khi thành viên push code.
3. **Giao việc song song theo bảng deadline ở Mục 3.1:** Thành viên 1–2 chạy E1 và xuất Top-K các seeds còn lại; Thành viên 3 tính E2; Thành viên 4 kiểm tra và triển khai E3; Thành viên 5 bắt đầu E4 dựa trên Top-K seed 2023.


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
- Mỗi người chạy exporter và lưu ở `experiments/predictions/<model>/seed_<seed>/` tương ứng; TV3/4/5 lưu riêng tại `experiments/E2/`, `E3/`, `E4/`.
- **GitHub chứa code và hướng dẫn chạy; Drive chứa dataset, logs, checkpoint và output lớn.** Mỗi PR cần ghi đúng đường dẫn Drive để Leader tìm kết quả.
- Không chia sẻ mật khẩu/tài khoản Google cho nhau. Quyền Editor có thể xóa/ghi đè file, nên tránh đổi tên/xóa dữ liệu chung và giữ bản sao lưu những lượt chạy đã đạt.

**Ghi chú tái hiện cho người ngoài nhóm:** Quyền truy cập Drive chung chỉ phục vụ nhóm. `README.md` vẫn phải hướng dẫn tải Amazon-book từ nguồn công khai và tạo cấu trúc Drive tương đương trên tài khoản riêng; **không yêu cầu người chấm phải được cấp quyền Drive cá nhân của Leader**.
