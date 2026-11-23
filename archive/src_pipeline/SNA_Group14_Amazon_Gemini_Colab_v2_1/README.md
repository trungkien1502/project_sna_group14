# SNA Group 14 — Amazon-book + Gemini — Google Colab v2.1

**Đây là bộ mã nguồn / notebook hướng dẫn chạy trên Colab. Không phải kết quả đã train E1–E4.**

## Bắt đầu nhanh (không cần cài Python trên máy)

1. Tải `SNA_Group14_Amazon_Gemini_Colab_v2_1.zip`; **upload đúng ZIP này lên `MyDrive/`**.
2. Trên [Google Colab](https://colab.research.google.com/), chọn **File → Upload notebook**, mở `notebooks/01_START_HERE_Colab.ipynb` (có thể tải notebook riêng ngoài ZIP).
3. Sửa **ô cấu hình đầu tiên**: `ZIP_ON_DRIVE`, `ORIGINAL_PROJECT_ROOT`, `RUN_ID`, `RAW_METADATA`, `RAW_REVIEWS`, `REVIEW_POLICY`, `GEMINI_MODEL` và `EMBEDDING_PROVIDER`. Nếu upload ZIP ở vị trí khác, sửa đường dẫn.
4. Chạy lần lượt từng ô: Drive mount → clone repo đúng commit → `audit` split → `index-raw` → build item prompts → Gemini item profiles → user prompts/profiles → embeddings → `package` → `preflight`.
5. Khi đã có package PASS, mở `notebooks/02_PREPARE_STAGE_Colab.ipynb` trên Colab. Chọn `STAGE`, `MODEL`, `SEED`, `MODE`, `LAYER`, chạy từng ô để **tạo file notebook** vào `experiments_llm_colab_v21/<RUN_ID>/runtime/notebooks/` trên Drive.
6. Mở **từng notebook mới sinh ra** trên Colab và chọn **Runtime → Run all**; chọn **T4/A100 GPU** cho các bước training/export phù hợp; xem log và xác nhận output trước khi chuyển stage tiếp.

**Cần hiểu đúng:** `01_START_HERE` có các cờ `DO_*` mặc định `False` để không vô tình gọi API hay GPU. `02_PREPARE_STAGE` **không train**; nó chỉ chuẩn bị notebook chạy gốc được vá đường dẫn riêng cho nhánh Gemini.

## Nguồn đối chiếu

- Source tác giả: [HKUDS/RLMRec](https://github.com/HKUDS/RLMRec), commit `22413752246de3dee8ab0d509f7f7a8889080f95`.
- Source nhóm: [trungkien1502/project_sna_group14](https://github.com/trungkien1502/project_sna_group14), commit `c17caeb4448a8add7939510411339af918d258bb`.
- Dataset nhóm: `MyDrive/Data_SNA_Group14/data/raw/amazon/`, 7 file RLMRec cũ (ba ma trận + bốn file semantic). Ba ma trận giữ nguyên. **Không dùng semantic cũ cho nhánh Gemini.**
- Raw metadata/review: **không có sẵn trong 7 file**; cần nguồn Amazon gốc khớp ASIN/reviewerID qua `data/mapper/amazon_item.json`, `data/mapper/amazon_user.json`.

## Nghiên cứu và kiểm soát leakage

- Không tự tách lại train/validation/test; kiểm tra chính xác shape, nnz, giao cặp user–item, SHA256.
- `REVIEW_POLICY='none'` (mặc định): **bỏ hẳn review text** để không đưa dữ liệu cấp event không rõ split vào prompt; cách này **khác prompt gốc** ở trường review và không phải reproduction đúng từng tín hiệu.
- `REVIEW_POLICY='train_pair'`: chỉ dùng review mapping tới cặp `(user,item)` thuộc ma trận train, loại review thuộc validation/test pair và bản trùng không xác định. **Chưa chứng minh được event-level train-only** nếu split gốc không ghi review event/timestamp. Phải báo cáo điều này.
- Item prompt giữ đúng instruction Amazon `title` và `description`; **không tự lấy review vào item prompt** nếu thiếu description. Chúng ta chưa có recipe đầy đủ author-side cho fallback, phải ghi khác biệt.
- System prompt Amazon được copy chính xác từ `generation/instruction/amazon_item.txt` và `amazon_user.txt`; prompt *body* là phần reconstruction minh bạch.
- Đầu ra LLM: JSON `summarization`/`reasoning`, giới hạn word count theo instruction.
- Chỉ dùng Gemini profiles/embeddings mới trong nhánh; điều kiện embedding `openai` giữ `text-embedding-ada-002` như paper (có thể tốn phí/không còn khả dụng), `local` là **thay đổi thêm yếu tố embedding** nên không được quy mọi cải thiện cho Gemini.

## Bộ stages thực nghiệm gốc (14 source notebooks, 34 lượt gọi stage)

| Stage | Source notebook gốc nhóm | Ghi chú |
| --- | --- | --- |
| E1 train | `03_E1_Full_Training.ipynb` | 2 models × 3 seeds; chạy từng notebook native |
| E1 export | `04_E1_Export_TopK.ipynb` + `src/inference/export_topk.py` | 6 Top-K |
| Most Popular | `05_E1_Most_Popular.ipynb` | baseline |
| E1 summary | `06_E1_Summary.ipynb` | sau đủ E1 |
| E2 | `07_E2_Popularity_Bias.ipynb` | đọc Top-K E1 mới |
| E3A | `08_E3A_Semantic_Alignment_Ablation_FIXED.ipynb` | user_only/item_only × 3 seeds |
| E3B | `10_E3B_Popularity_Bias_Analysis.ipynb` | đọc E1 mới + E3A mới |
| E3C train | `11_E3C_Semantic_Weight_Sensitivity_FIXED_v3.ipynb` | weight 0/.005/.02/.05 × 3; weight .01 reuse E1 **mới** |
| E3C export | `E3C_Export_TopK_Fixed.ipynb` | 15 configs |
| E3C analysis | `12_E3C_Analysis_Weight_Sensitivity.ipynb` | baseline lấy từ E1 nhánh mới |
| E3D | `E3D_Semantic_Collaborative_Agreement_Drive.ipynb` | đọc embeddings Gemini mới |
| E4 train | `E4_RQ3_Training_FIXED_Drive_Checkpoints.ipynb` | 2 models × 3 layers, mỗi notebook vòng 3 seeds |
| E4 export | `E4_RQ3_Export_TopK_FIXED.ipynb` | recovery-only new branch |
| E4 analysis | `E4_RQ3_Analysis.ipynb` | 18 Top-K và QA |

## Không được hiểu là đã kiểm thử end-to-end

- **Đã kiểm thử offline:** Python test suite 17/17 PASS (synthetic), soạn hai notebook Colab có cú pháp Python hợp lệ, packing 7 file mock đúng schema, chặn branch mock trong training, chặn chạy `nbclient` tự động, giữ source notebook bất biến khi stage.
- **Chưa thực thi thật:** Gemini API (không có key), embedding API/local full 20k profiles, raw Amazon release chưa gắn vào nhánh, GPU training/E1–E4 trên Colab chưa chạy thử. `Run all` từng stage có thể dừng vì Colab runtime/package CUDA hoặc notebook gốc; không đánh dấu thành công trước khi QA PASS.
- Output cũ `MyDrive/Data_SNA_Group14/experiments/` **không bị xóa/ghi đè**. Nhánh mới `MyDrive/Data_SNA_Group14/experiments_llm_colab_v21/<RUN_ID>/`.

Chi tiết: `RUN_COLAB.md`, `docs/V2_1_IMPLEMENTATION_REPORT.md`.
