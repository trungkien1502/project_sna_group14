# Báo cáo bàn giao — SNA Group 14 Amazon-book Gemini Colab v2.1
**Ngày:** 21/11/2026  
**Trạng thái:** Mã nguồn/Notebook build hoàn thành, **chưa nghiệm thu Colab GPU end-to-end**.

## Phạm vi đối chiếu

Đã dùng hai GitHub repo làm chuẩn: [HKUDS/RLMRec](https://github.com/HKUDS/RLMRec) tại `22413752246de3dee8ab0d509f7f7a8889080f95` và [trungkien1502/project_sna_group14](https://github.com/trungkien1502/project_sna_group14) tại `c17caeb4448a8add7939510411339af918d258bb`. Sử dụng đúng 14 stage theo các notebook đã kiểm kê E1–E4. Drive nhóm đã xác nhận 7 file `.pkl` Amazon-book, các thư mục output E1–E4, E3C export 15/15 và E4 Top-K QA 18/18 **của nhánh GPT-3.5 cũ**; chúng không phải output Gemini và không được tự động tái sử dụng.

## Điểm sửa chính từ v2

1. Thay `QuickStart_Colab.ipynb` kiểu nhập thủ công đường dẫn code bằng **`01_START_HERE_Colab.ipynb`**: mount Drive; upload/giải nén ZIP; clone/checkout source SHA; config trong một ô; audit/index/profile/embed/package theo thứ tự và các cờ xác nhận API.
2. Thêm **`02_PREPARE_STAGE_Colab.ipynb`**: chọn stage/model/seed/mode/layer → xuất notebook riêng vào Drive, không gọi GPU ngầm.
3. `sna.experiments.stage_source` chèn **prelude chạy trong native Colab kernel**, kiểm tra Drive mount, dữ liệu đóng gói mới, cả 7 files, source SHA; thay đường dẫn output gốc bằng branch riêng.
4. `run-stage --execute` và `run-all --execute` được **cấm** để không giả vờ nested `nbclient` là tương đương Colab (không có kiểm thử thật). Thay bằng mở notebook `.ipynb` đã chuẩn bị và `Runtime > Run all`.
5. Config mới dùng nhánh `experiments_llm_colab_v21/<RUN_ID>`; không sửa dữ liệu, checkpoint hay Top-K trong `experiments/` cũ.
6. `REVIEW_POLICY='none'` là mặc định *nghiêm ngặt hơn* khi không có provenance cấp review. `index-raw` không yêu cầu raw review file nếu không dùng text review. `train_pair` là chế độ opt-in, chưa chứng minh event-level leakage-free.
7. API Gemini ledger ghi **từng network attempt** kể cả retries; giới hạn requests là cap nội bộ UTC, không đảm bảo quota API thực tế.
8. Hai notebook UI và generated native stage cell được kiểm cú pháp; kiểm thử offline chạy lại.

## Kiểm thử thực hiện

- Unit/integration tests synthetic: **17/17 PASS** (Python 3.13 offline, không gọi API/GPU).
- Kiểm toán split/mapper; loại holdout pair reviews; khóa prompt; đóng gói schema RLMRec đủ 7 file cho mock; chặn mock package training; staging không sửa source; test notebook syntax; không gọi nested-kernel execution.
- **Không chứng minh được:** khả dụng model theo API key/tier thực tế của người dùng; raw metadata khớp release Amazon và review provenance; 9332+11000 profile thật; embeddings 20k thật; PyTorch/Colab CUDA compatibility; metrics E1–E4 nhánh mới.

## Điều kiện chặn khi chưa có dữ liệu

- Nếu thiếu raw metadata Amazon có ASIN khớp mapper ⇒ **dừng trước `index-raw`**, không được dùng `itm_prf.pkl` làm raw.
- Nếu `REVIEW_POLICY='train_pair'` mà thiếu review JSONL đúng release ⇒ dừng; dù có file cũng phải báo cáo giới hạn split event-level.
- Nếu chưa có 9332 item + 11000 user profiles và 2 embeddings arrays mới ⇒ không `package PASS`, không training.
- Nếu Git SHA nhóm bị đổi từ `c17caeb...` ⇒ dừng để đối chiếu thay đổi và cập nhật manifest, không bỏ qua commit pin.
- Nếu notebook E1/E3/E4 trên Colab lỗi ở runtime ⇒ dừng, giữ logs, không gán PASS giả.

## Mapping input và notebook

| Nguồn | Đích mới | Ghi chú |
|--|--|--|
| `MyDrive/Data_SNA_Group14/data/raw/amazon/trn_mat.pkl` | `<BRANCH>/data/raw/amazon/trn_mat.pkl` | Copy byte-identical, SHA256 |
| `val_mat.pkl` + `tst_mat.pkl` | tương tự | Không resplit |
| Raw Amazon metadata + ASIN mapper | `<BRANCH>/raw_index/books.jsonl` | 9,332 items, coverage threshold |
| Raw reviews optional + reviewer mapper | `<BRANCH>/raw_index/train_reviews.jsonl` | none hoặc train_pair |
| Prompt Amazon | `<BRANCH>/prompts/{item,user}` | Giữ system instruction của paper |
| Gemini item & user | `<BRANCH>/profiles/{item,user}` | riêng từng ID và prompt hash |
| embeddings | `<BRANCH>/embeddings/{item,user}` | local / original Ada embedding |
| `usr_prf.pkl`,`itm_prf.pkl`,`usr_emb_np.pkl`,`itm_emb_np.pkl` | `<BRANCH>/data/raw/amazon/` | **Tạo mới, không copy từ author** |
| Group notebooks E1–E4 | `<BRANCH>/runtime/notebooks/` | Staged, chạy native Colab |
| Group outputs E1–E4 | `<BRANCH>/experiments/` | Không đụng outputs cũ |

## Kết luận nghiêm ngặt

**v2.1 là Colab-native notebook preparation + pipeline code được kiểm thử offline**, không phải "chạy thành công E1–E4 trên Colab". Chỉ coi từng stage là hoàn thành khi notebook đó chạy native trong Colab thật và output QA được đối chiếu với protocol/metrics đã khóa. Đọc `RUN_COLAB.md` trước khi dùng API hoặc GPU.

## Kiểm soát số lượt gọi API

- Thời gian tính cap tự đặt theo ngày UTC; nếu Google AI Studio có giới hạn khác, hệ thống Google vẫn có thể trả 429 sớm hơn.
- File `<RUN_ID>/logs/api_attempts.jsonl` lưu một dòng cho **từng lần thử gọi Gemini**, kể cả retries; có model/prompt hash nhưng không chứa API key.
- Với 20.332 profiles và cap nội bộ 400 request/ngày, thời gian thực tế có thể rất dài và còn phụ thuộc retry/quota thực tế. Không cam kết đủ miễn phí hay tốc độ.
- Mọi yêu cầu API chỉ chạy sau khi người dùng bật flag `DO_*` và nhập API key trong Colab.
