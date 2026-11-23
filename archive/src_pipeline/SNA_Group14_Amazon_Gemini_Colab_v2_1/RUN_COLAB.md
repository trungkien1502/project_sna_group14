# HƯỚNG DẪN CHẠY SNA GROUP 14 TRÊN GOOGLE COLAB (v2.1)

**Mở `notebooks/01_START_HERE_Colab.ipynb` bằng Colab, không chạy `python run.py ...` bằng terminal trên máy cá nhân.** Các lệnh Python là mã nguồn notebook gọi phía sau.

## Chuẩn bị dữ liệu

- Trên Drive đã có `MyDrive/Data_SNA_Group14/data/raw/amazon/` chứa đúng `trn_mat.pkl`, `val_mat.pkl`, `tst_mat.pkl`, `usr_prf.pkl`, `itm_prf.pkl`, `usr_emb_np.pkl`, `itm_emb_np.pkl`.
- Cần raw *metadata* sách Amazon có ASIN để build profiles mới. Bản ZIP không kèm raw Amazon. Ở chế độ mặc định `REVIEW_POLICY='none'`, không cần raw review file. Muốn sử dụng review (gần paper hơn) mới yêu cầu file reviews và phải chấp nhận giới hạn event-level leakage.
- Tải ZIP mới và notebook `01_START_HERE_Colab.ipynb` theo link của ChatGPT. Upload ZIP mới vào **MyDrive root**, giữ đúng tên ZIP.

## Runbook Colab cụ thể

1. Vào `https://colab.research.google.com/` → `File > Upload notebook` → mở `01_START_HERE_Colab.ipynb`.
2. Chỉnh ô `ZIP_ON_DRIVE`, `RUN_ID`, `RAW_METADATA`, `REVIEW_POLICY`, `GEMINI_MODEL`, `EMBEDDING_PROVIDER` trước khi chạy. `ZIP_ON_DRIVE` mặc định `'/content/drive/MyDrive/SNA_Group14_Amazon_Gemini_Colab_v2_1.zip'`.
3. Chạy các ô theo thứ tự: mount → unzip → verify clones → config → `audit` → `index-raw` → `build-prompts item`.
4. Nếu chưa có metadata, **dừng ở index-raw**; không cố tạo item prompts bằng `usr_prf.pkl`/`itm_prf.pkl` cũ.
5. Bật `DO_ITEM_PILOT=True`; dùng Colab Secrets `GEMINI_API_KEY` hoặc nhập bằng getpass; xem output 3 items; nếu đúng mới bật `DO_ITEM_BATCH=True` để tạo dần 9332 items.
6. Sau đủ item profiles, bật `DO_BUILD_USER_PROMPTS=True`; bật `DO_USER_PILOT` sau đó `DO_USER_BATCH` cho 11000 users. Tất cả output trên Drive, chạy nhiều phiên vẫn bỏ qua profiles hoàn tất.
7. Chọn một embedding experiment: `local` hoặc `openai` (có phí/kiểm tra khả dụng model); bật `DO_EMBED=True` để tạo item và user embeddings.
8. Bật `DO_PACKAGE=True` để đóng gói 7 file RLMRec **trong nhánh mới**; chỉ tiếp tục khi `package` + `preflight` PASS.
9. Mở `02_PREPARE_STAGE_Colab.ipynb` trên Colab; điền cùng RUN_ID và ZIP, chọn `STAGE`, `MODEL`, `SEED`, `MODE`, `LAYER`, bấm chạy. Notebook này sinh 1 notebook vào `<RUN_ID>/runtime/notebooks/` trên Drive.
10. Mở notebook được sinh ra từ Google Drive bằng Google Colab → bật GPU nếu cần → `Runtime > Run all`, xem output/metrics/checkpoint PASS. Quay lại manager chuẩn bị lượt tiếp theo, đến đủ E1–E4.

`02_PREPARE_STAGE` có ô in status `run-all` nhưng chỉ để xem kế hoạch; `run-all --execute` đã **bị vô hiệu hóa** vì không xác minh được thực thi notebook qua nested kernel trên Colab.

## Quy tắc vận hành

- Khởi động mỗi notebook training bằng **runtime Colab mới** nếu thư viện RLMRec `config.configurator` đã được import trước đó; code gốc yêu cầu môi trường sạch.
- Source RLMRec pin commit `22413752246de3dee8ab0d509f7f7a8889080f95`; group repo `c17caeb4448a8add7939510411339af918d258bb`; nếu repo đã update, chốt lại SHA nghiên cứu trước khi đổi run ID, không reset ngầm.
- `RUN_ID` duy nhất cho mỗi protocol/model/embedding. Giữ `RUN_ID` cũ khi tiếp tục API quota với cấu hình y hệt; thay RUN_ID khi sửa raw dataset/prompt/model/embedding.
- **Chỉ sau QA PASS** mới ghi kết quả E1–E4 vào báo cáo. Bản này chưa có GPU Colab integration PASS.
- Mặc định `EMBEDDING_PROVIDER='local'` là PILOT thay đổi cả LLM & embedding; để thử riêng tác động LLM cần cân nhắc giữ encoder `text-embedding-ada-002` của tác giả, nếu còn được cấp quyền.
- Có thể thay giới hạn requests trong Colab nhưng đây chỉ là **cap nội bộ tính theo UTC**, không phải quota chính thức Google. Xem quotas/tier ở Google AI Studio.
