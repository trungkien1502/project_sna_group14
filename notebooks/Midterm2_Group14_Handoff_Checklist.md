# Group 14 — Checklist bàn giao Midterm 2 (A.25, RQ1–RQ3)

**Hạn dự kiến:** 25/11/2026, 21:00. **Gói nộp:** `SNA-Group14-A.25-Recommendation-Midterm2.zip`. Đối chiếu thông báo môn học mới nhất trước khi nộp.

## A. Pipeline kỹ thuật (phải làm theo thứ tự)

- [ ] **Train/recovery E4**: 2 models × 3 layers × 3 seeds = 18 run hợp lệ; giữ nguyên E4/runs cũ. Lưu checkpoint **theo model/layer/seed** ở `E4/recovery_runs/`; kiểm tra file tồn tại, `torch.load` được, SHA256 và config khớp.
- [ ] **Smoke test 1 run**: xác nhận checkpoint thực sự tồn tại trên Drive sau khi kết thúc Colab; chỉ sau đó mới chạy các run còn lại.
- [ ] **Export E4 Top-K**: dùng `E4_RQ3_Export_TopK_SAFE.ipynb`; xác nhận 18/18 `export_passed` hoặc `already_verified`; kiểm tra train-only masking, không trùng, đúng user/item, scores, provenance, đối chiếu Recall/NDCG với log.
- [ ] **Evaluate/Analyze E4**: dùng `E4_RQ3_Evaluation_Analysis_SAFE.ipynb`; QA PASS 18/18, tạo CSV per-run, mean±SD, paired differences, 14 PNG, manifest.
- [ ] **Audit**: kiểm tra số liệu E1/E2/E3A/E3B/E3C đã có, nhất quán dataset/split/seed/K và metric definitions; không tự động thay thế run E4 bằng E1.

## B. Những thứ giao cho người viết báo cáo

1. **Nguồn đề bài và báo cáo trước**: Mini Proposal, Midterm 1, hướng dẫn Midterm 2 của giảng viên.
2. **Protocol thí nghiệm**: Amazon Books (11,000 users, 9,332 items); train/test matrices; ranking toàn bộ item, train-only mask; seeds 2023/24/25; K=10/20; định nghĩa head = top ceil(20%) theo train degree; model và hyperparameters, commit nguồn.
3. **Kết quả E1/E2**: bảng baseline Most Popular, LightGCN, RLMRec-Con, mean±SD; Recall/NDCG, ARP/APLT, Head/Tail Recall, Coverage, Spearman.
4. **Kết quả E3**: user-only, item-only, full (E3A), phân tích popularity (E3B), sensitivity semantic weight 0/.005/.01/.02/.05 (E3C). Ghi rõ RQ2 semantic–collaborative agreement chưa hoàn thành nếu chưa làm.
5. **Kết quả E4**: 18 configs; file `E4_metrics_per_run.csv`, `E4_summary_mean_sd.csv`, `E4_paired_model_deltas.csv`, `E4_paired_depth_deltas.csv`, biểu đồ, `E4_topk_inventory_QA.csv`, `E4_analysis_manifest.json`.
6. **Bằng chứng tái lập**: notebooks train/export/analysis, `config_snapshot.yml`, training logs, checkpoint verification, export provenance/hash, QA, phiên bản source code.
7. **Tình trạng và giới hạn**: ghi rõ checkpoint E4 ban đầu không được lưu do lỗi thu thập artifact; các recovery runs là lần train mới, không coi là kết quả của cùng lần train; kết luận dựa trên kết quả đã QA.

## C. Phân công gợi ý

| Vai trò | Việc cần làm | Đầu ra |
|---|---|---|
| Nhóm chạy E4 | Hoàn thành checkpoint + export, báo trạng thái | 18/18 QA PASS |
| Phụ trách phân tích | Chạy notebook evaluation, kiểm tra plots/paired deltas | Thư mục E4/analysis timestamp |
| Người viết phần phương pháp | Mô tả mô hình, protocol, thiết kế E1–E4 | Methods + Experiment Setup |
| Người viết kết quả | Bảng, biểu đồ, nhận xét RQ1/RQ2/RQ3 | Results & Discussion |
| Người tổng hợp | Kiểm tra tính nhất quán, hạn chế, phụ lục, tài liệu tham khảo | PDF cuối + ZIP |

## D. Khung nội dung Midterm 2

1. Tóm tắt thay đổi kể từ Midterm 1, mục tiêu RQ1–RQ3 (tránh lặp nguyên văn proposal).
2. Dataset/split, preprocessing, baselines, mô hình, thiết lập thực nghiệm, metric definitions, reproducibility.
3. E1/E2: so sánh baseline và đánh đổi accuracy–bias.
4. E3A/E3B/E3C: semantic alignment ablation, popularity, semantic weight sensitivity; nêu rõ giới hạn RQ2.
5. E4: depth 1/2/3, LightGCN vs RLMRec-Con, mean±SD và paired seed differences, nhận định H3.
6. Thảo luận đa chiều về bias: ARP/APLT, TailRecall, Coverage, Spearman có thể không cùng hướng.
7. Hạn chế, việc chưa hoàn thành, kế hoạch sau Midterm 2, tài liệu tham khảo, phụ lục code/QA.

## E. Quy tắc kết luận

- Không dùng training loss để thay cho ranking metrics.
- Không kết luận “bias giảm toàn diện” chỉ vì ARP thấp hơn hoặc APLT cao hơn.
- Không gọi paired differences 3 seeds là kiểm định ý nghĩa thống kê.
- Không khẳng định RQ2 semantic–collaborative agreement đã được giải quyết nếu chưa có phân tầng theo agreement.
- Không ghi E4 hoàn tất nếu chưa có 18 exports QA PASS và bảng phân tích.

## F. Kiểm tra trước khi nộp

- [ ] Mọi bảng/biểu đồ có nhãn K, model, layer, mean±SD và nguồn dữ liệu.
- [ ] Kết luận khớp số liệu thực, nêu cả trường hợp trái giả thuyết.
- [ ] PDF cập nhật nối tiếp Midterm 1; ZIP đúng tên; link Drive/code/log truy cập được.
- [ ] Một người không trực tiếp chạy thí nghiệm đọc lại phương pháp và thử đối chiếu một run.
