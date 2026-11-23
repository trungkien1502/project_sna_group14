# Notebooks

Thư mục này chỉ chứa các notebook đã tạo ra kết quả đang dùng trong báo cáo. Kết quả nằm trên Google Drive tại `MyDrive/Data_SNA_Group14/experiments/`.

| Notebook | Thí nghiệm | Kết quả trên Drive |
|---|---|---|
| 01_data_inspection | Kiểm tra dữ liệu Amazon-book | |
| 02_E1_LightGCN_Setup | E1: smoke test | `E1/` |
| 03_E1_Full_Training | E1: train LightGCN / RLMRec-Con, 3 seeds | `E1/<model>/seed_*` |
| 04_E1_Export_TopK | E1: xuất Top-20 | `predictions/<model>/seed_*` |
| 05_E1_Most_Popular | E1: baseline Most Popular | `predictions/most_popular` |
| 06_E1_Summary | E1: bảng tổng hợp | `E1/summary` |
| 07_E2_Popularity_Bias | E2: ARP, APLT, Head/Tail Recall, Spearman | `E2/popularity_bias` |
| 08_E3A_Semantic_Alignment_Ablation | E3A: user-only / item-only / full | `E3/runs` |
| 09_Audit | Kiểm tra 13 file Top-K của E1 + E3A | `audit_topk_20261029` |
| 10_E3B_Popularity_Bias_Analysis | E3B: popularity bias của ablation | `E3B/popularity_bias` |
| 11_E3C_Semantic_Weight_Sensitivity | E3C: train theo kd_weight | `E3C/` |
| 11b_E3C_Export_TopK | E3C: xuất Top-20 | `E3C/` |
| 12_E3C_Analysis_Weight_Sensitivity | E3C: phân tích | `E3C/analysis` |
| 12b_E3D_Semantic_Collaborative_Agreement | E3D: semantic–collaborative agreement | `E3D/agreement_analysis` |
| 13a_E4_Graph_Depth_Training | E4: train 1/2/3 layer | `E4/recovery_runs` |
| 13b_E4_Export_TopK | E4: xuất Top-20 | `E4/exports` |
| 13c_E4_Analysis | E4: phân tích | `E4/analysis` |
| 14_E2_Popularity_Groups | Góp ý 2: 5 nhóm độ phổ biến | `E2/popularity_groups` |
| 15_E3_Error_Analysis | Góp ý 3: error analysis LightGCN và RLMRec-Con | `E3/error_analysis` |
| 16_E3_Shuffled_Embedding_Control | Góp ý 6: đối chứng xáo embedding | `E3/shuffle_control` |

Ghi chú:
- `E4/runs` trên Drive là lần train E4 đầu tiên. Lần đó không lưu được checkpoint nên không dùng. Kết quả E4 lấy từ `E4/recovery_runs`.
- Các bản cũ và bản thử nghiệm đã chuyển sang `archive/`.
