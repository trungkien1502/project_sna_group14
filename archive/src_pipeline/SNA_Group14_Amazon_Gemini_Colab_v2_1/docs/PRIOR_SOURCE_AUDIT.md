# SNA Group 14 — File-to-Experiment Audit (21/11/2026)

## Source scope and identity
- Group repo: https://github.com/trungkien1502/project_sna_group14 ; inspected main at `c17caeb4448a8add7939510411339af918d258bb` (2026-11-21). 24 `.ipynb`; 35 tracked blob files (excluding trees). The repo's README is only 44 bytes, not a functioning execution guide.
- Paper repo: https://github.com/HKUDS/RLMRec ; inspected pinned/main commit `22413752246de3dee8ab0d509f7f7a8889080f95`.
- Artifacts: https://drive.google.com/drive/folders/1t-T93iTwoA5GDkLaVL6tNw-004XCbzeU (Data_SNA_Group14). `data/raw/amazon/` contains 7 `.pkl` files listed below.
- **Evidence levels:** code file present ≠ notebook actually executed. Drive output/pass files provide execution evidence but do not necessarily identify the exact notebook or git commit used. "PRIMARY" below is the recommended canonical entrypoint for a new run; **not** historical certainty.

## Graph: actual dependencies used by group notebook code
1. **E1:** `trn_mat.pkl`, `val_mat.pkl`, `tst_mat.pkl` + old `usr_emb_np.pkl` and `itm_emb_np.pkl` loaded by paper `configurator.py`; RLMRec-con model consumes embeddings, LightGCN does not directly consume them. Run `03_E1_Full_Training.ipynb`; checkpoints and `metrics.json` live under `experiments/E1/{lightgcn,lightgcn_plus}/seed_{2023,2024,2025}`. E1 TopK via `04_E1_Export_TopK.ipynb` -> group `src/inference/export_topk.py`. Popularity baseline via `05_E1_Most_Popular.ipynb`; summary via `06_E1_Summary.ipynb`.
2. **E2:** `07_E2_Popularity_Bias.ipynb` consumes seven TopK sets (one Most Popular + 3 LightGCN + 3 RLMRec-Con) and train/test matrices; E1 summary optional. Outputs `experiments/E2/popularity_bias/`.
3. **E3A:** `08_E3A_Semantic_Alignment_Ablation_FIXED.ipynb` runs RLMRec-Con ablations. Drive shows user_only/item_only × three seeds at default E1 weight; E1 LightGCN/full are references rather than necessarily new graph_only/full runs.
4. **E3B:** `10_E3B_Popularity_Bias_Analysis.ipynb` reads E1 full TopK + E3A user/item TopK, train/test, E2 metrics; outputs E3B plots/CSV.
5. **E3C:** `11_E3C_Semantic_Weight_Sensitivity_FIXED_v3.ipynb` trains weight 0/.005/.02/.05 × seeds 2023/24/25; weight .01 reuses E1 **from the same profile generation run**. Export via `E3C_Export_TopK_Fixed.ipynb`, analyze via `12_E3C_Analysis_Weight_Sensitivity.ipynb`. Drive `E3C_export_status.csv` has 12 export_passed and 3 reused_E1_passed; analysis JSON status `ANALYSIS_QA_PASS`, 15 TopK, 30 metric rows, date 2026-11-13.
6. **E3D:** `E3D_Semantic_Collaborative_Agreement_Drive.ipynb` reads train/test matrices, `usr_emb_np.pkl`, `itm_emb_np.pkl`, E1 LightGCN/full TopK and E3A user/item TopK; outputs `E3D_agreement_groups.csv`, `E3D_seed_group_metrics.csv`, `E3D_group_summary.csv`, `E3D_continuous_correlations.csv`, `E3D_method.json`. Colab edition is mostly delivery/ZIP alternative.
7. **E4:** `E4_RQ3_Training_FIXED_Drive_Checkpoints.ipynb` training 2 models × 3 layers × 3 seeds to `experiments/E4/recovery_runs/`; export 18 TopK via `E4_RQ3_Export_TopK_FIXED.ipynb` to `experiments/E4/exports/`; analysis candidate `E4_RQ3_Analysis.ipynb`. Drive analysis JSON `QA_PASS_18_OF_18` dated 2026-11-18; inventory CSV has 18 PASS and SHA256 per TopK.

## Paper source code — which files matter
- `generation/instruction/amazon_item.txt`, `generation/instruction/amazon_user.txt`: AMAZON **system prompt instructions** to retain in the new profile pipeline.
- `generation/item/generate_profile.py`, `generation/user/generate_profile.py` and `generation/item/item_system_prompt.txt`, `generation/user/user_system_prompt.txt`: **Yelp demo** in the public repo; these are not complete Amazon batch prompt builders. Do not substitute their Yelp prompt for Amazon.
- `generation/emb/generate_emb.py`: single-example use of `text-embedding-ada-002` from `summarization`, not complete mass embedding pipeline; keep as evidence, not direct mass runner.
- `data/mapper/amazon_item.json` maps processed `iid` to original Amazon `asin`; `data/mapper/amazon_user.json` maps `uid` to original `reviewerID`. Required to match original book metadata/reviews to the released matrix IDs for raw-profile regeneration.
- `encoder/config/configurator.py`: **unconditionally loads** `data/amazon/usr_emb_np.pkl` and `itm_emb_np.pkl` for the given model, including the LightGCN config. Loading does not mean LightGCN trains against those embeddings.
- `encoder/data_utils/data_handler_general_cf.py`: reads the 3 matrices and creates train graph, validation and test loaders.
- `encoder/models/general_cf/lightgcn.py` and `lightgcn_plus.py`: baseline and semantic-alignment backbone.
- `encoder/config/modelconf/lightgcn.yml` / `lightgcn_plus.yml`: per-model hyperparameters; do not overwrite source commit in place.
- `encoder/train_encoder.py`, `encoder/trainer/trainer.py`, `encoder/trainer/metrics.py`: command, training/checkpoint, metrics source.
- `data/read_profile.py`: print example profiles; *not* an experiment runner.

## Current dataset file roles
| File in `Data_SNA_Group14/data/raw/amazon/` | Purpose | New Gemini pipeline |
|---|---|---|
| `trn_mat.pkl` | Train interaction matrix | **Keep unchanged**, build graph and define allowed interaction/review IDs |
| `val_mat.pkl` | Validation labels | **Keep unchanged**, only tune/checkpoint selection |
| `tst_mat.pkl` | Test labels | **Keep unchanged**, final evaluation only |
| `usr_prf.pkl` | Author GPT-3.5 user profiles | Archived historical baseline, **never prompt input** for Gemini |
| `itm_prf.pkl` | Author GPT-3.5 item profiles | Archived historical baseline, **never prompt input** for Gemini |
| `usr_emb_np.pkl` | Author user embeddings | Old baseline; **replace with Gemini-generated embedding artifact** for new RLMRec-Con branch |
| `itm_emb_np.pkl` | Author item embeddings | Old baseline; **replace with new artifact** for new RLMRec-Con branch |

Only these 7 files are in the inspected group `data/raw/amazon/` Drive folder. Original raw Amazon title/description/review text **not present here**. The mapper files in the author's GitHub give identifiers, not the required original text. Must retrieve raw source and join via mapper **then gate every review/interacted item against train**; otherwise profile leakage claims are not established.

## Notebooks categorized (24 total)
`PRIMARY` = recommended for new run; `LEGACY_CANDIDATE` = preserve for traceability, not canonical runner; `SUPPORT` = checks/setup only. No destructive deletion recommended.

| Experiment | Canonical files | Older/alternative files |
|---|---|---|
| E1 | 03 full training; 04 export + `src/inference/export_topk.py`; 05 Most Popular; 06 summary | 01 inspection/02 setup support |
| E1/E3A QA | 09_Audit | — |
| E2 | 07_E2_Popularity_Bias | — |
| E3A | 08_E3A_Semantic_Alignment_Ablation_FIXED | — |
| E3B | 10_E3B_Popularity_Bias_Analysis | — |
| E3C | 11_E3C_Semantic_Weight_Sensitivity_FIXED_v3; E3C_Export_TopK_Fixed; 12_E3C_Analysis_Weight_Sensitivity | 11_E3C_Semantic_Weight_Sensitivity (old); E3C_Colab (unrelated orphan workflow that expects `E3C_agreement_analysis.py`, `users.csv`, `metrics.csv`, absent in current GitHub tree) |
| E3D | E3D_Semantic_Collaborative_Agreement_Drive | E3D_Semantic_Collaborative_Agreement_Colab (same analysis, ZIP wrapper) |
| E4 | E4_RQ3_Training_FIXED_Drive_Checkpoints; E4_RQ3_Export_TopK_FIXED; E4_RQ3_Analysis (candidate) | 13_E4_Graph_Depth_RQ3; E4_RQ3_6_Members_Training; E4_RQ3_Training_Choose_Model_Layer (1); E4_RQ3_Export_TopK_SAFE |

### Historical uncertainty and documentation drift
- GitHub notebooks often have cleared execution counts and outputs. Existence of matching Drive output does not prove which notebook revision produced it. Verify against run logs, config snapshots, source hashes or timestamps when available.
- `docs/EXPERIMENT_SPEC.md` says "DỰ THẢO CHỜ LEADER CHỐT" and proposes some nonexistent code paths, such as `src/evaluation/popularity_metrics.py`, `src/ablation/`, `src/analysis/` and tests; they do **not** exist in inspected group main tree.
- `notebooks/Midterm2_Group14_Handoff_Checklist.md` names `E4_RQ3_Evaluation_Analysis_SAFE.ipynb`, but this file is **not in current group GitHub main**. Do not list it as available executable.
- E4 `runs/` and `recovery_runs/` are distinct training attempts. The approved E4 exporter uses only recovery checkpoints.
- E3C .01 baseline reuses E1 only within the *same dataset/split/embedding-profile branch*. When switching LLM, don't mix old author GPT-3.5 E1 TopKs with Gemini training.
- New profiles should go under a separate immutable run root, e.g., `experiments_gemini/<run_id>/`, preserving all existing E1–E4 results and hashes.
- **Data-leakage checks are separate from train-only ranking masking**; a TopK QA PASS does not demonstrate LLM prompt sources excluded validation/test.

## Next verification gates before implementing a new end-to-end pipeline
1. Find and checksum original Amazon book metadata + review text, align `asin` and `reviewerID` through GitHub mapper JSON to 11,000×9,332 internal IDs; document missing/duplicate joins.
2. Freeze the exact 3 matrix splits, their hashes and ID maps. Restrict user history to train; if item description unavailable and reviews used, restrict review sources to eligible train interactions (or explicitly label another protocol).
3. Save every item/user prompt, allowed input IDs, LLM model/version/response/usage and deterministic entity ordering. Use paper Amazon instruction text, not demo Yelp instructions.
4. Generate new semantic embedding files with matching array row IDs and expected model loader shape; test no cross-branch file confusion.
5. Run E1 full new branch; export QA, E2, E3A/B/C/D, E4 in the dependency order above and preserve input hashes/commit at each step.

**Audit status:** GitHub source trees + selected notebook code + Drive folder listings + E3C export status + E3C/E4 analysis QA were inspected. This is a *source-to-artifact inventory*, **not** a byte-level integrity audit of every checkpoint nor a proof of exact historical notebook executions.
