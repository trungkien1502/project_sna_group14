# SNA Group 14 — audit & reusable evaluation scaffold (2026-11-21)

**Status:** A carefully scoped **evaluation scaffold**, **NOT** a complete reimplementation of HKUDS/RLMRec training. It will not load old `.pkl` datasets, pretrained checkpoints, call an LLM, or execute all notebooks. All legacy experiment artefacts remain unchanged. The code has been tested locally on a synthetic dataset only. To reproduce E1–E4 on a real new dataset, the model training/export adapter, semantic-profile generator, and provenance recording must still be integrated and tested.

## 1. Source-of-truth audit

- Group GitHub: https://github.com/trungkien1502/project_sna_group14 (`main` tree SHA `c17caeb4448a8add7939510411339af918d258bb` at inspection).
- Actual trainer source referenced by E1/E3A/E3C/E4 notebooks: https://github.com/HKUDS/RLMRec , pinned Git commit `22413752246de3dee8ab0d509f7f7a8889080f95`. **Do not confuse this upstream commit with a commit of Group 14's repository.**
- Old dataset: Drive `Data_SNA_Group14/data/raw/amazon`, files `trn_mat.pkl`, `val_mat.pkl`, `tst_mat.pkl`, `usr_prf.pkl`, `itm_prf.pkl`, `usr_emb_np.pkl`, `itm_emb_np.pkl`.
- Old data inspection: 11,000 users, 9,332 items, train 120,464 interactions, val 40,290, test 40,106; no **interaction pair overlaps** found. This does **not** exclude leakage via profiles.
- Old profile audit `profile_leakage_audit.json`: 0 explicit ASIN matches; `provenance_verified = false`.
- Old title audit `title_leakage_audit.json`: 13 validation title matches and 16 test title matches; 29 suspected, `provenance_verified = false`. **Suspects, not proof of leakage.** Title matching alone misses paraphrases and can produce false positives.
- Old group `src/inference/export_topk.py`: masks **train only**, leaves validation unmasked by original E1 protocol, and exports test-user Top-K. A fresh strict pipeline can choose train + validation masking but results will not be directly protocol-identical with legacy E1–E4.
- E3C source-of-record output: `experiments/E3C/analysis/20261113T054415Z/` with `E3C_weight_summary.csv`, `E3C_topk_inventory_QA.csv`, `E3C_analysis_protocol_and_QA.json`; QA 15/15 exports; full-only kd_weight values `0, 0.005, 0.01, 0.02, 0.05`, three seeds. E3C protocol has `validation_masked = false`.
- E4 source-of-record output: `experiments/E4/analysis/20261118T022554Z/`, `E4_analysis_manifest.json`, `E4_topk_inventory_QA.csv`; QA 18/18 exports, three seeds × two models × three layers. The manifest also declares train-only masking and unmasked validation.

## 2. Notebook disposition: don't run everything

| Component | File(s) | Current classification | Verification still needed |
|---|---|---|---|
| E1 training | `notebooks/03_E1_Full_Training.ipynb` | Candidate canonical training workflow | Source commit + checkpoint hash + run metadata per seed |
| E1 Top-K | `notebooks/04_E1_Export_TopK.ipynb`, `src/inference/export_topk.py` | Candidate exporter + Python implementation | Hash of exporter used for each old prediction |
| E1 popular | `notebooks/05_E1_Most_Popular.ipynb` | Baseline candidate | Exact output/seed policy |
| E1 summary | `notebooks/06_E1_Summary.ipynb` | Analysis candidate | Pin archived input paths |
| E2 | `notebooks/07_E2_Popularity_Bias.ipynb` | Analysis candidate | Check metrics against existing E2 output |
| E3A | `notebooks/08_E3A_Semantic_Alignment_Ablation_FIXED.ipynb` | Candidate ablation training | Audit role-selective patch vs upstream commit and exports |
| Audit | `notebooks/09_Audit.ipynb` | Historical QA helper, not an experiment runner | Compare checks with new independent validators |
| E3B | `notebooks/10_E3B_Popularity_Bias_Analysis.ipynb` | Analysis candidate | Verify exact E3A/E1 export inventory |
| E3C training | `notebooks/11_E3C_Semantic_Weight_Sensitivity_FIXED_v3.ipynb` | Better candidate than original `11_E3C_Semantic_Weight_Sensitivity.ipynb`, but not yet proven exact executed code | Match run manifests & checkpoint hashes; original and v3 should **not** both run |
| E3C export | `notebooks/E3C_Export_TopK_Fixed.ipynb` | Candidate exporter | Check per-run export-provenance and E1 0.01 reuse |
| E3C analysis | `notebooks/12_E3C_Analysis_Weight_Sensitivity.ipynb` | Analysis used to generate matching-named QA reports | Verify source hashes; no training required |
| E3D | `notebooks/E3D_Semantic_Collaborative_Agreement_Drive.ipynb` | Candidate matching output paths | Compare with `E3D_method.json`; Colab variant not additive experiment |
| E4 training | `notebooks/E4_RQ3_Training_FIXED_Drive_Checkpoints.ipynb` | Candidate checkpoint-recovery trainer | Match individual Drive checkpoint verification; do not run `13_E4_Graph_Depth_RQ3.ipynb` as well |
| E4 export | `notebooks/E4_RQ3_Export_TopK_SAFE.ipynb` or `..._FIXED.ipynb` | **Unresolved which exact version produced outputs** | Need run logs/exporter hash from `E4_export_provenance.json` |
| E4 analysis | `notebooks/E4_RQ3_Analysis.ipynb` | Matches named E4 analysis outputs | Verify report and manifest hashes |

**Important:** `FIXED`, `SAFE`, and `v3` in a filename are human labels, not executable provenance. Those notebooks are *candidates*, not yet certified as the uniquely executed file. The existing QA CSVs validate exports but are not enough to reconstruct all historical execution environments.

## 3. Proposed locked pipeline for a NEW dataset

```text
raw events + item metadata (immutable version & SHA256)
  -> validate timestamps/IDs/deduplication policy
  -> user-temporal train/validation/test split (immutable SHA256)
  -> user profiles from TRAIN interactions only (record source interaction IDs)
  -> item profiles from metadata allowed at prediction time (record metadata snapshot)
  -> LLM/embedding generation (model revision, prompt, tokenizer, seed, params, hashes)
  -> train Most Popular, LightGCN, RLMRec-Con (same split, controlled seeds)
  -> checkpoint selection by VALIDATION Recall@20 only
  -> export Top-K for TEST, train+validation masking under NEW protocol
  -> independent evaluator + split/profile/export checks
  -> E1, E2, E3A, E3B, E3C, E3D, E4 summaries
  -> immutable run manifest, tables, figures and report-ready outputs
```

**No test labels are allowed upstream of final ranking evaluation.** Generate user profiles without train/validation/test leakage, and distinguish metadata that is genuinely available before a user's test event. Exclude target item identity and held-out interactions from LLM prompt construction. Do not use test Recall to select embedding model, LLM prompt or hyperparameters; tune using validation. If the old split/protocol changes, label old-vs-new metrics *not directly comparable*.

### Which model is new?

An instruction/chat **LLM** used to summarize user book preferences is not the same thing as an **embedding encoder** that produces the vector used for KD/alignment. Version both separately. If changing user text generation and embedding encoder simultaneously, improvement attribution is ambiguous. Use a 2x2 ablation where feasible: old/new profile generator × old/new embedding model, all on the **same new split**.

### Accuracy before optimization

1. New dataset + leakage-safe profiles + previous model architecture (control).
2. Same dataset/split + new semantic generator/encoder (test semantic contribution).
3. Use matched seeds, common evaluator, same candidate universe and K.
4. Only after passing QA, tune kd_weight, backbone depth and user/item alignment with **validation only**.
5. Run one final test evaluation on a frozen configuration.

## 4. Safe CLI available now

Requires Python 3.10+ and `pip install -r requirements.txt`.

**Input interactions.csv** columns: `user_id,item_id,timestamp` (ISO 8601 timestamps). Unique user–item pairs required; otherwise resolve repeat-interaction policy first.

```bash
python pipeline.py split --interactions inputs/interactions.csv --out runs/new_dataset/splits --min-history 5
python pipeline.py audit --splits runs/new_dataset/splits
```

To check declared *training-only evidence* for user profiles, provide `semantic_evidence.csv` columns `user_id,item_id,source_split` with one row per user-item interaction cited/used by a generated profile. This is a necessary but **not sufficient** profile leakage check.

```bash
python pipeline.py audit --splits runs/new_dataset/splits --semantic-provenance runs/new_dataset/semantic_evidence.csv
```

After a separately audited trainer/exporter creates `rankings.csv` with columns `user_id,item_id,rank,score`, containing ranks `1..K` for every test-eligible user, run:

```bash
python pipeline.py evaluate --splits runs/new_dataset/splits --rankings runs/new_dataset/rankings.csv --k 20 --out runs/new_dataset/eval
```

The evaluator uses TRAIN-known items to define the head/tail popularity threshold and coverage denominator; a production-grade, time-frozen item catalog is still needed. The evaluator enforces a **new strict policy**: no item in train or validation may be recommended. To reproduce the *old* mask rule for historical comparisons, pass `--no-mask-validation` explicitly; this is not the recommended fresh protocol. Run an identical evaluator with `--k 10` only if the exported ranking contains at least 10 candidates for each user.

**What this code currently validates:** immutable split hashes, pair-disjoint splits, per-user temporal ordering, semantic evidence references subset of training pairs, duplicate and rank checks, test-user coverage, train/validation masking, Recall, NDCG, ARP, APLT, head/tail recall and coverage.

**What it cannot yet validate or run:** the actual generation of user/item profiles; whether a generative LLM memorized/used future information; model training, checkpoint selection, inference/export, E3D agreement, E3C/E4 experiment matrices, performance or deterministic re-runs on real hardware. Connecting these pieces will require an audited upstream trainer and a frozen dataset/LLM version. No real new-data model performance was measured here.

## 5. Gates that must pass before real training

- Dataset `dataset_id`, immutable raw files and source licensing documented.
- Timestamps genuine; split temporal policy justified; no duplicate/overlap after split.
- Item descriptions available at the relevant time; user text sourced from train only.
- SHA256 every input, profile, embedding, model checkpoint, Top-K and metrics output.
- Record Git SHA of upstream RLMRec, group patch SHA, model/prompt/embedding revisions and key environment versions.
- Fix evaluation user universe, candidate item universe, train+val masking policy, K=10/20 and head/tail thresholds in one config. The starter evaluator defines the item universe from TRAIN-known items, never deriving head/tail thresholds from validation/test labels. A production catalog should instead be a separately frozen item-availability snapshot.
- Unit-test evaluator on synthetic data and verify baseline/Top-K recomputation.
- Never run old and new pipelines into the same Drive output folder; make a new immutable `dataset_version/run_id` namespace.

## Sources verified at audit time

- [GitHub Group14](https://github.com/trungkien1502/project_sna_group14)
- [Drive E3C QA & summaries](https://drive.google.com/drive/folders/1aa-3qE-LqLejL954XbD2_7hbPKxenEwd)
- [Drive E4 QA & summaries](https://drive.google.com/drive/folders/1YZokV3InEGVJ9o87cFVMhq5c1z-cmTl-)
- [Drive old data manifests](https://drive.google.com/drive/folders/1xmdApHPF_5It0jy5XEdIEIeUoDhK15ak)
- [Drive old raw dataset](https://drive.google.com/drive/folders/1p2ydDv39WCaDaW39itul7UrLUg73bHgO)
