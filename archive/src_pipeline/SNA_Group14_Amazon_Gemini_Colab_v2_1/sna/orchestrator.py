"""A resumable, fail-closed wrapper around the user's existing E1--E4 notebooks.

No trained model is fabricated: all actual training/evaluation calls go to user source.
"""
from pathlib import Path
import json
from .common import run_dir, write_json, utc
from .experiments import run_stage, preflight


def steps():
    result=[]
    for model in ('lightgcn','lightgcn_plus'):
        for seed in (2023,2024,2025):result.append(dict(stage='e1-train',model=model,seed=seed))
    for model in ('lightgcn','lightgcn_plus'):
        for seed in (2023,2024,2025):result.append(dict(stage='e1-export',model=model,seed=seed))
    for stage in ('e1-popular','e1-summary','e2'):result.append(dict(stage=stage))
    for mode in ('user_only','item_only'):
        for seed in (2023,2024,2025):result.append(dict(stage='e3a',mode=mode,seed=seed))
    for stage in ('e3b','e3c-train','e3c-export','e3c-analysis','e3d'):result.append(dict(stage=stage))
    for model in ('LightGCN','RLMRec-Con'):
        for layer in (1,2,3):result.append(dict(stage='e4-train',model=model,layer=layer))
    for stage in ('e4-export','e4-analysis'):result.append(dict(stage=stage))
    return result


def has_outputs(cfg,s):
    b=run_dir(cfg);stage=s['stage'];seed=s.get('seed');model=s.get('model');layer=s.get('layer');mode=s.get('mode')
    def j(path):
        p=b/path
        if not p.is_file():return {}
        try:return json.loads(p.read_text())
        except Exception:return {}
    def exists(path):return (b/path).is_file()
    if stage=='e1-train':
        root=f'experiments/E1/{model}/seed_{seed}'
        return j(root+'/metrics.json').get('status')=='completed' and exists(root+f'/checkpoints/{model}-amazon-{seed}.pth')
    if stage=='e1-export':
        root=f'experiments/predictions/{model}/seed_{seed}'
        return exists(root+'/topk_test.npz') and j(root+'/export_verification.json').get('status')=='passed'
    if stage=='e1-popular':return exists('experiments/E1/most_popular/metrics.json') and exists('experiments/predictions/most_popular/topk_test.npz')
    if stage=='e1-summary':return exists('experiments/E1/summary/e1_baseline_summary.json')
    if stage=='e2':return exists('experiments/E2/popularity_bias/E2_model_summary.csv')
    if stage=='e3a':
        root=f'experiments/E3/runs/{mode}/weight_e1_default/seed_{seed}'
        return j(root+'/e3a_verification.json').get('status')=='passed' and exists(root+'/topk/topk_test.npz')
    if stage=='e3b':return exists('experiments/E3B/popularity_bias/E3B_model_summary.csv')
    if stage=='e3c-train':
        return all(j(f'experiments/E3C/runs/full/weight_{w}/seed_{s}/train_status.json').get('status') in ('trained_pending_export','passed','completed')
                   and exists(f'experiments/E3C/runs/full/weight_{w}/seed_{s}/checkpoints/lightgcn_plus-amazon-{s}.pth')
                   for w in ('0','0.005','0.02','0.05') for s in (2023,2024,2025))
    if stage=='e3c-export':
        return all(exists(f'experiments/E3C/runs/full/weight_{w}/seed_{s}/topk/topk_test.npz')
                   for w in ('0','0.005','0.01','0.02','0.05') for s in (2023,2024,2025))
    if stage=='e3c-analysis':
        return any(p.name=='E3C_analysis_protocol_and_QA.json' and j(str(p.relative_to(b))).get('status')=='ANALYSIS_QA_PASS'
                   for p in (b/'experiments/E3C/analysis').rglob('E3C_analysis_protocol_and_QA.json'))
    if stage=='e3d':return exists('experiments/E3D/agreement_analysis/E3D_method.json') and exists('experiments/E3D/agreement_analysis/E3D_group_summary.csv')
    if stage=='e4-train':
        k='lightgcn' if model=='LightGCN' else 'lightgcn_plus'
        return all(j(f'experiments/E4/recovery_runs/{k}/layers_{layer}/seed_{s}/train_status.json').get('status')=='trained_pending_export'
                   and exists(f'experiments/E4/recovery_runs/{k}/layers_{layer}/seed_{s}/checkpoint_verification.json')
                   and exists(f'experiments/E4/recovery_runs/{k}/layers_{layer}/seed_{s}/checkpoints/{k}-amazon-{s}.pth')
                   for s in (2023,2024,2025))
    if stage=='e4-export':
        return all(exists(f'experiments/E4/exports/{m}/layers_{l}/seed_{s}/topk/topk_test.npz')
                   and j(f'experiments/E4/exports/{m}/layers_{l}/seed_{s}/topk/E4_export_provenance.json')
                   for m in ('lightgcn','lightgcn_plus') for l in (1,2,3) for s in (2023,2024,2025))
    if stage=='e4-analysis':
        return any(j(str(p.relative_to(b))).get('status')=='QA_PASS_18_OF_18'
                   for p in (b/'experiments/E4/analysis').rglob('E4_analysis_manifest.json'))
    raise ValueError(stage)


def run_all(cfg,execute=False):
    all_steps=steps();status=[{'step':s,'output_ready':has_outputs(cfg,s)} for s in all_steps]
    if not execute:
        return {'status':'PLAN_ONLY_NO_GPU_CALLS','n_steps':len(all_steps),'pending':sum(not x['output_ready'] for x in status),'steps':status}
    raise RuntimeError('Colab v2.1 disables run-all --execute: prepare and run each notebook natively in Colab, stage-by-stage. No unverified background kernel.')
