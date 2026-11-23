from __future__ import annotations
import json, os, re, shutil, traceback
from pathlib import Path
import numpy as np
from .common import check_git, run_dir, write_json, sha_file, sha_text, utc
from .data import assert_same_splits

# Exact filenames found in user's GitHub repository main (2026-11-21).
STAGES = {
  'e1-train': 'notebooks/03_E1_Full_Training.ipynb',
  'e1-export': 'notebooks/04_E1_Export_TopK.ipynb',
  'e1-popular': 'notebooks/05_E1_Most_Popular.ipynb',
  'e1-summary': 'notebooks/06_E1_Summary.ipynb',
  'e2': 'notebooks/07_E2_Popularity_Bias.ipynb',
  'e3a': 'notebooks/08_E3A_Semantic_Alignment_Ablation_FIXED.ipynb',
  'e3b': 'notebooks/10_E3B_Popularity_Bias_Analysis.ipynb',
  'e3c-train': 'notebooks/11_E3C_Semantic_Weight_Sensitivity_FIXED_v3.ipynb',
  'e3c-export': 'notebooks/E3C_Export_TopK_Fixed.ipynb',
  'e3c-analysis': 'notebooks/12_E3C_Analysis_Weight_Sensitivity.ipynb',
  'e3d': 'notebooks/E3D_Semantic_Collaborative_Agreement_Drive.ipynb',
  'e4-train': 'notebooks/E4_RQ3_Training_FIXED_Drive_Checkpoints.ipynb',
  'e4-export': 'notebooks/E4_RQ3_Export_TopK_FIXED.ipynb',
  'e4-analysis': 'notebooks/E4_RQ3_Analysis.ipynb',
}

OLD_PATH='/content/drive/MyDrive/Data_SNA_Group14'
OLD_GROUP='/content/project_sna_group14'


def native_colab_bootstrap(cfg, stage):
    """Insert into each staged notebook; executes inside Colab's native Python kernel.

    It deliberately does NOT silently delete/reset sources, overwrite the old Drive
    result tree, or launch a second Jupyter kernel.
    """
    branch=str(run_dir(cfg))
    pap_sha=cfg['source']['rlmrec_commit'];grp_sha=cfg['source']['group_commit']
    root_orig=cfg['paths']['project_root']
    return f"""# COLAB v2.1: native-runtime preflight, NOT nbclient.
from google.colab import drive
from pathlib import Path
import subprocess, sys, json, os

drive.mount('/content/drive')
BRANCH = Path({branch!r})
ORIGINAL_ROOT = Path({root_orig!r})
assert BRANCH.resolve() != ORIGINAL_ROOT.resolve()
assert (BRANCH/'data/package_manifest.json').is_file(), 'Package the Gemini data FIRST in START_HERE_Colab.ipynb'
PKG = json.loads((BRANCH/'data/package_manifest.json').read_text(encoding='utf-8'))
assert PKG.get('status')=='PASS', 'MOCK/partial package is not accepted for research training'
for fname in ('trn_mat.pkl','val_mat.pkl','tst_mat.pkl','usr_prf.pkl','itm_prf.pkl','usr_emb_np.pkl','itm_emb_np.pkl'):
    assert (BRANCH/'data/raw/amazon'/fname).is_file(), 'Missing new run data: '+fname

def _checkout(url, dest, commit):
    dest=Path(dest)
    if not dest.exists():
        subprocess.run(['git','clone',url,str(dest)],check=True)
        subprocess.run(['git','-C',str(dest),'checkout','--detach',commit],check=True)
    assert (dest/'.git').exists(), f'Not a git checkout: {{dest}}'
    head=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
    assert head==commit, f'Wrong pinned checkout {{dest}}: {{head}} != {{commit}}. New clean runtime required.'
_checkout('https://github.com/HKUDS/RLMRec.git','/content/RLMRec',{pap_sha!r})
_checkout('https://github.com/trungkien1502/project_sna_group14.git','/content/project_sna_group14',{grp_sha!r})
# Never auto-reset an existing data link: a wrong link must block the run.
for repo in ('/content/RLMRec',):
    link=Path(repo)/'data/amazon'
    if link.exists() or link.is_symlink():
        assert link.resolve()==(BRANCH/'data/raw/amazon').resolve(), 'Old dataset link detected; restart runtime before this stage'
os.chdir('/content')
print('NATIVE_COLAB_PREFLIGHT PASS | stage={stage} | new branch:',BRANCH)
print('Next: Runtime > Run all; preserve all Gemini outputs on Drive, never run original notebooks separately against old Drive root.')"""


def preflight(cfg, check_source=True):
    s=assert_same_splits(cfg)
    branch=run_dir(cfg)
    manifest=branch/'data/package_manifest.json'
    if not manifest.is_file():raise FileNotFoundError('First finish item/user profiles, embeddings and `package`')
    p=json.loads(manifest.read_text())
    if p['status']!='PASS':raise ValueError('Test-only package is not allowed for real training')
    for name,sha in p['packed_files_sha256'].items():
        path=branch/'data/raw/amazon'/name
        if not path.is_file() or sha_file(path)!=sha:
            raise ValueError(f'Packed dataset changed: {path}')
    orig=Path(cfg['paths']['project_root'])/'data/raw/amazon'
    for name in ('trn_mat.pkl','val_mat.pkl','tst_mat.pkl'):
        if sha_file(orig/name)!=sha_file(branch/'data/raw/amazon'/name):
            raise ValueError(f'Original split mismatch: {name}')
    # RLMRec builds an implicit data symlink under its checkout; do not use old profile/embeddings.
    report={'status':'PASS','branch':str(branch),'model':p['profile_model'],
            'embedding_model':p['embedding_model'],'dataset_manifest':str(manifest),
            'source_hashes':{},'time_utc':utc(),'source_tree_checked':check_source}
    if check_source:
        for key,loc in [('rlmrec','rlmrec_repo'),('group','group_repo')]:
            report['source_hashes'][key]=check_git(cfg['paths'][loc],cfg['source'][key+'_commit'],
                       cfg['execution'].get('allow_group_commit_drift',False) if key=='group' else False)
        missing=[p for p in STAGES.values() if not (Path(cfg['paths']['group_repo'])/p).is_file()]
        if missing:raise FileNotFoundError(f'Missing user repo notebooks: {missing}')
        export=Path(cfg['paths']['group_repo'])/'src/inference/export_topk.py'
        if not export.is_file():raise FileNotFoundError(export)
    write_json(branch/'audit'/'experiment_preflight.json',report)
    return report


def branch_baseline_recall(cfg):
    b=run_dir(cfg)
    vals=[]
    for seed in (2023,2024,2025):
        p=b/f'experiments/E1/lightgcn_plus/seed_{seed}/metrics.json'
        if not p.is_file():raise FileNotFoundError(f'New E1 Gemini baseline is missing: {p}')
        m=json.loads(p.read_text())
        if m.get('status')!='completed':raise ValueError(f'Incomplete Gemini E1 run: {p}')
        vals.append(float(m['test_metrics']['recall@20']))
    return float(np.mean(vals))


def replace_assigned(src,key,value,stage):
    """Modify only an explicit top-level config assignment, never experimental math."""
    q=json.dumps(value) if isinstance(value,str) else str(value)
    pat=rf'(?m)^([ \t]*{re.escape(key)}[ \t]*=[ \t]*)([^\n#]*?)([ \t]*(?:\#.*)?)$'
    changed,n=re.subn(pat,lambda m:m.group(1)+q+m.group(3),src,count=1)
    if n!=1:raise ValueError(f'{stage}: expected a single setting {key}=...')
    return changed


def prepare_group_runtime(cfg):
    """Clone working copy of group source into isolated run and patch only exporter ROOT examples.

    Original GitHub checkout and user's prior results are never modified.
    """
    origin=Path(cfg['paths']['group_repo'])
    dest=run_dir(cfg)/'runtime/group_repo'
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(origin,dest,ignore=shutil.ignore_patterns('__pycache__','.ipynb_checkpoints'))
    exporter=dest/'src/inference/export_topk.py'
    if not exporter.is_file():raise FileNotFoundError(exporter)
    source=exporter.read_text(encoding='utf-8')
    if OLD_PATH in source:
        source=source.replace(OLD_PATH,str(run_dir(cfg)))
        exporter.write_text(source,encoding='utf-8')
    if OLD_PATH in exporter.read_text(encoding='utf-8'):
        raise ValueError('Old Drive root remains in staged group exporter')
    return dest


def stage_source(cfg,stage,model=None,seed=None,mode=None,layer=None,execute=False):
    if stage not in STAGES:raise ValueError(f'Unknown stage {stage}; valid {list(STAGES)}')
    branch=run_dir(cfg);group=Path(cfg['paths']['group_repo'])
    src=group/STAGES[stage]
    if not src.is_file():raise FileNotFoundError(f'Missing group notebook {src}')
    import nbformat
    nb=nbformat.read(src,as_version=4)
    applied=[]
    group_runtime=prepare_group_runtime(cfg) if stage in ('e1-export','e3c-export','e4-export') else None
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        t=cell.source
        if OLD_PATH in t:
            t=t.replace(OLD_PATH,str(branch));applied.append('project_ROOT_rewritten')
        if group_runtime is not None and OLD_GROUP in t:
            t=t.replace(OLD_GROUP,str(group_runtime));applied.append('isolated_group_exporter')
        if stage=='e3a' and '/content/RLMRec_E3' in t:
            t=t.replace('/content/RLMRec_E3',f'/content/RLMRec_E3_{mode}_{seed}')
            applied.append('unique_clean_paper_clone_for_e3a_run')
        if stage=='e4-train' and '/content/RLMRec_E4_source' in t:
            key='lightgcn' if model=='LightGCN' else 'lightgcn_plus'
            t=t.replace('/content/RLMRec_E4_source',f'/content/RLMRec_E4_source_{key}_L{layer}')
            applied.append('unique_clean_paper_clone_for_e4_model_layer')

        # Never change the pinned RLMRec clone; source code itself remains versioned.
        if stage in ('e1-train','e1-export'):
            if model is None or seed is None:raise ValueError(f'{stage} needs --model lightgcn|lightgcn_plus and --seed')
            if model not in ('lightgcn','lightgcn_plus') or seed not in (2023,2024,2025):raise ValueError('Invalid E1 model/seed')
            if re.search(r'(?m)^\s*MODEL\s*=',t):t=replace_assigned(t,'MODEL',model,stage);applied.append('model')
            if re.search(r'(?m)^\s*SEED\s*=',t):t=replace_assigned(t,'SEED',seed,stage);applied.append('seed')
        if stage=='e3a':
            if mode not in ('user_only','item_only'):raise ValueError('E3A canonical uses user_only/item_only; full=E1, graph_only=E1 LightGCN')
            if seed not in (2023,2024,2025):raise ValueError('E3A needs --seed 2023|2024|2025')
            if re.search(r'(?m)^\s*ALIGN_MODE\s*=',t):t=replace_assigned(t,'ALIGN_MODE',mode,stage);applied.append('mode')
            if re.search(r'(?m)^\s*SEED\s*=',t):t=replace_assigned(t,'SEED',seed,stage);applied.append('seed')
        if stage=='e4-train':
            if model not in ('LightGCN','RLMRec-Con') or layer not in (1,2,3):
                raise ValueError('E4 needs --model LightGCN|RLMRec-Con and --layer 1|2|3')
            if re.search(r'(?m)^\s*MODEL\s*=',t):t=replace_assigned(t,'MODEL',model,stage);applied.append('model')
            if re.search(r'(?m)^\s*LAYER\s*=',t):t=replace_assigned(t,'LAYER',layer,stage);applied.append('layer')
        if stage in ('e4-train','e3c-train') and execute:
            if re.search(r'(?m)^\s*DRY_RUN\s*=',t):t=replace_assigned(t,'DRY_RUN',False,stage);applied.append('DRY_RUN=False')
            if re.search(r'(?m)^\s*SMOKE_TEST_ONLY\s*=',t):t=replace_assigned(t,'SMOKE_TEST_ONLY',False,stage);applied.append('SMOKE_TEST_ONLY=False')
        if stage in ('e4-export','e3c-export') and execute:
            if re.search(r'(?m)^\s*MAX_NEW_EXPORTS\s*=',t):t=replace_assigned(t,'MAX_NEW_EXPORTS',None,stage);applied.append('MAX_NEW_EXPORTS=None')
        if stage=='e3c-analysis' and re.search(r'(?m)^\s*BASELINE_RECALL20_EXPECTED\s*=',t):
            v=branch_baseline_recall(cfg)
            t=replace_assigned(t,'BASELINE_RECALL20_EXPECTED',v,stage)
            applied.append(f'new_E1_baseline={v:.9f}')
        cell.source=t
        cell.execution_count=None
        cell.outputs=[]
    # Colab must execute the cloned notebooks in its own kernel; add safe prep cell.
    nb.cells.insert(0, nbformat.v4.new_code_cell(native_colab_bootstrap(cfg,stage)))
    applied.append('native_colab_bootstrap')
    if any(OLD_PATH in c.source for c in nb.cells[1:] if c.cell_type=='code'):
        raise ValueError(f'Unpatched old Drive path remains in staged {stage}; fail closed')
    if 'project_ROOT_rewritten' not in applied:
        raise ValueError(f'No original Drive ROOT found in {src}. Inspect notebook before running.')
    if stage in ('e1-train','e1-export') and not {'model','seed'}.issubset(set(applied)):
        raise ValueError('Could not patch E1 model and seed')
    if stage=='e3a' and not {'mode','seed'}.issubset(set(applied)):
        raise ValueError('Could not patch E3A mode and seed')
    if stage=='e4-train' and not {'model','layer'}.issubset(set(applied)):
        raise ValueError('Could not patch E4 model and layer')
    # Author's original notebook is never modified; log both SHA and the modifications.
    name=stage+('-'+str(model) if model else '')+('-'+str(mode) if mode else '')+('-'+str(layer) if layer else '')+('-'+str(seed) if seed else '')
    out=branch/'runtime/notebooks'/f'{name}.ipynb';out.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(nb,out)
    manifest={'status':'STAGED','stage':stage,'source':str(src),'source_sha256':sha_file(src),
              'staged':str(out),'staged_sha256':sha_file(out),'changes':sorted(set(applied)),
              'mode':'NATIVE_COLAB_STAGED_UNEXECUTED',
              'branch_root':str(branch),'time_utc':utc(),
              'warning':'Run this .ipynb natively in Colab via Runtime > Run all; Colab GPU execution not verified in package build'}
    write_json(branch/'runtime/stages'/f'{name}.json',manifest)
    return out,manifest


def run_stage(cfg,stage,model=None,seed=None,mode=None,layer=None,execute=False):
    if not execute:
        # Prepare native Colab notebook only after the sealed dataset is verified.
        preflight(cfg)
        out,m=stage_source(cfg,stage,model,seed,mode,layer,False)
        return {'status':'STAGED_NOT_EXECUTED','notebook':str(out),'manifest':m}
    raise RuntimeError('Native Colab execution only. Prepare a stage without --execute, then open the resulting .ipynb and select Runtime > Run all. No nested Jupyter kernel is launched.')


def stage_plan(cfg):
    branch=run_dir(cfg)
    group=Path(cfg['paths']['group_repo'])
    return {'branch_root':str(branch),'paper_commit':cfg['source']['rlmrec_commit'],
            'group_commit':cfg['source']['group_commit'],
            'stages':[{'name':key,'source':path,'exists':(group/path).is_file()} for key,path in STAGES.items()],
            'execution_order':[
              'e1-train: lightgcn and lightgcn_plus x seeds 2023/24/25',
              'e1-export: same six checkpoints','e1-popular once','e1-summary once',
              'e2 once','e3a: user_only/item_only x seeds 2023/24/25','e3b once',
              'e3c-train: full weight 0/.005/.02/.05 x three seeds (0.01 reuse new E1)',
              'e3c-export once','e3c-analysis once',
              'e3d once (uses NEW semantic embeddings)',
              'e4-train: 2 models x 3 layers x three seeds, six notebook configurations',
              'e4-export once','e4-analysis once'],
            'notes':['No stage reuses GPT-3.5-era Top-K or checkpoints',
                     'Run IDs separate original results from Gemini branch',
                     'RLMRec weights/checkpoints cannot be verified without actual GPU execution']}
