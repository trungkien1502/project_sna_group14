from __future__ import annotations
import hashlib, json, os, pickle, subprocess, tempfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    p = Path(path).resolve()
    cfg = json.loads(p.read_text(encoding='utf-8'))
    for key in ('project_root','run_root','rlmrec_repo','group_repo','raw_metadata','raw_reviews'):
        value = cfg['paths'].get(key)
        if not value or not Path(value).is_absolute():
            raise ValueError(f'paths.{key} must be an absolute filesystem path')
    if not cfg['run_id'] or not cfg['run_id'].replace('-','').replace('_','').isalnum():
        raise ValueError('run_id may only contain letters, digits, _ or -')
    return cfg


def run_dir(cfg):
    root = Path(cfg['paths']['run_root']).resolve() / cfg['run_id']
    old = Path(cfg['paths']['project_root']).resolve()
    if root == old or root.is_relative_to(old/'data') or root.is_relative_to(old/'experiments'):
        raise ValueError('run directory collides with original data/experiments; use experiments_llm_v21 or another separate branch')
    return root


def sha_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for part in iter(lambda: fh.read(chunk), b''): h.update(part)
    return h.hexdigest()


def sha_text(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()


def write_json(path, val):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=p.parent,delete=False,prefix='.tmp_') as f:
        json.dump(val,f,ensure_ascii=False,indent=2,sort_keys=True)
        f.write('\n'); name=f.name
    os.replace(name,p)


def read_jsonl(path):
    with open(path,encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try: yield json.loads(line)
                except Exception as exc: raise ValueError(f'Invalid JSONL {path}:{n}') from exc


def write_jsonl(path, records):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=p.parent,delete=False,prefix='.tmp_') as f:
        n=0
        for row in records:
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n'); n+=1
        name=f.name
    os.replace(name,p)
    return n


def utc(): return datetime.now(timezone.utc).isoformat()


def load_pickle(path):
    # ONLY load pickle data supplied by the trusted original RLMRec release/group.
    with open(path,'rb') as f:return pickle.load(f)


def check_git(repo, sha, allow_drift=False):
    if not (Path(repo)/'.git').exists(): raise FileNotFoundError(f'Missing git checkout: {repo}')
    actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if actual!=sha and not allow_drift:
        raise ValueError(f'Git commit mismatch: {repo}: expected {sha}, got {actual}')
    return actual


def assert_inputs(cfg):
    p=Path(cfg['paths']['project_root'])/'data/raw/amazon'
    names=('trn_mat.pkl','val_mat.pkl','tst_mat.pkl')
    for n in names:
        if not (p/n).is_file(): raise FileNotFoundError(f'Missing original split: {p/n}')
    return p


def ensure_run_spec(cfg):
    """Freeze all scientifically relevant settings for an immutable run_id.

    Quota pacing is intentionally NOT part of the lock: it may change between Colab sessions.
    """
    branch=run_dir(cfg)
    spec={
      'dataset':cfg['data'],
      'sources':cfg['source'],
      'raw_metadata_path':cfg['paths']['raw_metadata'],
      'raw_reviews_path':cfg['paths']['raw_reviews'],
      'project_root':cfg['paths']['project_root'],
      'rlmrec_repo':cfg['paths']['rlmrec_repo'],
      'group_repo':cfg['paths']['group_repo'],
      'profiles':{k:v for k,v in cfg['profiles'].items() if k not in
                   ('max_daily_requests','min_interval_seconds','max_retries')},
      'embedding':cfg['embedding']
    }
    path=branch/'audit/run_spec.json'
    if path.is_file():
        existing=json.loads(path.read_text())
        if existing!=spec:
            raise ValueError('Run config drift detected. Use a NEW run_id for changed prompts/model/sampling/embedding/raw source.')
    else:write_json(path,spec)
    return spec
