#!/usr/bin/env python3
"""SNA Group14 safe data split + evaluation scaffold. No model training, no legacy pickle loading.
CSV inputs only; never run arbitrary notebook cells or load untrusted pickle/checkpoints.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_INTERACTIONS = ("user_id", "item_id", "timestamp")


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def persist_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def load_interactions(path):
    data = pd.read_csv(path, dtype={"user_id": str, "item_id": str})
    missing = set(REQUIRED_INTERACTIONS) - set(data.columns)
    if missing:
        raise ValueError(f'Missing interaction columns: {sorted(missing)}')
    data = data.copy()
    if data[list(REQUIRED_INTERACTIONS)].isna().any().any():
        raise ValueError('Null user_id/item_id/timestamp is prohibited')
    data['timestamp_parsed'] = pd.to_datetime(data.timestamp, utc=True, errors='coerce')
    if data.timestamp_parsed.isna().any():
        raise ValueError('Every timestamp must parse to a real date')
    if data.duplicated(['user_id','item_id']).any():
        raise ValueError('Duplicate user-item interactions: define an explicit deduplication policy')
    return data


def split(args):
    src = Path(args.interactions)
    raw = load_interactions(src)
    sizes = raw.groupby('user_id').size()
    eligible_users = sizes[sizes >= args.min_history].index
    if len(eligible_users) == 0:
        raise ValueError('No user meets min_history; cannot create temporal holdout')
    df = raw[raw.user_id.isin(eligible_users)].sort_values(['user_id','timestamp_parsed','item_id'], kind='stable').copy()
    df['position'] = df.groupby('user_id').cumcount()
    df['history_count'] = df.groupby('user_id')['item_id'].transform('size')
    df['split'] = np.where(df.position == df.history_count - 1, 'test',
                    np.where(df.position == df.history_count - 2, 'validation', 'train'))
    # A user must retain a nonempty training history.
    assert (df.groupby('user_id').apply(lambda g: (g.split == 'train').sum(), include_groups=False) >= 1).all()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Refuse to overwrite nonempty split folder: {out}')
    out.mkdir(parents=True, exist_ok=True)
    cols=['user_id','item_id','timestamp']
    for name in ('train','validation','test'):
        df.loc[df.split==name,cols].to_csv(out/f'{name}.csv', index=False)
    manifest={'schema':'sna-split-v1','policy':'user_temporal_last2','source_sha256':digest(src),
              'source_path':str(src.resolve()),'min_history':args.min_history,
              'tie_breaker':'item_id','included_users':len(eligible_users),
              'excluded_users':int(len(sizes)-len(eligible_users)),
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'counts':{name:int((df.split==name).sum()) for name in ('train','validation','test')},
              'checksums':{name:digest(out/f'{name}.csv') for name in ('train','validation','test')}}
    persist_json(out/'split_manifest.json',manifest)
    print(json.dumps(manifest,indent=2))


def load_splits(folder, verify_manifest=True):
    folder=Path(folder)
    manifest_path=folder/'split_manifest.json'
    if not manifest_path.exists():
        raise ValueError('Missing locked split_manifest.json')
    m=json.loads(manifest_path.read_text(encoding='utf-8'))
    data={}
    for s in ('train','validation','test'):
        path=folder/f'{s}.csv'
        if verify_manifest and digest(path) != m['checksums'][s]:
            raise ValueError(f'CHANGED SPLIT: {path}')
        df=pd.read_csv(path,dtype={'user_id':str,'item_id':str})
        if df[['user_id','item_id']].isna().any().any():
            raise ValueError('Null interaction IDs')
        if df.duplicated(['user_id','item_id']).any():
            raise ValueError(f'Duplicates in {s}')
        data[s]=df
    seen={k:set(map(tuple,v[['user_id','item_id']].itertuples(index=False,name=None))) for k,v in data.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if seen[a] & seen[b]:
            raise ValueError(f'LEAKAGE: {a}/{b} interaction pair overlap')
    for earlier,later in [('train','validation'),('train','test'),('validation','test')]:
        e=data[earlier].copy();l=data[later].copy()
        e['t']=pd.to_datetime(e['timestamp'],utc=True);l['t']=pd.to_datetime(l['timestamp'],utc=True)
        max_earlier=e.groupby('user_id')['t'].max()
        min_later=l.groupby('user_id')['t'].min()
        common=max_earlier.index.intersection(min_later.index)
        if (max_earlier.loc[common] > min_later.loc[common]).any():
            raise ValueError(f'TEMPORAL LEAKAGE: {earlier} after {later}')
    return data,m


def audit(args):
    data,m=load_splits(args.splits)
    report={'status':'PASS_SPLIT_ONLY','counts':{k:len(v) for k,v in data.items()},'split_sha256':m['checksums'],
            'semantic_provenance':'NOT_VERIFIED','ranking_provenance':'NOT_VERIFIED'}
    if args.semantic_provenance:
        p=Path(args.semantic_provenance)
        prof=pd.read_csv(p,dtype=str).fillna('')
        required={'user_id','item_id','source_split'}
        if not required.issubset(prof.columns):
            raise ValueError('semantic provenance CSV must contain user_id,item_id,source_split; one row per evidence interaction')
        train_pairs=set(map(tuple,data['train'][['user_id','item_id']].itertuples(index=False,name=None)))
        evidence=set(map(tuple,prof[['user_id','item_id']].itertuples(index=False,name=None)))
        invalid=prof.source_split.ne('train').any() or not evidence.issubset(train_pairs)
        if invalid:
            raise ValueError('SEMANTIC LEAKAGE: user profile evidence must be a subset of train interactions')
        report['semantic_provenance']='PASS_DECLARED_EVIDENCE_ONLY'
        report['semantic_evidence_rows']=len(prof)
        report['semantic_evidence_sha256']=digest(p)
        report['warning']='Declared evidence verified; textual paraphrase/LLM memorization and unlogged evidence not proven absent.'
    if args.out: persist_json(args.out,report)
    print(json.dumps(report,indent=2))


def evaluate(args):
    data,m=load_splits(args.splits)
    path=Path(args.rankings)
    top=pd.read_csv(path,dtype={'user_id':str,'item_id':str})
    required={'user_id','item_id','rank','score'}
    if not required.issubset(top.columns):
        raise ValueError(f'Ranking CSV missing {sorted(required-set(top.columns))}')
    if top[list(required)].isna().any().any():
        raise ValueError('Rankings contain null fields')
    if len(top)==0: raise ValueError('Empty ranking file')
    top['rank']=pd.to_numeric(top['rank'],errors='raise').astype(int)
    top['score']=pd.to_numeric(top['score'],errors='raise')
    if not np.isfinite(top.score).all(): raise ValueError('Nonfinite model scores')
    if (top['rank']<1).any() or top.duplicated(['user_id','rank']).any() or top.duplicated(['user_id','item_id']).any():
        raise ValueError('Duplicate recommendations or invalid rank')
    top=top[top['rank']<=args.k].copy()
    counts=top.groupby('user_id')['rank'].agg(['size','min','max'])
    if (counts['size']!=args.k).any() or (counts['min']!=1).any() or (counts['max']!=args.k).any():
        raise ValueError('Every evaluated user must have exactly ranks 1..K')
    test_by_user=data['test'].groupby('user_id')['item_id'].apply(set).to_dict()
    if set(top.user_id)!=set(test_by_user):
        raise ValueError('Ranking user IDs must exactly equal test-eligible user IDs')
    prohibited = set(map(tuple,data['train'][['user_id','item_id']].itertuples(index=False,name=None)))
    if args.mask_validation:
        prohibited |= set(map(tuple,data['validation'][['user_id','item_id']].itertuples(index=False,name=None)))
    recommended = set(map(tuple,top[['user_id','item_id']].itertuples(index=False,name=None)))
    if recommended & prohibited:
        raise ValueError('MASK VIOLATION: recommended item was seen in train/validation')
    train_pop=data['train'].groupby('item_id').size().sort_values(ascending=False)
    # Define item universe from TRAIN only: avoid deriving head/tail threshold from held-out labels.
    # In production, replace with a separately versioned item catalog available before prediction.
    all_items=set(data['train'].item_id)
    num_head=int(np.ceil(len(all_items)*0.2))
    head=set(sorted(all_items,key=lambda it:(-int(train_pop.get(it,0)),it))[:num_head])
    test_pos=data['test'].groupby('user_id')['item_id'].apply(set).to_dict()
    records=[]
    all_recs=[]
    for user,g in top.sort_values(['user_id','rank']).groupby('user_id'):
        items=g.item_id.tolist();all_recs.extend(items)
        truth=test_pos[user]; hits=np.array([1 if it in truth else 0 for it in items],dtype=float)
        dcg=float((hits/np.log2(np.arange(2,len(items)+2))).sum())
        ideal=float((1/np.log2(np.arange(2,min(len(truth),args.k)+2))).sum())
        htruth=truth&head;ttruth=truth-head
        records.append({'user_id':user,'Recall':float(hits.sum()/len(truth)),'NDCG':dcg/ideal,
                        'ARP':float(np.mean([train_pop.get(it,0) for it in items])),
                        'APLT':float(np.mean([it not in head for it in items])),
                        'HeadRecall':sum(it in htruth for it in items)/len(htruth) if htruth else np.nan,
                        'TailRecall':sum(it in ttruth for it in items)/len(ttruth) if ttruth else np.nan})
    per_user=pd.DataFrame(records)
    results={k:(float(per_user[k].mean()) if pd.notna(per_user[k].mean()) else None)
             for k in ['Recall','NDCG','ARP','APLT','HeadRecall','TailRecall']}
    results.update({'K':int(args.k),'n_users':len(per_user),'Coverage':len(set(all_recs))/len(all_items),
                    'split_checksums':m['checksums'],'ranking_sha256':digest(path),
                    'mask_validation':bool(args.mask_validation),
                    'source_model_provenance': 'USER_MUST_SUPPLY_VERIFIABLE_MANIFEST',
                    'head_definition':'top ceil(20%) TRAIN-known items by train interaction count, tie-break item ID',
                    'head_tail_recall_denominator':'only users with relevant positive targets in that category',
                    'catalog_policy':'train_known_only; use a separately frozen availability-time item catalog for production'})
    if args.out:
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        per_user.to_csv(out/'per_user_metrics.csv',index=False)
        persist_json(out/'evaluation_summary.json',results)
    print(json.dumps(results,indent=2))


def main():
    p=argparse.ArgumentParser(description='Safe reusable recommendation evaluation scaffold')
    sp=p.add_subparsers(dest='command',required=True)
    s=sp.add_parser('split');s.add_argument('--interactions',required=True);s.add_argument('--out',required=True);s.add_argument('--min-history',type=int,default=5);s.set_defaults(func=split)
    a=sp.add_parser('audit');a.add_argument('--splits',required=True);a.add_argument('--semantic-provenance');a.add_argument('--out');a.set_defaults(func=audit)
    e=sp.add_parser('evaluate');e.add_argument('--splits',required=True);e.add_argument('--rankings',required=True);e.add_argument('--k',type=int,default=20);e.add_argument('--out');e.add_argument('--mask-validation',action=argparse.BooleanOptionalAction,default=True);e.set_defaults(func=evaluate)
    args=p.parse_args();args.func(args)

if __name__=='__main__': main()
