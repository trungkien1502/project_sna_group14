from __future__ import annotations
import ast, gzip, html, json, re, shutil
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import sparse
from .common import assert_inputs, load_pickle, sha_file, sha_text, run_dir, read_jsonl, write_json, write_jsonl, utc, ensure_run_spec


def matrices(cfg):
    src=assert_inputs(cfg)
    vals=[]
    for n in ('trn_mat.pkl','val_mat.pkl','tst_mat.pkl'):
        mat=load_pickle(src/n)
        if not sparse.issparse(mat): raise ValueError(f'Expected sparse matrix: {n}')
        m=mat.tocsr();m.sum_duplicates()
        if m.data.size and ((m.data<0).any() or not np.isfinite(m.data).all()):
            raise ValueError(f'Invalid matrix entries in {n}')
        m=(m!=0).astype(np.int8).tocsr()
        m.eliminate_zeros()
        vals.append(m)
    return vals


def load_mappers(cfg, shape):
    source=Path(cfg['paths']['rlmrec_repo'])/'data/mapper'
    output=[]
    for kind,filename,num,ident,id_key in [
        ('item','amazon_item.json',shape[1],'asin','iid'),
        ('user','amazon_user.json',shape[0],'reviewerID','uid')]:
        p=source/filename
        if not p.is_file():raise FileNotFoundError(f'Missing RLMRec mapper {p}')
        d={};rev={}
        for row in read_jsonl(p):
            k=int(row[id_key]);v=str(row[ident])
            if k in d or v in rev:raise ValueError(f'Duplicate mapping in {p}: {k}/{v}')
            d[k]=v;rev[v]=k
        if set(d)!=set(range(num)):
            raise ValueError(f'{p}: mapping ID set does not equal 0..{num-1}')
        output.append((d,rev))
    return output


def audit(cfg, create=True):
    ensure_run_spec(cfg)
    src=assert_inputs(cfg)
    trn,val,tst=matrices(cfg)
    d=cfg['data']
    if len({m.shape for m in (trn,val,tst)})!=1:raise ValueError('Split matrix shapes differ')
    nu,ni=trn.shape
    if (nu,ni)!=(d['expected_users'],d['expected_items']):
        raise ValueError(f'Unexpected Amazon matrix shape {(nu,ni)}')
    counts=[int(m.nnz) for m in (trn,val,tst)]
    exp=[d['expected_train'],d['expected_validation'],d['expected_test']]
    if counts!=exp:raise ValueError(f'Interaction count mismatch: {counts} != {exp}')
    for a,b,name in ((trn,val,'train vs val'),(trn,tst,'train vs test'),(val,tst,'val vs test')):
        if a.multiply(b).nnz:raise ValueError(f'Overlapping user-item positive pairs: {name}')
    maps=load_mappers(cfg,(nu,ni))
    hashes={n:sha_file(src/n) for n in ('trn_mat.pkl','val_mat.pkl','tst_mat.pkl')}
    srcmap=Path(cfg['paths']['rlmrec_repo'])/'data/mapper'
    hashes['amazon_item.json']=sha_file(srcmap/'amazon_item.json')
    hashes['amazon_user.json']=sha_file(srcmap/'amazon_user.json')
    report={'status':'PASS','time_utc':utc(),'dataset':'amazon-book','shape':[nu,ni],
            'counts':dict(zip(['train','validation','test'],counts)),
            'pair_intersections':0,'source_hashes_sha256':hashes,
            'split_policy':'PRESERVE_AUTHOR_MATRICES_NEVER_RESPLIT',
            'raw_text_leakage_status':'NOT_DETERMINED_BY_MATRIX_AUDIT'}
    if create:write_json(run_dir(cfg)/'audit'/'split_audit.json',report)
    return report


def assert_same_splits(cfg):
    p=run_dir(cfg)/'audit'/'split_audit.json'
    if not p.exists():raise FileNotFoundError('Run audit first: split_audit.json not found')
    old=json.loads(p.read_text())
    new=audit(cfg,create=False)
    if old['source_hashes_sha256']!=new['source_hashes_sha256']:
        raise ValueError('Split or mapper changed since audit; refusing continuation')
    return new


def rows_source(path):
    p=Path(path)
    openf=gzip.open if p.name.endswith('.gz') else open
    with openf(p,'rt',encoding='utf-8',errors='replace') as f:
        for no,line in enumerate(f,1):
            if not line.strip():continue
            try: yield json.loads(line)
            except json.JSONDecodeError:
                try:yield ast.literal_eval(line)  # 2014 Amazon meta sometimes Python repr
                except Exception as e:raise ValueError(f'Cannot parse raw JSON at {p.name}:{no}') from e


def plain(s,max_chars):
    if isinstance(s,list):s=' '.join(str(x) for x in s)
    if s is None:s='None'
    s=html.unescape(re.sub(r'<[^>]*>', ' ',str(s)))
    s=re.sub(r'\s+',' ',s).strip()
    return (s[:max_chars] if s else 'None')


def index_raw(cfg):
    assert_same_splits(cfg)
    trn,val,tst=matrices(cfg)
    (item_id,item_back),(user_id,user_back)=load_mappers(cfg,trn.shape)
    p=run_dir(cfg)/'raw_index'; p.mkdir(parents=True,exist_ok=True)
    raw_meta=Path(cfg['paths']['raw_metadata']);raw_reviews=Path(cfg['paths']['raw_reviews'])
    if not raw_meta.is_file():raise FileNotFoundError(f'Raw Amazon metadata missing: {raw_meta}')
    use_reviews=cfg['profiles'].get('user_review_policy','none')=='train_pair'
    if use_reviews and not raw_reviews.is_file():
        raise FileNotFoundError(f'Raw reviews required for train_pair policy: {raw_reviews}')
    if cfg['profiles'].get('user_review_policy','none') not in ('none','train_pair'):
        raise ValueError('Invalid review policy')
    review_sha=sha_file(raw_reviews) if use_reviews else None
    previous=p/'index_manifest.json'
    if previous.is_file():
        old=json.loads(previous.read_text())
        expected=old.get('source_hashes_sha256',{})
        if expected.get('metadata')!=sha_file(raw_meta) or expected.get('reviews')!=review_sha:
            raise ValueError('Raw source changed in locked run; use a new run_id')
        if not (p/'books.jsonl').is_file() or not (p/'train_reviews.jsonl').is_file():
            raise ValueError('Raw index partial/missing; manually audit instead of silent overwrite')
        return old
    maxdesc=cfg['data']['max_description_chars']; maxreview=cfg['data']['max_review_chars']
    metadata={};metacnt=Counter()
    for row in rows_source(raw_meta):
        asin=str(row.get('asin',''))
        if asin not in item_back:continue
        iid=item_back[asin]
        title=plain(row.get('title','None'),500)
        desc=plain(row.get('description',row.get('feature',None)),maxdesc)
        # If multiple metadata rows map to same ASIN, deterministic first-match.
        if iid in metadata:
            metacnt['duplicate_asin']+=1;continue
        metadata[iid]={'iid':iid,'asin':asin,'title':title,'description':desc,'source':'raw_amazon_metadata'}
    coverage=len(metadata)/trn.shape[1]
    if coverage<cfg['data']['min_metadata_coverage']:
        raise ValueError(f'Low metadata coverage {coverage:.1%}; likely wrong Amazon raw vintage')
    for iid in range(trn.shape[1]):
        if iid not in metadata:
            metadata[iid]={'iid':iid,'asin':item_id[iid],'title':'None','description':'None','source':'metadata_not_found'}
    train_p={tuple(map(int,pair)) for pair in zip(*trn.nonzero())}
    # Ambiguous duplicate source reviews must NOT be selected as if they were one known train review.
    reviews={}; duplicate=set();cnt=Counter()
    for row in (rows_source(raw_reviews) if use_reviews else ()):
        uid=user_back.get(str(row.get('reviewerID','')))
        iid=item_back.get(str(row.get('asin','')))
        if uid is None or iid is None:continue
        k=(uid,iid)
        if k not in train_p:
            cnt['mapped_nontrain_rows_rejected']+=1;continue
        if k in duplicate:continue
        if k in reviews:
            reviews.pop(k,None);duplicate.add(k);cnt['ambiguous_train_pairs']+=1;continue
        review=plain(row.get('reviewText','None'),maxreview)
        key=sha_text(json.dumps([uid,iid,review,row.get('unixReviewTime')],ensure_ascii=False))
        reviews[k]={'uid':uid,'iid':iid,'review':review,'review_sha256':key,
                    'source':'raw_amazon_review_matched_to_train_pair'}
        cnt['train_pair_reviews_seen']+=1
    n1=write_jsonl(p/'books.jsonl',(metadata[i] for i in range(trn.shape[1])))
    n2=write_jsonl(p/'train_reviews.jsonl',(reviews[k] for k in sorted(reviews)))
    meta={'status':'PASS','source_metadata':str(raw_meta),'source_reviews':str(raw_reviews),
          'source_hashes_sha256':{'metadata':sha_file(raw_meta),'reviews':review_sha},
          'item_coverage':coverage,'metadata_rows':n1,'matched_unambiguous_train_reviews':n2,
          'books_with_missing_title':sum(v['title']=='None' for v in metadata.values()),
          'books_with_missing_description':sum(v['description']=='None' for v in metadata.values()),
          'train_positive_pairs':len(train_p),
          'train_pairs_with_unambiguous_raw_review_fraction':n2/max(1,len(train_p)),
          'mapped_nontrain_review_rows_rejected':cnt['mapped_nontrain_rows_rejected'],
          'ambiguous_train_pairs_with_reviews_omitted':len(duplicate),
          'review_gate':'TRAIN_POSITIVE_USER_ITEM_PAIR_ONLY' if use_reviews else 'NO_RAW_REVIEWS_USED',
          'limitation':'Matrices identify user-item pairs, not individual review events. train_pair mode cannot prove event-level leakage absence; none mode excludes review text entirely.',
          'items_metadata_missing':trn.shape[1]-int(round(coverage*trn.shape[1])),'time_utc':utc()}
    write_json(p/'index_manifest.json',meta)
    return meta
