from __future__ import annotations
import hashlib, json, os, re, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .common import ROOT, run_dir, read_jsonl, write_json, write_jsonl, sha_file, sha_text, utc
from .data import assert_same_splits, matrices


def prompt_system(kind):
    return (ROOT/'prompts'/f'amazon_{kind}.txt').read_text(encoding='utf-8')


def stable_sample(ids,k,uid,seed):
    ids=sorted(set(map(int,ids)))
    if k is None or k<=0 or len(ids)<=k:return ids
    def score(i):return hashlib.sha256(f'{seed}:{uid}:{i}'.encode()).hexdigest()
    return sorted(sorted(ids,key=score)[:k])


def build_prompts(cfg,kind):
    if kind not in ('item','user'):raise ValueError(kind)
    assert_same_splits(cfg)
    branch=run_dir(cfg);index=branch/'raw_index'
    if not (index/'index_manifest.json').is_file():raise FileNotFoundError('Run index-raw first')
    books={r['iid']:r for r in read_jsonl(index/'books.jsonl')}
    train,*_=matrices(cfg)
    out=branch/'prompts'/kind;out.mkdir(parents=True,exist_ok=True)
    system=prompt_system(kind)
    sourcehash=sha_text(system)
    if kind=='item':
        for iid in range(train.shape[1]):
            book=books[iid]
            payload={'title':book['title'], 'description':book['description']}
            user='BOOK INFORMATION:\n'+json.dumps(payload,ensure_ascii=False)
            prov={'iid':iid,'asin':book['asin'],'book_source':book['source'],
                  'item_prompt_data_fields':['title','description'],'train_review_ids':[],
                  'train_interaction_pairs':[], 'val_test_pairs':[],
                  'note':'Uses static metadata only; no review-derived item description fallback'}
            _write_prompt(out,iid,kind,system,user,sourcehash,prov)
        count=train.shape[1]
    else:
        ips=branch/'profiles'/'item';items={}
        for iid in range(train.shape[1]):
            p=ips/f'{iid}.json'
            if not p.is_file():raise FileNotFoundError(f'Item profile {iid} missing. Generate all item profiles before users.')
            row=json.loads(p.read_text())
            if row.get('status')!='ok':raise ValueError(f'Invalid item profile: {p}')
            expected_prompt=json.loads((branch/'prompts'/'item'/f'{iid}.json').read_text())['prompt_sha256']
            if row.get('prompt_sha256')!=expected_prompt:
                raise ValueError(f'Stale/mismatched item profile {iid}')
            items[iid]=row['summarization']
        reviews={(r['uid'],r['iid']):r for r in read_jsonl(index/'train_reviews.jsonl')}
        csr=train.tocsr()
        for uid in range(train.shape[0]):
            ids=csr.indices[csr.indptr[uid]:csr.indptr[uid+1]]
            keep=stable_sample(ids,cfg['profiles']['max_history_items'],uid,cfg['profiles']['history_seed'])
            rows=[];ids_used=[]
            for iid in keep:
                b=books[iid]
                policy=cfg['profiles'].get('user_review_policy','train_pair')
                if policy not in ('train_pair','none'):raise ValueError('Unknown user_review_policy')
                rv=reviews.get((uid,iid)) if policy=='train_pair' else None
                rows.append({'title':b['title'],'description':items[iid],
                             'review':rv['review'] if rv is not None else 'None'})
                if rv is not None:ids_used.append(rv['review_sha256'])
            user='PURCHASED ITEMS:\n'+json.dumps(rows,ensure_ascii=False)
            prov={'uid':uid,'history_total_train':len(ids),'history_selected':len(keep),
                  'history_selection':'deterministic_sha256_sample',
                  'history_seed':cfg['profiles']['history_seed'],
                  'history_cap':cfg['profiles']['max_history_items'],
                  'train_interaction_pairs':[[uid,int(i)] for i in keep],
                  'train_review_ids':ids_used,'val_test_pairs':[],
                  'referenced_item_profile_ids':keep,
                  'review_gate':'train_pair_only_and_ambiguous_duplicates_omitted' if cfg['profiles'].get('user_review_policy','train_pair')=='train_pair' else 'all_reviews_omitted'}
            if not len(keep):prov['warning']='No train interactions; empty purchased-items list'
            _write_prompt(out,uid,kind,system,user,sourcehash,prov)
        count=train.shape[0]
    manifest={'status':'PASS','kind':kind,'prompt_count':count,'system_prompt_sha256':sourcehash,
              'format':'Amazon-book source instruction + explicit reconstruction of input body',
              'note':'Paper publishes Amazon system instructions, not an exact dataset-wide payload construction script.',
              'source':'HKUDS/RLMRec generation/instruction/amazon_{item,user}.txt',
              'review_leakage_policy':'STATIC_ITEM_METADATA_AND_TRAIN_ONLY_USER_HISTORY',
              'time_utc':utc()}
    write_json(branch/'prompts'/f'{kind}_manifest.json',manifest)
    return manifest


def _write_prompt(folder,id_,kind,system,user,system_hash,prov):
    # No overwriting existing prompts with different content: freeze before sending paid requests.
    hash_=sha_text(system+'\nUSER\n'+user)
    doc={'kind':kind,'id':int(id_),'system':system,'user':user,'system_sha256':system_hash,
         'prompt_sha256':hash_,'provenance':prov}
    target=folder/f'{id_}.json'
    if target.is_file():
        old=json.loads(target.read_text(encoding='utf-8'))
        if old['prompt_sha256']!=hash_:
            raise ValueError(f'Prompt already locked and changed: {target}; use a fresh run_id')
    else:write_json(target,doc)


def parse_response(text_,kind):
    try:data=json.loads(text_)
    except Exception as exc:raise ValueError(f'LLM response must be strict JSON: {str(text_)[:200]!r}') from exc
    if not isinstance(data,dict) or not isinstance(data.get('summarization'),str) or not isinstance(data.get('reasoning'),str):
        raise ValueError('LLM output missing summarization or reasoning string')
    if not data['summarization'].strip():raise ValueError('Empty profile summary')
    limit=200 if kind=='item' else 100
    if len(data['summarization'].split())>limit:raise ValueError(f'{kind} summarization exceeds {limit} words')
    if kind=='item' and len(data['reasoning'].split())>200:raise ValueError('Item reasoning exceeds 200 words')
    return {'summarization':data['summarization'].strip(),'reasoning':data['reasoning'].strip()}


def mocked(prompt):
    if prompt['kind']=='item':
        return json.dumps({'summarization':'MOCK profile for book testing only','reasoning':'No real AI call'},ensure_ascii=False)
    return json.dumps({'summarization':'MOCK user preference testing only','reasoning':'No real AI call'},ensure_ascii=False)


def generate(cfg,kind,limit=None,dry_run=False):
    if kind not in ('item','user'):raise ValueError(kind)
    assert_same_splits(cfg)
    branch=run_dir(cfg);promptdir=branch/'prompts'/kind
    manifest=branch/'prompts'/f'{kind}_manifest.json'
    if not manifest.exists():raise FileNotFoundError(f'Build {kind} prompts first')
    expected=json.loads(manifest.read_text())['prompt_count']
    profiledir=branch/'profiles'/kind;profiledir.mkdir(parents=True,exist_ok=True)
    provider=cfg['profiles']['provider']; model=cfg['profiles']['model']
    if provider not in ('gemini','mock'):raise ValueError('Provider must be gemini or mock')
    client=None
    if provider=='gemini' and not dry_run:
        from google import genai
        from google.genai import types
        if not os.environ.get('GEMINI_API_KEY'):raise EnvironmentError('Set GEMINI_API_KEY via getpass; never paste keys into notebooks')
        if os.environ.get('SNA_ALLOW_REMOTE_LLM')!='YES':raise EnvironmentError('Set SNA_ALLOW_REMOTE_LLM=YES to authorize remote calls')
        client=genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    count=0;success=0;skipped=0;last_remote_call=None;attempts_path=branch/'logs'/'api_attempts.jsonl'
    attempts_path.parent.mkdir(parents=True,exist_ok=True)
    day_counts={}
    if attempts_path.is_file():
        for rec in read_jsonl(attempts_path):
            date=rec.get('utc_date')
            day_counts[date]=day_counts.get(date,0)+1
    for id_ in range(expected):
        pf=promptdir/f'{id_}.json'
        if not pf.is_file():raise FileNotFoundError(f'Missing prompt {pf}')
        doc=json.loads(pf.read_text(encoding='utf-8'))
        dest=profiledir/f'{id_}.json'
        if dest.exists():
            done=json.loads(dest.read_text())
            if done.get('prompt_sha256')!=doc['prompt_sha256'] or done.get('model')!=model or done.get('provider')!=provider:
                raise ValueError(f'Existing profile belongs to another prompt/model: {dest}')
            skipped+=1;continue
        if limit is not None and count>=limit:break
        if dry_run:
            print(f'DRY RUN {kind} {id_}: {doc["user"][:1100]}')
            count+=1;continue
        if provider=='mock':
            out=mocked(doc);usage={}
        else:
            # Log EACH network attempt (including retries), not just one per entity.
            # The local safety cap does not guarantee Google project quota availability.
            from google.genai import types
            err=None
            for attempt in range(int(cfg['profiles']['max_retries'])):
                now_date=datetime.now(timezone.utc).strftime('%Y-%m-%d')
                if day_counts.get(now_date,0)>=cfg['profiles']['max_daily_requests']:
                    print('LOCAL UTC-DAY API CAP REACHED; pending profiles remain untouched')
                    return {'generated':success,'skipped':skipped,'considered':count,
                            'total':expected,'kind':kind,'status':'local_daily_cap_reached'}
                if last_remote_call is not None:
                    delay=max(0.0,cfg['profiles']['min_interval_seconds']-(time.monotonic()-last_remote_call))
                    if delay:time.sleep(delay)
                last_remote_call=time.monotonic()
                with attempts_path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'utc_date':now_date,'time':utc(),'kind':kind,'id':id_,
                                        'provider':provider,'model':model,'attempt':attempt+1,
                                        'prompt_sha256':doc['prompt_sha256']})+'\n')
                day_counts[now_date]=day_counts.get(now_date,0)+1
                try:
                    res=client.models.generate_content(
                        model=model,contents=doc['user'],config=types.GenerateContentConfig(
                          system_instruction=doc['system'],temperature=cfg['profiles']['temperature'],
                          max_output_tokens=cfg['profiles']['max_output_tokens'],response_mime_type='application/json'))
                    out=res.text
                    met=getattr(res,'usage_metadata',None)
                    usage={'prompt_tokens':getattr(met,'prompt_token_count',None),
                           'candidates_tokens':getattr(met,'candidates_token_count',None),
                           'total_tokens':getattr(met,'total_token_count',None)}
                    err=None;break
                except Exception as exc:
                    err=exc
                    if attempt+1<int(cfg['profiles']['max_retries']):
                        time.sleep(min(60,2**attempt*5))
            if err is not None:raise RuntimeError(f'Gemini failed at {kind} {id_}; safe to retry same run') from err
        parsed=parse_response(out,kind)
        row={'id':id_,'kind':kind,'status':'ok','provider':provider,'model':model,
             'prompt_sha256':doc['prompt_sha256'],'system_sha256':doc['system_sha256'],
             'summarization':parsed['summarization'],'reasoning':parsed['reasoning'],
             'usage':usage,'generated_at':utc()}
        write_json(dest,row);count+=1;success+=1
        print(f'GENERATED {kind} {id_} ({success} new, {skipped} existing)')
    return {'generated':success,'skipped':skipped,'considered':count,'total':expected,'kind':kind,
            'status':'dry_run' if dry_run else 'partial_or_complete'}
