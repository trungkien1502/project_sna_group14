from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
from .common import run_dir, write_json, sha_file, sha_text, utc
from .data import assert_same_splits


def test_embed(texts,dim=16):
    # Non-semantic fake embeddings; ONLY for the offline test suite.
    out=[]
    for s in texts:
        rng=np.random.default_rng(int(sha_text(s)[:15],16))
        a=rng.normal(size=dim).astype('float32');a/=max(np.linalg.norm(a),1e-8)
        out.append(a)
    return np.stack(out)


def _encode(texts, cfg,provider, model,client=None):
    if provider=='mock_hash':return test_embed(texts)
    if provider=='local':
        from sentence_transformers import SentenceTransformer
        if client is None:client=SentenceTransformer(model)
        return np.asarray(client.encode(texts,batch_size=cfg['embedding']['batch_size'],
                                        show_progress_bar=False,normalize_embeddings=False),dtype='float32')
    if provider=='openai':
        # OpenAI is used ONLY when explicitly enabled by the user.
        res=client.embeddings.create(model=model,input=texts)
        return np.asarray([o.embedding for o in sorted(res.data,key=lambda x:x.index)],dtype='float32')
    raise ValueError(provider)


def embed(cfg,kind):
    if kind not in ('item','user'):raise ValueError(kind)
    assert_same_splits(cfg)
    branch=run_dir(cfg); source=branch/'profiles'/kind
    count=cfg['data']['expected_items'] if kind=='item' else cfg['data']['expected_users']
    provider=cfg['embedding']['provider'];model=(cfg['embedding']['local_model'] if provider=='local' else
            cfg['embedding']['openai_model'] if provider=='openai' else 'MOCK_HASH_16')
    if provider not in ('local','openai','mock_hash'):raise ValueError('embedding.provider not supported')
    from .profiles import prompt_system
    texts=[];profile_hashes=[]
    for id_ in range(count):
        fp=source/f'{id_}.json'
        if not fp.is_file():raise FileNotFoundError(f'Missing {kind} profile {id_}; finish all profile calls first')
        data=json.loads(fp.read_text())
        if data.get('status')!='ok' or data.get('model')!=cfg['profiles']['model'] or data.get('provider')!=cfg['profiles']['provider']:
            raise ValueError(f'Stale/different LLM profile: {fp}')
        # Verify provenance against frozen prompt.
        prompt=json.loads((branch/'prompts'/kind/f'{id_}.json').read_text())
        if prompt['prompt_sha256']!=data['prompt_sha256']:raise ValueError(f'Prompt/profile hash mismatch {id_}')
        texts.append(data['summarization']);profile_hashes.append(sha_file(fp))
    provenance_hash=sha_text('|'.join(profile_hashes))
    dest=branch/'embeddings'/kind;dest.mkdir(parents=True,exist_ok=True)
    meta_file=dest/'manifest.json';npy_file=dest/'embeddings.npy'
    if meta_file.exists():
        old=json.loads(meta_file.read_text())
        if old['source_profiles_hash']!=provenance_hash or old['provider']!=provider or old['model']!=model:
            raise ValueError('Existing embedding belongs to a different profile/model; use a new run_id')
        if npy_file.is_file():return old
    batch=int(cfg['embedding']['batch_size']);client=None
    if provider=='openai':
        if not os.environ.get('OPENAI_API_KEY') or os.environ.get('SNA_ALLOW_REMOTE_EMBEDDING')!='YES':
            raise EnvironmentError('Paid OpenAI embeddings require OPENAI_API_KEY and SNA_ALLOW_REMOTE_EMBEDDING=YES')
        from openai import OpenAI
        client=OpenAI()
    elif provider=='local':
        from sentence_transformers import SentenceTransformer
        client=SentenceTransformer(model)
    pieces=[]
    # Local per-batch checkpoints, preserving token consumption on interruption.
    for start in range(0,count,batch):
        end=min(count,start+batch)
        seg=dest/f'batch_{start}_{end}.npy'
        batch_sha=sha_text('\n'.join(texts[start:end]))
        hdr=dest/f'batch_{start}_{end}.json'
        if seg.exists() and hdr.exists():
            m=json.loads(hdr.read_text())
            if m.get('batch_sha256')!=batch_sha or m.get('model')!=model:
                raise ValueError(f'Stale embedding batch: {seg}')
            a=np.load(seg,allow_pickle=False)
        else:
            a=_encode(texts[start:end],cfg,provider,model,client)
            if a.ndim!=2 or a.shape[0]!=(end-start) or not np.isfinite(a).all():
                raise ValueError(f'Bad embedding batch shape/data at {start}')
            np.save(seg,a)
            write_json(hdr,{'range':[start,end],'batch_sha256':batch_sha,'provider':provider,'model':model,'dimension':int(a.shape[1])})
        pieces.append(a)
        print(f'EMBED {kind} {end}/{count}')
    arr=np.concatenate(pieces,axis=0).astype(np.float32)
    if arr.shape[0]!=count or not np.isfinite(arr).all():raise ValueError('Bad final embedding array')
    np.save(npy_file,arr)
    meta={'status':'PASS','kind':kind,'provider':provider,'model':model,'shape':list(arr.shape),
          'source_profiles_hash':provenance_hash,'embedding_sha256':sha_file(npy_file),
          'note':'Embedding model is a distinct experimental factor; default local model is NOT original text-embedding-ada-002',
          'created_utc':utc()}
    write_json(meta_file,meta);return meta
