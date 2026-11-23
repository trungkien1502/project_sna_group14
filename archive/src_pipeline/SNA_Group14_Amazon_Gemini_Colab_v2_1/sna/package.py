"""Produce the seven-file RLMRec input directory atomically in the isolated run branch."""
from __future__ import annotations
import json, os, pickle, shutil
from pathlib import Path
import numpy as np
from .common import assert_inputs, run_dir, sha_file, write_json, utc
from .data import assert_same_splits


def package(cfg,allow_mock=False):
    audit=assert_same_splits(cfg)
    provider=cfg['profiles']['provider']; embed_provider=cfg['embedding']['provider']
    test_only=(provider=='mock' or embed_provider=='mock_hash')
    if test_only and not allow_mock:
        raise ValueError('MOCK profile/embedding is NOT a valid research input; pass --allow-mock for TESTS ONLY')
    branch=run_dir(cfg);orig=assert_inputs(cfg)
    dest=branch/'data/raw/amazon';stage=branch/'data/raw/amazon._partial'
    manifest_path=branch/'data/package_manifest.json'
    if manifest_path.is_file():
        old=json.loads(manifest_path.read_text())
        if old.get('profile_model')!=cfg['profiles']['model'] or old.get('embedding_provider')!=embed_provider:
            raise ValueError('This run ID is bound to a different profile or embedding configuration')
        for filename,sha in old['packed_files_sha256'].items():
            if not (dest/filename).is_file() or sha_file(dest/filename)!=sha:
                raise ValueError(f'Previously packed artifact changed: {filename}')
        return old
    if dest.exists() or stage.exists():
        raise FileExistsError('Incomplete output directory exists. Inspect before cleanup; never overwrite old artifacts.')
    stage.mkdir(parents=True)
    output={};dimensions={}
    try:
        for kind,count,prf,emb in [('user',cfg['data']['expected_users'],'usr_prf.pkl','usr_emb_np.pkl'),
                                   ('item',cfg['data']['expected_items'],'itm_prf.pkl','itm_emb_np.pkl')]:
            inputs=[]
            for id_ in range(count):
                pp=branch/'profiles'/kind/f'{id_}.json'
                if not pp.is_file():raise FileNotFoundError(f'Missing {kind} profile id={id_}')
                d=json.loads(pp.read_text(encoding='utf-8'))
                if d.get('model')!=cfg['profiles']['model'] or d.get('provider')!=provider or d.get('status')!='ok':
                    raise ValueError(f'Profile model mismatch {pp}')
                prompt=json.loads((branch/'prompts'/kind/f'{id_}.json').read_text())
                if d['prompt_sha256']!=prompt['prompt_sha256']:raise ValueError(f'Prompt/profile mismatch {id_}')
                inputs.append({'profile':d['summarization'],'reasoning':d['reasoning']})
            emb_path=branch/'embeddings'/kind/'embeddings.npy'
            mfile=branch/'embeddings'/kind/'manifest.json'
            if not mfile.exists() or not emb_path.exists():raise FileNotFoundError(f'Embedding for {kind} missing')
            m=json.loads(mfile.read_text())
            if m['provider']!=embed_provider or sha_file(emb_path)!=m['embedding_sha256']:
                raise ValueError(f'Embedding provenance mismatch {kind}')
            arr=np.load(emb_path,allow_pickle=False)
            if arr.ndim!=2 or arr.shape[0]!=count or not np.isfinite(arr).all():
                raise ValueError(f'Embedding array invalid {kind}: {arr.shape}')
            dimensions[kind]=arr.shape[1]
            for fname, object_ in [(prf,inputs),(emb,arr)]:
                path=stage/fname
                with open(path,'wb') as f:pickle.dump(object_,f,pickle.HIGHEST_PROTOCOL)
                output[fname]=sha_file(path)
        if dimensions['user']!=dimensions['item']:
            raise ValueError('User/item embeddings must share dimension for RLMRec-Con MLP')
        for n in ('trn_mat.pkl','val_mat.pkl','tst_mat.pkl'):
            src=orig/n;target=stage/n
            shutil.copyfile(src,target)
            if sha_file(target)!=audit['source_hashes_sha256'][n]:
                raise ValueError(f'Split copy checksum mismatch {n}')
            output[n]=sha_file(target)
        # Prevent accidentally identical embeddings being carried over.
        for name in ('usr_emb_np.pkl','itm_emb_np.pkl'):
            old=orig/name
            if old.exists() and output[name]==sha_file(old):
                raise ValueError(f'New embedding IDENTICAL to author embedding: {name}')
        # Change only a new, never-existing destination; no author data touched.
        os.rename(stage,dest)
        report={'status':'PASS_TEST_ONLY' if test_only else 'PASS',
                'dataset':'amazon','n_users':cfg['data']['expected_users'],
                'n_items':cfg['data']['expected_items'],
                'profile_provider':provider,'profile_model':cfg['profiles']['model'],
                'embedding_provider':embed_provider,
                'embedding_model':cfg['embedding']['local_model'] if embed_provider=='local' else cfg['embedding']['openai_model'] if embed_provider=='openai' else 'MOCK_HASH_16',
                'embedding_dimension':dimensions['user'],
                'source_split_hashes':audit['source_hashes_sha256'],
                'packed_files_sha256':output,
                'run_root':str(branch),'original_dataset_unmodified':True,'created_utc':utc()}
        write_json(manifest_path,report)
        return report
    except Exception:
        # The partial output never becomes usable; retain it for diagnosis rather than silently removing evidence.
        raise
