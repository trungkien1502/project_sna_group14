"""Profile leakage forensic test (mention-rate test).
Usage: python profile_leakage_forensic.py <dir with *_mat.pkl, usr_prf.pkl> <dir with amazon_book_titles.json>
H_leak: if val/test items were in the profile prompt, the LLM mentions them at ~ the same rate as train items.
"""
import pickle, re, random, numpy as np, scipy.sparse as sp, json, sys
# Restricted unpickler: only numpy/scipy/builtins classes may be constructed.
class R(pickle.Unpickler):
    def find_class(self, m, n):
        if m.split('.')[0] in ('numpy','scipy','builtins','collections','_codecs','copyreg'): return super().find_class(m,n)
        raise pickle.UnpicklingError(f'blocked {m}.{n}')
def L(p): return R(open(p,'rb')).load()

D, M = sys.argv[1], sys.argv[2]
norm = lambda s: ' '+re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()+' '
trn, val, tst = [L(f'{D}/{n}_mat.pkl').tocsr() for n in ('trn','val','tst')]
prf = L(f'{D}/usr_prf.pkl')
titles = {r['iid']: (r['title'] or '') for r in json.load(open(f'{M}/amazon_book_titles.json'))}
def aliases(t):
    out=set()
    for c in (t, t.split('(')[0], t.split(':')[0]):
        a=norm(c)
        if len(a.split())>=2 and len(a.strip())>=8: out.add(a)
    return out
AL={i:aliases(t) for i,t in titles.items()}
TXT={u: norm((p.get('profile') or '')+' '+(p.get('reasoning') or '')) for u,p in prf.items()}
def ment(u,i): return any(a in TXT[u] for a in AL[i])
pop=np.asarray(trn.sum(0)).ravel(); dec=np.digitize(pop, np.quantile(pop,[.1*k for k in range(1,10)]))
bydec={d:np.where(dec==d)[0] for d in range(10)}
inter=(trn+val+tst).tocsr()
rng=np.random.default_rng(0)
res={}; hits={'val':[],'test':[]}
for name,mat in (('train',trn),('val',val),('test',tst)):
    n=h=0; ctrl_n=ctrl_h=0
    for u in range(mat.shape[0]):
        seen=set(inter[u].indices)
        for i in mat[u].indices:
            if not AL[i]: continue
            n+=1; m=ment(u,i); h+=m
            if m and name!='train': hits[name].append((u,int(i)))
            for _ in range(3):  # popularity-matched non-interacted controls
                j=int(rng.choice(bydec[dec[i]]))
                if j in seen or not AL[j]: continue
                ctrl_n+=1; ctrl_h+=ment(u,j)
    res[name]=dict(pairs=n,mentions=h,rate=h/n,ctrl_pairs=ctrl_n,ctrl_mentions=ctrl_h,ctrl_rate=ctrl_h/ctrl_n)
for k,v in res.items(): print(k, {a:(round(b,6) if isinstance(b,float) else b) for a,b in v.items()})
# series explanation: held-out alias contained in a TRAIN item title of same user, or a train item alias also mentioned & shares a 2+word alias
def series(u,i):
    tr=trn[u].indices
    for a in AL[i]:
        for j in tr:
            if a in norm(titles[j]): return True
    return False
for k in hits:
    s=[series(u,i) for u,i in hits[k]]
    print(k,'held-out mentions',len(s),'explained by same-series train title',sum(s), 'users',len({u for u,_ in hits[k]}))
allh=hits['val']+hits['test']
print('UNEXPLAINED:')
for u,i in allh:
    if not series(u,i): print(' ',u,i,repr(titles[i]),'| train titles:',[titles[j][:40] for j in trn[u].indices][:6])
