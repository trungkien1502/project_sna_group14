"""Popularity-group analysis (instructor comment: which popularity levels are recommended more, which are recommended correctly).

Items are sorted by train degree (desc, tie-break by item ID) and cut into 5 equal-size groups G1..G5.
G1 = top 20% = the E2 Head set; G2..G5 split the E2 Tail set.
Per group, at K (default 20), from existing Top-K exports (no retraining):
  ExposureShare  share of all Top-K slots that fall in the group
  TestShare      share of test interactions in the group (reference: what a perfect recommender would expose)
  Precision      hits in group / Top-K slots in group  ("when the model recommends this group, how often is it right")
  Recall         conditional macro recall: mean over users with >=1 test item in the group of hits_in_group / test_items_in_group
                 (same definition as E2 Head/Tail Recall)
Usage: python popularity_groups.py <raw dir with trn_mat.pkl, tst_mat.pkl> <experiments dir> <out csv> [K]
"""
import sys, pickle, glob, os
import numpy as np


class _R(pickle.Unpickler):
    def find_class(self, m, n):
        if m.split('.')[0] in ('numpy', 'scipy', 'builtins', 'collections', '_codecs', 'copyreg'):
            return super().find_class(m, n)
        raise pickle.UnpicklingError(f'blocked {m}.{n}')


def load(p):
    with open(p, 'rb') as f:
        return _R(f).load()


RAW, EXP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 20
NG = 5

trn = load(f'{RAW}/trn_mat.pkl').tocsr()
tst = load(f'{RAW}/tst_mat.pkl').tocsr()
n_items = trn.shape[1]
deg = np.asarray(trn.sum(0)).ravel()
order = np.lexsort((np.arange(n_items), -deg))          # degree desc, item id asc
group = np.empty(n_items, dtype=int)
bounds = np.ceil(np.linspace(0, n_items, NG + 1)).astype(int)
for g in range(NG):
    group[order[bounds[g]:bounds[g + 1]]] = g

MODELS = {
    'Most Popular': [f'{EXP}/predictions/most_popular/topk_test.npz'],
    'LightGCN': sorted(glob.glob(f'{EXP}/predictions/lightgcn/seed_*/topk_test.npz')),
    'User-only': sorted(glob.glob(f'{EXP}/E3/runs/user_only/*/seed_*/topk/topk_test.npz')),
    'Item-only': sorted(glob.glob(f'{EXP}/E3/runs/item_only/*/seed_*/topk/topk_test.npz')),
    'RLMRec-Con (Full)': sorted(glob.glob(f'{EXP}/predictions/lightgcn_plus/seed_*/topk_test.npz')),
}


def run_metrics(path):
    z = np.load(path, allow_pickle=False)
    users, items = z['user_ids'], z['item_ids'][:, :K]
    slots = np.zeros(NG); hits = np.zeros(NG)
    rec_sum = np.zeros(NG); rec_n = np.zeros(NG); test_cnt = np.zeros(NG)
    for u, row in zip(users, items):
        t = tst[u].indices
        if len(t) == 0:
            continue
        tset = set(t.tolist())
        tg = np.bincount(group[t], minlength=NG)
        test_cnt += tg
        rg = group[row]
        hit = np.fromiter((i in tset for i in row), bool, len(row))
        slots += np.bincount(rg, minlength=NG)
        hg = np.bincount(rg[hit], minlength=NG)
        hits += hg
        m = tg > 0
        rec_sum[m] += hg[m] / tg[m]; rec_n[m] += 1
    exp_cnt = np.bincount(items.ravel(), minlength=n_items)
    cov = np.array([(exp_cnt[group == g] > 0).mean() for g in range(NG)])
    return dict(ExposureShare=slots / slots.sum(), TestShare=test_cnt / test_cnt.sum(), Coverage=cov,
                Precision=np.divide(hits, slots, out=np.zeros(NG), where=slots > 0),
                Recall=rec_sum / rec_n, n_users=rec_n)


rows = []
for model, paths in MODELS.items():
    if not paths:
        print('missing', model); continue
    runs = [run_metrics(p) for p in paths]
    for g in range(NG):
        r = dict(model=model, K=K, group=f'G{g + 1}', n_seeds=len(runs),
                 n_items=int((group == g).sum()),
                 deg_min=int(deg[group == g].min()), deg_max=int(deg[group == g].max()),
                 n_eligible_users=int(runs[0]['n_users'][g]))
        for k in ('ExposureShare', 'TestShare', 'Coverage', 'Precision', 'Recall'):
            v = np.array([x[k][g] for x in runs])
            r[f'{k}_mean'] = v.mean()
            r[f'{k}_sd'] = v.std(ddof=1) if len(v) > 1 else np.nan
        rows.append(r)

import csv
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['model']:<18} {r['group']} deg[{r['deg_min']},{r['deg_max']}] "
          f"exp={r['ExposureShare_mean']:.4f} test={r['TestShare_mean']:.4f} "
          f"prec={r['Precision_mean']:.4f} rec={r['Recall_mean']:.4f}±{r['Recall_sd']:.4f}")
