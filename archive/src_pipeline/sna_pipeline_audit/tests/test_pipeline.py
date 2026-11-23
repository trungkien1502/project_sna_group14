import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import load_splits, split, audit
from argparse import Namespace

def test_temporal_split_and_checksum(tmp_path):
    rows=[]
    for u in ['A','B']:
        for i in range(6): rows.append({'user_id':u,'item_id':f'I{i}','timestamp':f'2026-01-{i+1:02d}T00:00:00Z'})
    p=tmp_path/'src.csv';pd.DataFrame(rows).to_csv(p,index=False)
    out=tmp_path/'split'
    split(Namespace(interactions=str(p),out=str(out),min_history=5))
    d,m=load_splits(out)
    assert list(map(len,[d['train'],d['validation'],d['test']]))==[8,2,2]
    with (out/'test.csv').open('a') as f: f.write('A,I99,2027-01-01\n')
    with pytest.raises(ValueError,match='CHANGED SPLIT'): load_splits(out)
