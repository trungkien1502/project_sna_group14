import copy, gzip, json, pickle, tempfile, unittest
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sna.data import audit, index_raw
from sna.profiles import build_prompts, generate, parse_response, stable_sample
from sna.embedding import embed
from sna.package import package
from sna.common import run_dir, sha_file
from sna.experiments import stage_source, replace_assigned, STAGES


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.project=root/'original';original=self.project/'data/raw/amazon';original.mkdir(parents=True)
        src=root/'paper';(src/'data/mapper').mkdir(parents=True)
        self.group=root/'group';(self.group/'notebooks').mkdir(parents=True)
        self.raw=root/'external';self.raw.mkdir()
        trn=csr_matrix(([1,1,1,1],([0,0,1,2],[0,1,1,2])),shape=(3,4))
        val=csr_matrix(([1],([0],[2])),shape=(3,4))
        tst=csr_matrix(([1],([1],[3])),shape=(3,4))
        for fname,mat in zip(('trn_mat.pkl','val_mat.pkl','tst_mat.pkl'),(trn,val,tst)):
            with (original/fname).open('wb') as f:pickle.dump(mat,f)
        with (src/'data/mapper/amazon_item.json').open('w') as f:
            for i in range(4):f.write(json.dumps({'iid':i,'asin':f'asin{i}'})+'\n')
        with (src/'data/mapper/amazon_user.json').open('w') as f:
            for i in range(3):f.write(json.dumps({'uid':i,'reviewerID':f'reviewer{i}'})+'\n')
        with gzip.open(self.raw/'meta.json.gz','wt') as f:
            for i in range(4):f.write(json.dumps({'asin':f'asin{i}','title':f'Book {i}','description':f'Desc {i}'})+'\n')
        with gzip.open(self.raw/'reviews.json.gz','wt') as f:
            for u,i,text in [(0,0,'TRAIN ONLY'),(0,1,'TRAIN SECOND'),(1,1,'TRAIN THIRD'),(2,2,'TRAIN FOURTH'),
                             (0,2,'VALID SHOULD NEVER APPEAR'),(1,3,'TEST SHOULD NEVER APPEAR')]:
                f.write(json.dumps({'reviewerID':f'reviewer{u}','asin':f'asin{i}','reviewText':text})+'\n')
        self.cfg={
         'run_id':'unittest','paths':{'project_root':str(self.project),'run_root':str(root/'branches'),
          'rlmrec_repo':str(src),'group_repo':str(self.group),
          'raw_metadata':str(self.raw/'meta.json.gz'),'raw_reviews':str(self.raw/'reviews.json.gz')},
         'data':{'expected_users':3,'expected_items':4,'expected_train':4,'expected_validation':1,
                 'expected_test':1,'min_metadata_coverage':1,'max_description_chars':100,'max_review_chars':200},
         'profiles':{'provider':'mock','model':'MOCK','temperature':0,'max_output_tokens':300,
                     'max_history_items':2,'history_seed':2023,'min_interval_seconds':0,
                     'max_daily_requests':1000,'max_retries':2},
         'embedding':{'provider':'mock_hash','batch_size':2,'local_model':'mock','openai_model':'mock'},
         'source':{'rlmrec_commit':'abc','group_commit':'xyz'},'execution':{'notebook_timeout_seconds':1}}

    def prep(self):
        audit(self.cfg);index_raw(self.cfg)
        build_prompts(self.cfg,'item');generate(self.cfg,'item')
        build_prompts(self.cfg,'user');generate(self.cfg,'user')

    def test_raw_index_excludes_holdout_reviews(self):
        self.cfg['profiles']['user_review_policy']='train_pair'
        self.assertEqual(audit(self.cfg)['counts']['train'],4)
        r=index_raw(self.cfg)
        self.assertEqual(r['mapped_nontrain_review_rows_rejected'],2)
        s=(run_dir(self.cfg)/'raw_index/train_reviews.jsonl').read_text()
        self.assertNotIn('VALID SHOULD NEVER APPEAR',s)
        self.assertNotIn('TEST SHOULD NEVER APPEAR',s)
        self.assertIn('TRAIN ONLY',s)

    def test_end_to_end_mock_data_package(self):
        self.prep()
        branch=run_dir(self.cfg)
        content=(branch/'prompts/user/0.json').read_text()
        self.assertNotIn('VALID SHOULD NEVER APPEAR',content)
        self.assertNotIn('TEST SHOULD NEVER APPEAR',content)
        self.assertIn('MOCK profile for book testing only',content)
        for kind in ('item','user'):embed(self.cfg,kind)
        self.assertRaises(ValueError,package,self.cfg)
        manifest=package(self.cfg,allow_mock=True)
        self.assertEqual(manifest['status'],'PASS_TEST_ONLY')
        d=branch/'data/raw/amazon'
        self.assertEqual(len(list(d.glob('*.pkl'))),7)
        with (d/'usr_prf.pkl').open('rb') as f: profiles=pickle.load(f)
        with (d/'itm_emb_np.pkl').open('rb') as f:embeddings=pickle.load(f)
        self.assertEqual(len(profiles),3)
        self.assertEqual(embeddings.shape,(4,16))
        for name in ('trn_mat.pkl','val_mat.pkl','tst_mat.pkl'):
            self.assertEqual(sha_file(d/name),sha_file(self.project/'data/raw/amazon'/name))

    def test_split_overlap_fails_closed(self):
        d=self.project/'data/raw/amazon'
        with (d/'val_mat.pkl').open('wb') as f:
            pickle.dump(csr_matrix(([1],([0],[0])),shape=(3,4)),f)
        with self.assertRaisesRegex(ValueError,'Overlapping'): audit(self.cfg)

    def test_prompt_immutable(self):
        audit(self.cfg);index_raw(self.cfg);build_prompts(self.cfg,'item')
        p=run_dir(self.cfg)/'prompts/item/0.json'
        v=json.loads(p.read_text());v['prompt_sha256']='tampered';p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'locked'):build_prompts(self.cfg,'item')

    def test_no_raw_review_event_provenance_claim(self):
        audit(self.cfg); m=index_raw(self.cfg)
        self.assertIn('cannot prove',m['limitation'])

    def test_response_strict_json(self):
        with self.assertRaises(ValueError):parse_response('```json {"x":2} ```','item')
        with self.assertRaises(ValueError):parse_response('{"summarization":"ok"}','user')
        self.assertEqual(parse_response('{"summarization":"one","reasoning":"two"}','user')['summarization'],'one')

    def test_locked_history_determinism(self):
        self.assertEqual(stable_sample([3,2,1,0],2,0,2023),stable_sample([0,1,2,3],2,0,2023))

    def test_notebook_patch_does_not_modify_original(self):
        import nbformat
        folder=self.group/'notebooks';p=folder/'03_E1_Full_Training.ipynb'
        nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            'MODEL = "lightgcn_plus"\nSEED = 2023\nROOT = Path("/content/drive/MyDrive/Data_SNA_Group14")')])
        nbformat.write(nb,p)
        before=sha_file(p)
        out,report=stage_source(self.cfg,'e1-train',model='lightgcn',seed=2024,execute=False)
        self.assertEqual(sha_file(p),before)
        src=nbformat.read(out,as_version=4).cells[1].source
        self.assertIn('MODEL = "lightgcn"',src)
        self.assertIn('SEED = 2024',src)
        self.assertIn(str(run_dir(self.cfg)),src)
        self.assertNotIn('/content/drive/MyDrive/Data_SNA_Group14',src)


    def test_locked_run_spec_rejects_model_change(self):
        audit(self.cfg)
        self.cfg['profiles']['model']='ANOTHER_LLM'
        with self.assertRaisesRegex(ValueError,'config drift'):
            audit(self.cfg)

    def test_review_free_user_prompts_optional(self):
        self.cfg['profiles']['user_review_policy']='none'
        self.prep()
        p=run_dir(self.cfg)/'prompts/user/0.json'
        d=json.loads(p.read_text())
        self.assertNotIn('TRAIN ONLY',d['user'])
        self.assertNotIn('TRAIN SECOND',d['user'])
        self.assertEqual(d['provenance']['review_gate'],'all_reviews_omitted')

    def test_plan_is_safe_and_has_all_e1_e4(self):
        from sna.orchestrator import run_all
        plan=run_all(self.cfg,execute=False)
        self.assertEqual(plan['status'],'PLAN_ONLY_NO_GPU_CALLS')
        self.assertEqual(plan['n_steps'],34)
        self.assertFalse(any(x['output_ready'] for x in plan['steps']))

    def test_original_exporter_not_changed_by_staging(self):
        import nbformat
        (self.group/'src/inference').mkdir(parents=True)
        exp=self.group/'src/inference/export_topk.py'
        exp.write_text('"""--root /content/drive/MyDrive/Data_SNA_Group14"""\n')
        src=self.group/'notebooks/04_E1_Export_TopK.ipynb'
        nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
          'MODEL = "lightgcn"\nSEED = 2023\nROOT = Path("/content/drive/MyDrive/Data_SNA_Group14")\nGROUP_REPO = Path("/content/project_sna_group14")')])
        nbformat.write(nb,src)
        before=sha_file(exp)
        out,manifest=stage_source(self.cfg,'e1-export',model='lightgcn',seed=2023,execute=False)
        staged=nbformat.read(out,as_version=4).cells[1].source
        self.assertIn('/runtime/group_repo',staged)
        self.assertIn(str(run_dir(self.cfg)),staged)
        self.assertEqual(before,sha_file(exp))
        self.assertNotIn('/content/drive/MyDrive/Data_SNA_Group14',
            (run_dir(self.cfg)/'runtime/group_repo/src/inference/export_topk.py').read_text())


    def test_native_colab_bootstrap_and_stage_source_isolated(self):
        import nbformat
        src=self.group/'notebooks/03_E1_Full_Training.ipynb'
        nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            'MODEL = "lightgcn_plus"\nSEED = 2023\nROOT = Path("/content/drive/MyDrive/Data_SNA_Group14")')])
        nbformat.write(nb,src)
        out,meta=stage_source(self.cfg,'e1-train',model='lightgcn_plus',seed=2025)
        cells=nbformat.read(out,as_version=4).cells
        self.assertIn('from google.colab import drive',cells[0].source)
        self.assertIn('BRANCH',cells[0].source)
        self.assertIn('MODEL = "lightgcn_plus"',cells[1].source)
        self.assertIn('SEED = 2025',cells[1].source)
        self.assertEqual(meta['mode'],'NATIVE_COLAB_STAGED_UNEXECUTED')
        for cell in cells:
            if cell.cell_type=='code':compile(cell.source,'stage_cell','exec')

    def test_unverified_nested_kernel_execution_disabled(self):
        from sna.experiments import run_stage
        from sna.orchestrator import run_all
        with self.assertRaisesRegex(RuntimeError,'Native Colab execution only'):
            run_stage(self.cfg,'e1-train',model='lightgcn',seed=2023,execute=True)
        with self.assertRaisesRegex(RuntimeError,'disables run-all'):
            run_all(self.cfg,execute=True)

    def test_no_review_source_required_in_strict_none_mode(self):
        self.cfg['profiles']['user_review_policy']='none'
        Path(self.cfg['paths']['raw_reviews']).unlink()
        audit(self.cfg)
        report=index_raw(self.cfg)
        self.assertEqual(report['review_gate'],'NO_RAW_REVIEWS_USED')
        self.assertIsNone(report['source_hashes_sha256']['reviews'])
        self.assertEqual(report['matched_unambiguous_train_reviews'],0)

    def test_user_facing_colab_notebook_syntax(self):
        import nbformat
        folder=Path(__file__).resolve().parents[1]/'notebooks'
        for name in ('01_START_HERE_Colab.ipynb','02_PREPARE_STAGE_Colab.ipynb'):
            f=folder/name;self.assertTrue(f.is_file())
            nb=nbformat.read(f,as_version=4);nbformat.validate(nb)
            for i,c in enumerate(nb.cells):
                if c.cell_type=='code':compile(c.source,f'{name}:cell{i}','exec')



    def test_original_drive_root_rewrite_with_native_prelude(self):
        import nbformat
        self.cfg['paths']['project_root']='/content/drive/MyDrive/Data_SNA_Group14'
        p=self.group/'notebooks/03_E1_Full_Training.ipynb'
        nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            'MODEL = "lightgcn"\nSEED = 2023\nROOT = Path("/content/drive/MyDrive/Data_SNA_Group14")')])
        nbformat.write(nb,p)
        output,manifest=stage_source(self.cfg,'e1-train',model='lightgcn',seed=2023)
        result=nbformat.read(output,as_version=4)
        self.assertIn('/content/drive/MyDrive/Data_SNA_Group14',result.cells[0].source)  # guard display only
        self.assertNotIn('/content/drive/MyDrive/Data_SNA_Group14',result.cells[1].source)
        self.assertIn(str(run_dir(self.cfg)),result.cells[1].source)



if __name__=='__main__':unittest.main()
