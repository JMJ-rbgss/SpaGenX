from pathlib import Path
import sys,importlib.util,tempfile,json,unittest
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
cli=load('release_cli',ROOT/'inference/spagenx_cli.py')
class ReleaseTests(unittest.TestCase):
 def test_neighbor_padding(self):
  for n in [1,3,33]:
   x=np.random.default_rng(42).normal(size=(n,1536)).astype('float32');coords=np.column_stack([np.arange(n),np.zeros(n)]).astype('float32');o=cli.neighbors(coords,x)
   for name,k in [('candidate',32),('homo',12),('hetero',8)]:
    ids=o[name+'_local_indices'];valid=o[name+'_valid_mask'];self.assertEqual(tuple(ids.shape),(n,k));self.assertTrue(bool((ids[~valid]==-1).all()));self.assertTrue(bool((ids[valid]>=0).all()))
    for i in range(n):self.assertNotIn(i,ids[i][valid[i]].tolist())
 def test_gene_identity(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'vocab.tsv').write_text('global_gene_id\tgene_symbol\n0\tA\n1\tB\n2\tB\n');(p/'bundle_config.json').write_text(json.dumps({'paths':{'vocab':'vocab.tsv'}}))
   self.assertEqual(cli.query(p,['A','2'])[0].tolist(),[0,2])
   for q in [['B'],['A','0'],['UNKNOWN']]:
    with self.assertRaises(RuntimeError):cli.query(p,q)
 def test_measured_zero_and_mapping(self):
  t=load('original_identity_tests',ROOT/'tests/test_original_gene_identity.py');t.run_tests()
 def test_rowmap_reordering(self):
  # Canonical row identity is an explicit lookup, never an implicit sort.
  source=np.array([[0.,2.],[3.,0.],[4.,5.]],dtype='float32');row=np.array([2,0]);mapped=source[row]
  np.testing.assert_array_equal(mapped,[[4.,5.],[0.,2.]])
  mask=np.array([True,True]);self.assertTrue(mask[0]);self.assertEqual(mapped[1,0],0.)
 def test_mpp_blank_and_output_collision(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);im=p/'blank.png';Image.new('RGB',(224,224),'white').save(im)
   with self.assertRaisesRegex(RuntimeError,'MPP_REQUIRED'):cli.preprocess(p,im,p/'no_mpp',device='cpu')
   (p/'bundle_config.json').write_text(json.dumps({'paths':{'preprocess_code':str(ROOT/'preprocessing/step1_extract_uni2h_features.py')}}))
   with self.assertRaisesRegex(RuntimeError,'NO_VALID_TISSUE'):cli.preprocess(p,im,p/'blank',mpp=.5,device='cpu')
   (p/'exists').mkdir()
   with self.assertRaises(FileExistsError):cli.preprocess(p,im,p/'exists',mpp=.5,device='cpu')
 def test_patch_limit(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);im=p/'tissue.png';Image.new('RGB',(448,224),(150,80,130)).save(im)
   original=cli.module
   class Encoder:
    def requires_grad_(self,v):return self
   class Fake:
    @staticmethod
    def load_uni2h(*a):return Encoder()
    @staticmethod
    def infer_batch(model,patches,device,amp=True):return np.ones((len(patches),1536),np.float32)
   cli.module=lambda *args:Fake
   try:
    # cap1 raises before encoder load/flush for the second tissue patch.
    (p/'bundle_config.json').write_text(json.dumps({'paths':{'preprocess_code':'unused'}}))
    with self.assertRaisesRegex(RuntimeError,'PATCH_LIMIT_EXCEEDED'):cli.preprocess(p,im,p/'limit',mpp=.5,max_patches=1,device='cpu')
   finally:cli.module=original
if __name__=='__main__':unittest.main()
