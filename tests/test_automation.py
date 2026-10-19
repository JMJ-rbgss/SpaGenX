"""Standard-library-only tests: automation must not run models in preview mode."""
from pathlib import Path
import os,shutil,subprocess,tempfile,unittest
SOURCE=Path(__file__).resolve().parents[1]/'scripts/infer_wsi.sh'
class AutomationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'scripts').mkdir();shutil.copyfile(SOURCE,self.root/'scripts/infer_wsi.sh')
  (self.root/'bundle_config.json').write_text('{}');(self.root/'synthetic.png').write_bytes(b'synthetic placeholder, never decoded');(self.root/'genes.txt').write_text('SYNTHETIC_GENE\n')
  self.env=os.environ.copy();self.env.update(WSI=str(self.root/'synthetic.png'),GENES_FILE=str(self.root/'genes.txt'),ORGAN='Breast',RUN_DIR=str(self.root/'new_run'),PYTHON_BIN='MODEL_EXECUTION_IS_FORBIDDEN_IN_THIS_TEST')
 def tearDown(self):self.tmp.cleanup()
 def run_script(self,*args):return subprocess.run(['bash',str(self.root/'scripts/infer_wsi.sh'),*args],env=self.env,capture_output=True,text=True)
 def test_preview_does_not_execute(self):
  r=self.run_script('--dry-run');self.assertEqual(r.returncode,0,r.stderr);self.assertIn('DRY_RUN',r.stdout);self.assertFalse((self.root/'new_run').exists())
 def test_preview_is_default(self):
  r=self.run_script();self.assertEqual(r.returncode,0,r.stderr);self.assertIn('DRY_RUN',r.stdout)
 def test_existing_output_rejected(self):
  (self.root/'new_run').mkdir();r=self.run_script();self.assertNotEqual(r.returncode,0);self.assertIn('Output already exists',r.stderr)
 def test_missing_config_rejected(self):
  (self.root/'bundle_config.json').unlink();r=self.run_script();self.assertNotEqual(r.returncode,0);self.assertIn('bundle_config.json',r.stderr)
if __name__=='__main__':unittest.main()
