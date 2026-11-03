"""Public SpaGenX training entry point forwarding to the native trainer."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('spagenx_trainer.py')), run_name='__main__')
