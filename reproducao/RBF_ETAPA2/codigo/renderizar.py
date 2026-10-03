"""Fornece ao renderizador original o dataset recalculado, sem editar o fonte."""
from pathlib import Path
import os,runpy,shutil
ROOT=Path(__file__).resolve().parents[1]
model=ROOT/'original_reproduzivel';job=ROOT/'figura';job.mkdir(exist_ok=True)
(model/'data/model').mkdir(exist_ok=True)
shutil.copy2(ROOT/'dados/r2d2bfix2_production_ensemble.csv.gz',
             model/'data/model/r2d2bfix2_production_ensemble.csv.gz')
os.environ['ISDBD_ROOT']=str(model);os.environ['ISDBD_JOB']=str(job)
runpy.run_path(str(ROOT/'codigo/render_original.py'),run_name='__main__')
