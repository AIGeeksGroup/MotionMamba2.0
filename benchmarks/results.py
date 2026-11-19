"""Write reproducible per-trial metrics and family-cluster summaries."""
import csv
import hashlib
import json
from pathlib import Path
from benchmarks.metrics import family_interval

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def write_results(folder,rows,settings):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(model='MotionMamba2',trials=len(rows),
        metrics={k:family_interval(rows,k,settings['family_bootstrap_repeats'],settings['bootstrap_seed'])
                 for k in ['root_ce_cm','end_ce_cm','cmd_percent']},
        scope='Root/End terminal geometry in centimeters; Cmd is coarse root-state matching, not semantic quality.')
    (folder/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
