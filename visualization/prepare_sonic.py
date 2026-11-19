#!/usr/bin/env python3
"""Apply source-only simulator/controller integration to a user-owned SONIC checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
SOURCE=Path(__file__).resolve().parent/'physics/overlay'
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motion_mamba2.model.configuration import read_config
EXPECTED=read_config('configs/downloads.yaml')['sonic_source_revision']
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
    root=a.source.expanduser().resolve()
    if not (root/'gear_sonic_deploy/CMakeLists.txt').is_file():raise ValueError('Not a SONIC source checkout')
    result=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True)
    if result.stdout.strip()!=EXPECTED:raise ValueError('Checkout the pinned integration revision '+EXPECTED)
    records=[]
    for src in sorted(SOURCE.rglob('*')):
        if not src.is_file() or src.name=='LICENSE':continue
        relative=src.relative_to(SOURCE);dst=root/relative
        new=hashlib.sha256(src.read_bytes()).hexdigest()
        old=hashlib.sha256(dst.read_bytes()).hexdigest() if dst.is_file() else None
        if old==new:continue
        if a.apply:
            # Preserve original bytes next to the destination before changing source.
            if dst.is_file():
                backup=root/'.motion_mamba2_overlay_backup'/relative
                if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,backup)
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        records.append(dict(file=str(relative),original_sha256=old,sha256=new))
    print(json.dumps(dict(applied=a.apply,revision=EXPECTED,changes=records),indent=2))
if __name__=='__main__':main()
