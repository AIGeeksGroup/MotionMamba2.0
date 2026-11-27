"""Launch the 100k-update GT recipe with torchrun (eight GPUs by default)."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from motion_mamba2.model.variants import add_model_flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--config', default='configs/training_motion_mamba2.yaml')
    parser.add_argument('--resume')
    add_model_flags(parser)
    args = parser.parse_args()
    devices = [value.strip() for value in args.gpus.split(',') if value.strip()]
    if len(set(devices)) != len(devices) or not devices or 512 % len(devices):
        parser.error('Provide unique GPUs whose count divides global batch 512')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices),
                       CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={len(devices)}', '--module',
               'motion_mamba2.training.training_motion_mamba2', '--config', args.config]
    if args.minimal_baseline:
        command.append('--minimal_baseline')
    if args.resume:
        command += ['--resume', str(Path(args.resume).expanduser().resolve())]
    subprocess.run(command, env=environment, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    main()
