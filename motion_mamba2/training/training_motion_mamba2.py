"""Train the selected root/body diffusion model from scratch on real motion."""
import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import torch

from motion_mamba2.model.configuration import ROOT, load_environment, read_config
from motion_mamba2.model.variants import add_model_flags
from . import batching, checkpoint, data, distributed as parallel, history, losses
from .optimizer import FP32MasterAdamW
from .precision import prediction_loss_context
from .runtime import create_model


def validate_config(config):
    if config['updates'] != 100000 or config['global_batch'] != 512:
        raise ValueError('This recipe is 100000 updates with global batch 512')
    if config['microbatch'] < 1 or 512 % config['microbatch']:
        raise ValueError('Microbatch must divide global batch 512')
    options = config['optimizer']
    if not (0 < options['initial_learning_rate'] <= options['learning_rate']
            and 0 < options['minimum_learning_rate'] <= options['learning_rate']):
        raise ValueError('Invalid learning-rate schedule')
    if not 2 <= options['warmup_updates'] < 100000 or config['save_every'] < 1:
        raise ValueError('Invalid update schedule')
    if 'predicted_history_ratio' in config['data']:
        raise ValueError('Predicted history is unsupported; use clean GT history')
    if options['epsilon'] <= 0:
        raise ValueError('Adam epsilon must be positive')
    for key in ['text_dropout', 'constraint_dropout']:
        if not 0 <= config['data'][key] < 1:
            raise ValueError(f'Invalid {key}')


def learning_rate(config, step):
    if not 1 <= step <= config['updates']:
        raise ValueError('Learning-rate step must be within the update budget')
    options = config['optimizer']
    warmup = options['warmup_updates']
    if step <= warmup:
        # Inclusive warmup: update 1 uses the initial rate; update 500 the peak.
        fraction = (step - 1) / (warmup - 1)
        return options['initial_learning_rate'] + fraction * (
            options['learning_rate'] - options['initial_learning_rate'])
    progress = (step - warmup) / (config['updates'] - warmup)
    return options['minimum_learning_rate'] + .5 * (options['learning_rate'] -
        options['minimum_learning_rate']) * (1 + math.cos(math.pi * progress))


def resource_contract(config, plan):
    # Paths are recorded in local run outputs; released configuration has no local defaults.
    values = load_environment(config['paths_config'])
    code = {str(path.relative_to(ROOT)): checkpoint.file_hash(path)
            for folder in ['model', 'kernels', 'training']
            for path in sorted((ROOT / 'motion_mamba2' / folder).glob('*.py'))}
    resources = {key: checkpoint.file_hash(values[key])
                 for key in ['MM2_CODEC_CHECKPOINT']}
    resources['model_config'] = checkpoint.file_hash(ROOT / config['model_config'])
    resources['data_index'] = checkpoint.file_hash(config['data']['index_path'])
    resources['text'] = checkpoint.tensor_hash({caption: checkpoint.file_hash(path)
                                               for caption, path in plan.text_files.items()})
    resources['features'] = checkpoint.tensor_hash({path: checkpoint.file_hash(path)
                                                   for path in sorted(plan.feature_paths)})
    skeleton = Path(values['MM2_SKELETON_PATH'])
    resources['skeleton'] = {str(path.relative_to(skeleton)): checkpoint.file_hash(path)
                             for path in sorted(skeleton.rglob('*')) if path.is_file()}
    stats = Path(values['MM2_NORMALIZATION_PATH'])
    resources['normalization'] = {path.name: checkpoint.file_hash(path) for path in sorted(stats.glob('*.npy'))}
    # The alias map also controls imported modules.
    resources['imports'] = checkpoint.file_hash(ROOT / 'configs/imports.json')
    return dict(config=config, source=code, resources=resources, world_size=parallel.world())


class Trainer:
    def __init__(self, config):
        validate_config(config)
        self.config = config
        if 512 % parallel.world():
            raise ValueError('GPU count must divide 512')
        self.local_batch = 512 // parallel.world()
        if self.local_batch % config['microbatch']:
            raise ValueError('Microbatch must divide each rank batch')
        # All ranks construct exactly the same random generator and PCM weights.
        checkpoint.seed_all(config['seed'])
        device = f'cuda:{int(os.environ.get("LOCAL_RANK", "0"))}'
        self.model, self.client, self.spec = create_model(
            config['model_config'], config['paths_config'], device,
            minimal_baseline=config.get('minimal_baseline', False))
        self.raw = self.model.denoiser.model
        named = [(name, parameter) for name, parameter in self.raw.named_parameters() if parameter.requires_grad]
        options = config['optimizer']
        self.optimizer = FP32MasterAdamW([parameter for _, parameter in named],
            parameter_names=[name for name, _ in named], lr=options['learning_rate'],
            betas=tuple(options['betas']), eps=options['epsilon'],
            weight_decay=options['weight_decay'])
        if self.optimizer.state:
            raise RuntimeError('Fresh Adam state must be empty')
        self.initial_hash = checkpoint.tensor_hash(self.raw.state_dict())
        if len(set(parallel.gather(self.initial_hash))) != 1:
            raise RuntimeError('Random model initialization differs across ranks')
        self.plan = data.GTPlan(config)
        self.batch_config = dict(text_dropout=config['data']['text_dropout'],
            constraint_dropout=config['data']['constraint_dropout'],
            text_roots=config['data']['text_features_paths'])
        self.completed = 0

    def update(self, diagnostic=False):
        step = self.completed + 1
        if step > self.config['updates']:
            raise ValueError('Training budget exhausted')
        torch.cuda.synchronize()
        tick = time.perf_counter()
        records = self.plan.records_at(step - 1)
        start = parallel.rank() * self.local_batch
        local = records[start:start + self.local_batch]
        pending = batching.submit_reads(self.client, local, self.batch_config, self.config['cpu_read_workers'])
        self.optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(self.config, step)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        terms_sum = torch.zeros(4, device=self.model.device, dtype=torch.float32)
        receipts, inputs = [], []
        width = self.config['microbatch']
        for offset in range(0, self.local_batch, width):
            prepared = batching.materialize(pending[offset:offset + width])
            fixed, details, summary = history.build(self.model, self.client, prepared, self.batch_config, audit=diagnostic)
            checkpoint.seed_all(local[offset]['seed'])
            if diagnostic:
                inputs.append(checkpoint.tensor_hash({key: fixed[key] for key in losses.FORWARD_KEYS}
                              | dict(target=fixed['target'], decoder_prefix=fixed['decoder_history_hybrid'])))
                receipts.append(dict(samples=details, gt_history=summary))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = self.raw(*(fixed[key] for key in losses.FORWARD_KEYS))
                with prediction_loss_context(self.raw, prediction):
                    loss, _, terms = batching.components_group(self.raw, fixed, prediction)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; optimizer update not committed')
            (loss * (width / 512)).backward()
            terms_sum += torch.stack([value.detach().float() for value in terms.values()]) * (width / self.local_batch)
            del fixed, prediction, loss, terms, prepared
        # Each rank used sample/global_batch scaling, so SUM is the global mean.
        parallel.reduce_gradients(self.raw)
        gradients = {}
        for name, parameter in self.raw.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f'Nonfinite gradient: {name}')
                if diagnostic:
                    gradients[name] = parameter.grad
        pcm = {}
        for label, marker in [('read', '.content_memory.read_gate'),
                              ('write', '.content_memory.write_gate.'),
                              ('decoder', '.content_memory.content_decoder.'),
                              ('projector', '.state_projector.')]:
            pcm[label] = math.sqrt(sum(float(parameter.grad.float().square().sum())
                for name, parameter in self.raw.named_parameters() if marker in name and parameter.grad is not None))
        gradient_hash = checkpoint.tensor_hash(gradients) if diagnostic else None
        clip = float(torch.nn.utils.clip_grad_norm_(self.raw.parameters(),
                     self.config['optimizer']['gradient_clip'], error_if_nonfinite=True))
        self.optimizer.step()
        self.completed = step
        self.client.reset_memory()
        torch.cuda.synchronize()
        row = dict(step=step, rank=parallel.rank(), world_size=parallel.world(), global_batch=512,
            local_batch=self.local_batch, sampler_cursor=step * 512, lr=lr,
            loss_terms=terms_sum.tolist(), gradient_norm=clip, pcm_gradient_norm=pcm,
            H=records[0]['H'], G=52, F=records[0]['F'], seconds=time.perf_counter() - tick,
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            record_ids=[record['id'] for record in local])
        if diagnostic:
            row.update(input_hashes=inputs, gradient_hash=gradient_hash, history=receipts,
                state_hash=checkpoint.tensor_hash([self.raw.state_dict(), self.optimizer.state_dict()]),
                rng_hash=checkpoint.tensor_hash(checkpoint.rng_state()))
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training_motion_mamba2.yaml')
    parser.add_argument('--resume', help='Trusted full-state checkpoint from this same run')
    add_model_flags(parser)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    parallel.setup()
    config = read_config(args.config)
    if not config['output_path']:
        raise ValueError('Fill output_path in the training configuration')
    output = Path(config['output_path']).expanduser().resolve()
    if args.minimal_baseline:
        config['minimal_baseline'] = True
        output = output / 'minimal_baseline'
    output.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(config)
    contracts = parallel.gather(resource_contract(config, trainer.plan) if parallel.rank() == 0 else None)
    contract = contracts[0]
    lock = None
    if parallel.rank() == 0:
        import fcntl
        lock = (output / 'training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume and (output / 'manifest.json').exists():
            raise FileExistsError('Existing run: use --resume or a new output directory')
        if args.resume and json.loads((output / 'manifest.json').read_text()) != contract:
            raise ValueError('Run manifest differs')
        checkpoint.atomic_json(output / 'manifest.json', contract)
    parallel.barrier()
    if args.resume:
        trainer.completed = checkpoint.resume(args.resume, trainer.raw, trainer.optimizer, contract, parallel)
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    saved = trainer.completed
    while trainer.completed < config['updates']:
        if any(parallel.gather(stop or (output / 'STOP_REQUEST').exists())):
            break
        row = trainer.update(diagnostic=trainer.completed == 0)
        rows = parallel.gather(row)
        if parallel.rank() == 0:
            live = dict(step=trainer.completed, seconds=max(row['seconds'] for row in rows), ranks=rows)
            with (output / 'updates.jsonl').open('a') as stream:
                stream.write(json.dumps(live, allow_nan=False) + '\n')
            checkpoint.atomic_json(output / 'live.json', live)
            print(f'UPDATE {trainer.completed}/100000 {live["seconds"]:.3f}s', flush=True)
        if trainer.completed % config['save_every'] == 0:
            checkpoint.save(output / f'checkpoint_{trainer.completed:06d}.pt', trainer.raw,
                            trainer.optimizer, trainer.completed, contract, parallel)
            saved = trainer.completed
    if trainer.completed != saved:
        checkpoint.save(output / f'checkpoint_{trainer.completed:06d}.pt', trainer.raw,
                        trainer.optimizer, trainer.completed, contract, parallel)
    if parallel.rank() == 0 and trainer.completed == config['updates']:
        filename = 'MinimalBaseline.pt' if args.minimal_baseline else 'MotionMamba2.pt'
        checkpoint.export_model(output / filename, trainer.raw)
    if parallel.rank() == 0:
        checkpoint.atomic_json(output / ('complete.json' if trainer.completed == 100000 else 'stopped.json'),
                               dict(completed=trainer.completed, sampler_cursor=trainer.completed * 512))
    parallel.barrier()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
