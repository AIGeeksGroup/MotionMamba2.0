"""Finite paired prediction/sampling comparison. No optimizer or training data writes."""
import gc
import types
from pathlib import Path
import numpy as np
import torch
from . import runtime, data, losses, sampling, evaluation
from .engine import config as load_config
from .io import read, write, identity, sha, Budget
from .schema import VERSION


def execute(config, output):
    cfg = load_config(config)
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'RESULT.json').exists():
        return read(dest / 'RESULT.json')
    origin = Path(cfg.get('transfer_source', Path(config).resolve().parent.parent))
    cfg = dict(cfg, budget_root=str(dest / 'budget'))
    if not (dest / 'budget/LIMITS.json').exists():
        write(dest / 'budget/LIMITS.json', dict(caps=dict(B=0,C=0,checks=0,teacher=0), total_updates=0, gpu_hours=2))
    specs = [s for s in read(evaluation.F / 'EVAL_CONTRACT.json')['dev_sources']
             if s['action'] in ('walk', 'wave') and s['source_index'] == 0]
    assert len(specs) == 2
    protocol = dict(
        question='Does matched prediction improve from3500to5000 while free sampling worsens; does true noCFG change that?',
        checkpoints={str(s): dict(path=str(origin / f'Fresh/train/checkpoint{s}.pt'),
                                 sha256=sha(origin / f'Fresh/train/checkpoint{s}.pt')) for s in (3500, 5000)},
        specs=specs, cells=['cold_text', 'gt_accepted_text'], timesteps=[0, 4, 9],
        same_noise_across_checkpoints=True, optimizer_updates=0, sampling_steps=10,
        reused_cfg='Existing exact fixed-seed DEV trajectories; no regeneration',
        nocfg='Single conditional forward on the existing wrapper; controller/commit lifecycle retained',
        split='DEV never admitted to training', max_prediction_forwards=48, max_new_sample_windows=16,
        limitations=['Prediction has GT-corrupted inputs; free sampling starts from noise, not a pose reconstruction task.',
                     'Dense GT-history training and accepted-cache inference are distinct documented entry modes.',
                     'Four cases and two noises diagnose transfer, not population quality or deployment.',
                     'Metrics unchanged; noCFG cannot pass solely by collapsing to stillness.'],
        decision='If matched prediction worsens, do not blame only sampling. If prediction improves but sampling worsens, inspect guidance result. No optimizer/loss grid. No deployment on this panel.')
    if (dest / 'PROTOCOL.json').exists():
        assert read(dest / 'PROTOCOL.json') == protocol
    else:
        write(dest / 'PROTOCOL.json', protocol)
    all_predictions, all_samples = [], []
    with Budget(cfg['budget_root']).timed('paired_learning_transfer', closing=True):
      for step in (3500, 5000):
        m, c, ck = runtime.load(protocol['checkpoints'][str(step)]['path'], False)
        raw = m.denoiser.model
        for spec in specs:
          for cell in protocol['cells']:
            h = 0 if cell == 'cold_text' else 104
            for ni, seed in enumerate(spec['seeds']):
                key = f'{spec["action"]}_0_{ni}_{cell}'
                target = dest / f'{step}_{key}_prediction.json'
                if target.exists():
                    all_predictions.extend(read(target))
                    continue
                r = dict(schema=VERSION, id=key, source_family_id=spec['source_family_id'],
                    dataset=spec['dataset'], split='DEV', feature_path=spec['feature_path'],
                    crop_start=spec['crop_start'] + (104 if h == 0 else 0),
                    crop_end=spec['crop_start'] + 156, caption=spec['evaluation_text'],
                    supervision='GT', H=h, F=0, G=52, fps=25, skeleton='g1',
                    representation='physical414', condition='none', out_of_horizon=False, seed=seed)
                evalcfg = dict(cfg, text_dropout=0., constraint_dropout=0.)
                with torch.no_grad():
                    f, receipt = data.build_gt(m, c, r, evalcfg)
                    gt_metrics, _ = sampling.measure(m, f['supervised_motion'])
                    results = []
                    for t in protocol['timesteps']:
                        f['timestep'] = torch.tensor([t], device=m.device)
                        f['x'] = torch.zeros_like(f['target'])
                        f['x'][f['history_token_mask']] = f['target'][f['history_token_mask']]
                        f['x'][f['generation_token_mask']] = m.diffusion.q_sample(
                            f['target'], f['timestep'], f['noise'])[f['generation_token_mask']]
                        raw.train(); m.autoencoder.eval()
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            value, info, _ = losses.compute(raw, c, f, cfg['loss'], return_graph=True)
                        prediction = info.pop('prediction')
                        decoded = losses.decode_G(prediction.float(), f)
                        metric, _ = sampling.measure(m, decoded)
                        ph = m.motion_rep.unnormalize(decoded)[0].cpu().numpy()
                        artifact = dest / f'{step}_{key}_t{t}.npz'
                        np.savez_compressed(artifact, prediction=prediction.float().cpu().numpy(), physical=ph)
                        results.append(dict(step=step, id=key, action=spec['action'], cell=cell,
                            seed=seed, timestep=t, input_hash=identity([f['x'],f['text'],f['observed'],f['motion_mask']]),
                            target_hash=identity(f['target']), noise_hash=identity(f['noise']),
                            total=float(value), loss=info, metrics=metric, GT_metrics=gt_metrics, artifact=str(artifact)))
                    write(target, results); all_predictions.extend(results)
                print('PREDICTION', step, key, flush=True)
        raw.eval()
        # Reuse the existing CFG output and generate only the missing single-conditional arm.
        from ardy.model.cfg import AutoLatentClassifierFreeGuidedModelNoCFG
        original_forward = m.denoiser.forward
        m.denoiser.forward = types.MethodType(AutoLatentClassifierFreeGuidedModelNoCFG.forward, m.denoiser)
        probe = sampling.NoiseProbe(m)
        try:
          with torch.inference_mode():
            for spec in specs:
              for cell in protocol['cells']:
                for ni, seed in enumerate(spec['seeds']):
                    key = f'{spec["action"]}_0_{ni}_{cell}'
                    oldpath = origin / f'Fresh/eval{step}/foundation/{key}.json'
                    old = read(oldpath)
                    assert old['seed'] == seed and old['prompt'] == spec['evaluation_text']
                    all_samples.append(dict(step=step, id=key, mode='cfg2_reused', source=str(oldpath), **{
                        k:old[k] for k in ('metrics','semantic','function_pass','geometry_pass','joint_pass','noise','artifact')}))
                    target = dest / f'{step}_{key}_nocfg.json'
                    if target.exists():
                        all_samples.append(read(target)); continue
                    text = evaluation.text_for(cfg, spec['evaluation_text'], m.device)
                    a = np.load(spec['feature_path'], mmap_mode='r')[spec['crop_start']:spec['crop_start']+156].copy()
                    physical = torch.tensor(a, device=m.device)[None]
                    physical[:,:,[0,2]] -= physical[:,:1,[0,2]].clone()
                    if cell == 'cold_text':
                        runner = sampling.Native(m,c,probe,text)
                    else:
                        runner,_ = sampling.prime_gt(m,c,probe,physical[:,:104],text)
                    previous = runner.physical
                    noise = sampling.fresh_noise(seed,1)[0]
                    assert identity(noise) == old['noise']
                    raw_batches=[]
                    handle=raw.register_forward_pre_hook(lambda module,args:raw_batches.append(int(args[0].shape[0])))
                    measured,q,dec,_,_ = runner.step(noise,{})
                    handle.remove()
                    assert raw_batches and set(raw_batches)=={1}, raw_batches
                    limits=np.asarray(read(ck['config']['joint_limit_spec'])['bounds'])
                    metrics,_,_,_=evaluation.summarize(m,dec,[measured],spec['action'],limits,
                        m.motion_rep.normalize(previous[:,-52:]) if previous is not None else None)
                    ph=m.motion_rep.unnormalize(dec)[0].cpu().numpy()
                    sem=evaluation.semantic_features(q,spec['action'],ph)
                    artifact=dest/f'{step}_{key}_nocfg.npz'
                    np.savez_compressed(artifact,qpos=q,physical=ph,fps=25)
                    result=dict(step=step,id=key,mode='nocfg',metrics=metrics,semantic=sem,
                        function_pass=sem['passed'],geometry_pass=not metrics['geometry_failures'],
                        joint_pass=bool(sem['passed'] and not metrics['geometry_failures']),noise=identity(noise),
                        raw_forward_batches=raw_batches,artifact=str(artifact))
                    write(target,result);all_samples.append(result)
                    print('NO_CFG',step,key,sem['passed'],metrics['whole']['contact_slip_percent'],flush=True)
        finally:
            probe.close();m.denoiser.forward=original_forward
        del m,c,ck,raw,probe
        gc.collect();torch.cuda.empty_cache()
    paired=[]
    for a in [r for r in all_predictions if r['step']==3500]:
        b=next(r for r in all_predictions if r['step']==5000 and r['id']==a['id'] and r['timestep']==a['timestep'])
        assert all(a[k]==b[k] for k in ('input_hash','target_hash','noise_hash'))
        paired.append(dict(id=a['id'],timestep=a['timestep'],loss3500=a['total'],loss5000=b['total'],
            relative_loss_change=b['total']/a['total']-1,
            contact_speed3500=a['metrics']['contact_speed_m_s'],contact_speed5000=b['metrics']['contact_speed_m_s']))
    summary={}
    for step in (3500,5000):
        for mode in ('cfg2_reused','nocfg'):
            rr=[r for r in all_samples if r['step']==step and r['mode']==mode]
            summary[f'{step}_{mode}']=dict(n=len(rr),function=sum(r['function_pass'] for r in rr),
                joint=sum(r['joint_pass'] for r in rr),
                median_contact_speed=float(np.median([r['metrics']['whole']['contact_speed_m_s'] for r in rr])),
                median_contact_slip=float(np.median([r['metrics']['whole']['contact_slip_percent'] for r in rr])))
    result=dict(status='DONE',optimizer_updates=0,predictions=all_predictions,sampling=all_samples,
        paired=paired,paired_loss_improved=sum(r['relative_loss_change']<0 for r in paired),summary=summary,
        production_changed=False,limitations=protocol['limitations'])
    write(dest/'RESULT.json',result)
    print('TRANSFER_DONE',result['paired_loss_improved'],len(paired),summary,flush=True)
    return result
