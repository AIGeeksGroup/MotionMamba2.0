"""Executable contracts with actual codec, denoiser and inherited optimizer."""
import gc,math,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from . import data,runtime,losses,engine
from .io import Budget,read,write,identity,copy_state,sha

def contracts(config,output):
    cfg=engine.config(config);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    plan=data.prepare(cfg,Path(cfg['budget_root'])/'B/data');results=[]
    with Budget(cfg['budget_root']).timed('contracts'):
        m,c,ck=runtime.load(cfg['parent'],True);opt=runtime.optimizer(m,ck);raw=m.denoiser.model
        for h,future in data.BUCKETS:
            r=next(x for x in plan['plan'] if x['H']==h and x['F']==future and (not future or x['out_of_horizon']))
            r=dict(r,condition='root_waypoint');nc=dict(cfg,text_dropout=0.,constraint_dropout=0.)
            batch,rec=data.build_gt(m,c,r,nc);raw.train();raw.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,detail,terms=losses.compute(raw,c,batch,cfg['loss'],True)
            assert torch.isfinite(loss) and loss.requires_grad
            grad=torch.autograd.grad(loss,detail['prediction'],retain_graph=True)[0]
            assert torch.isfinite(grad).all() and grad.abs().sum()>0
            loss.backward();assert all(p.grad is None for p in m.autoencoder.parameters())
            assert all(p.dtype==torch.float32 for _,p in runtime.master_named(raw,opt))
            results.append(dict(H=h,F=future,loss=float(loss),prediction_gradient=float(grad.norm()),G_tokens=int(batch['generation_token_mask'].sum()),F_latent_zero=bool((batch['x'][batch['future_token_mask']]==0).all()),receipt=rec,terms={k:v for k,v in detail.items() if k!='prediction'}))
            assert results[-1]['G_tokens']==13 and results[-1]['F_latent_zero']
            if future:
                physical=np.load(r['feature_path'],mmap_mode='r')[r['crop_start']:r['crop_end']].copy();physical[h+52:,5:]+=10
                unchanged,_=data.build_gt(m,c,r,nc,physical_override=physical,observed_override=(batch['observed'],batch['motion_mask']))
                for key in ['x','target','supervised_motion','text','observed','motion_mask']:assert torch.equal(batch[key],unchanged[key]),('future_leak',key)
                changed=batch['observed'].clone();changed[:,h+52:]+=batch['motion_mask'][:,h+52:]*.1
                changed_batch,_=data.build_gt(m,c,r,nc,observed_override=(changed,batch['motion_mask']))
                assert not torch.equal(changed_batch['observed'],batch['observed'])
                seen=[]
                hook=raw.register_forward_pre_hook(lambda module,args:seen.append(args[-1].detach().clone()))
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):losses.compute(raw,c,changed_batch,cfg['loss'])
                hook.remove();assert torch.equal(seen[-1],changed)
                results[-1]['unobserved_future_invariant']=True;results[-1]['visible_future_reaches_raw_network']=True
            del loss,detail,terms,batch;raw.zero_grad(set_to_none=True)
        # Real reduction invariant, including empty samples and padding.
        x=torch.tensor([[[2.,4.]],[[9.,9.]]],device=m.device,requires_grad=True);mask=torch.tensor([[[1,1]],[[0,0]]],device=m.device,dtype=torch.bool)
        value,den=losses.mean_valid(x,mask);assert value==1.5
        padded=torch.cat([x,torch.full_like(x,1000)],1);pm=torch.cat([mask,torch.zeros_like(mask)],1)
        assert losses.mean_valid(padded,pm)[0]==value
        assert sum(losses.mean_valid(x[i:i+1],mask[i:i+1])[0]/2 for i in range(2))==value
        report=dict(passed=True,buckets=results,empty_mask_padding_microbatch=True,optimizer_exact_inherited=True,codec_frozen_with_input_gradient=True,imports=runtime.import_sources(),optimizer_updates=0,root_condition_detach_configuration=ck['config'].get('root_body_gradient_scale'))
        write(dest/'CONTRACTS.json',report);print('CONTRACTS_PASS',flush=True)

def migration(config,output):
    cfg=engine.config(config);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    protocol=dict(absolute_tolerance=0.,relative_tolerance=0.,policy='Exact loss, parameters, FP32 master, optimizer and RNG. Failure requires diagnosis, not tolerance inflation.',max_updates=6)
    if not (dest/'PROTOCOL.json').exists():write(dest/'PROTOCOL.json',protocol)
    launcher=Path(__file__).resolve().parents[2]/'examples/train_motion.sh'
    for mode,steps,resume in [('reference',1,None),('migrated',1,None),('continuous',2,None),('split',1,None),('split',2,str(dest/'split/checkpoint1.pt'))]:
        out=dest/mode;target=out/f'checkpoint{steps}.pt'
        if target.exists():continue
        cmd=['bash',str(launcher),'verify-worker','--config',str(config),'--output',str(out),'--mode',mode,'--steps',str(steps)]
        if resume:cmd+=['--resume',resume]
        with (dest/f'{mode}_{steps}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    checks=[]
    for a,b,n in [('reference','migrated',1),('continuous','split',2)]:
        ca=torch.load(dest/a/f'checkpoint{n}.pt',map_location='cpu',weights_only=False,mmap=True);cb=torch.load(dest/b/f'checkpoint{n}.pt',map_location='cpu',weights_only=False,mmap=True)
        matching={k:identity(ca[k])==identity(cb[k]) for k in ['root_model','body_model','denoiser_aux','optimizer','rng_by_rank','step']}
        ra=read(dest/a/f'update{n}.json');rb=read(dest/b/f'update{n}.json')
        matching.update(loss=ra['loss']==rb['loss'],clip=ra['clip']==rb['clip'],master_delta=ra['master_delta']==rb['master_delta'],sample_order=[r['id'] for r in ra['receipts']]==[r['id'] for r in rb['receipts']])
        checks.append(dict(a=a,b=b,checks=matching,passed=all(matching.values())))
        del ca,cb;gc.collect()
    write(dest/'RESULT.json',dict(passed=all(x['passed'] for x in checks),checks=checks,protocol=protocol))
    assert all(x['passed'] for x in checks),checks

def verify_worker(config,output,mode,steps,resume=None,pool=None):
    from . import distributed as parallel
    cfg=engine.config(config);dest=Path(output);dest.mkdir(parents=True,exist_ok=True);budget=Budget(cfg['budget_root']);start=0
    with parallel.timed(budget,'verify:'+mode):
        m,c,ck=runtime.load(resume or cfg['parent'],True);opt=runtime.optimizer(m,ck);runtime.restore_rng(ck['rng_by_rank'][0]);raw=m.denoiser.model
        manifest_root=Path(cfg['budget_root'])/'B/data'
        if parallel.leader():data.prepare(cfg,manifest_root)
        parallel.barrier();plan=read(manifest_root/'MANIFEST.json');r=next(x for x in plan['plan'] if x['H']==52 and x['F']==0 and x['condition']=='root_trajectory')
        if mode in ['reference','migrated']:
            from ._compat import legacy_loss
            r=dict(r,H=52,F=0);ph=torch.tensor(np.load(r['feature_path'])[r['crop_start']:r['crop_start']+104].copy(),device=m.device)[None];ph[:,:,[0,2]]-=ph[:,:1,[0,2]].clone()
            normalized=m.motion_rep.normalize(ph);mask=torch.zeros_like(normalized);mask[:,52:,:5]=1
            batch=dict(motion=normalized,observed_motion=normalized*mask,motion_mask=mask,history_frames=torch.tensor([52],device=m.device),text_feature=data.embedding(r['caption'],cfg['text_roots'],m.device)[:,0],text=[r['caption']],length=torch.tensor([104],device=m.device))
            m.diffusion.calc_diffusion_vars(torch.arange(m.diffusion.num_base_steps,device=m.device))
            names=['boundary_loss_weight','root_velocity_loss_weight','constraint_loss_weight','fk_constraint_loss_weight','foot_skate_loss_weight','decoded_body_loss_weight','kinematic_consistency_loss_weight','joint_limit_loss_weight','joint_limit_spec','accepted_boundary_loss_weight']
            opts={k:ck['config'][k] for k in names if k in ck['config']}
            fixed=legacy_loss._build_batch(m,batch,m.device,True,**opts);fixed.pop('hybrid_converter');torch.save(copy_state(fixed),dest/'fixture.pt')
            cfg=engine.config(Path(config).parent/'migrated_legacy.yaml')
            if mode=='reference':
                # Reference-only validation loads original function AST without executing
                # the historical module's imports/CLI/global-output side effects.
                import ast
                source=Path('/data3/zts/result/g1_loss_alignment_execution_20261003/paired_loss.py');tree=ast.parse(source.read_text())
                funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name not in ['main','parse_args','_evaluate']]
                namespace=dict(vars(legacy_loss));exec(compile(ast.Module(body=funcs,type_ignores=[]),str(source),'exec'),namespace)
                def reference_compute(raw,client,f,version,**unused):
                    out=namespace['_loss'](raw,client,f,True)
                    return out[0],dict(values={str(i):float(v.detach()) for i,v in enumerate(out)},old_extensions_enabled=True),{}
                losses.compute=reference_compute
            records=[dict(supervision='legacy_fixture',id=f'fixture:{j}',seed=202610031001+j,path=str(dest/'fixture.pt')) for j in range(10)]
        else:records=None
        if mode in ['dispatch_reference','dispatch_optimized']:
            from .acceleration import configure
            configure(c,'optimized' if mode=='dispatch_optimized' else 'reference')
        if mode=='history':
            pooldata=read(pool or '/data3/zts/result/g1_master_experiment_roadmap_20261003/R2_pool_g1_v2/POOL.json')
            labels=pooldata['labels'];assert all(r['qualified'] for r in labels)
            if pool:
                assert pooldata['checkpoint_sha256']==sha(resume or cfg['parent']), 'Refreshed labels must match the updated student'
                chosen=[]
                for r in labels:
                    if r['stage_key'] not in {x['stage_key'] for x in chosen}:chosen.append(r)
                    if len(chosen)==2:break
                assert len(chosen)==2,'Two qualified stages required for this bounded check'
            else:chosen=[next(r for r in labels if r['group']=='S'),next(r for r in labels if r['group']=='T')]
            records=plan['plan'][:8]+[dict(r,seed=202610038000+i) for i,r in enumerate(chosen)]
        if resume:
            assert ck['framework']['recipe_hash']==engine.recipe_hash(cfg)
            assert ck['framework'].get('world_size',1)==parallel.world()
            start=ck['framework']['stage_step'];runtime.restore_rng(ck['rng_by_rank'][parallel.rank()])
        ck['_framework_global_start']=ck.get('step',16600)-start
        manifest=dest/'MANIFEST.json'
        if parallel.leader():write(manifest,dict(test_mode='legacy' if records else 'newGT',plan_hash=identity(plan),implementation_version='framework_v1'))
        parallel.barrier()
        for step in range(start+1,steps+1):
            sample=records if records else plan['plan'][(step-1)*10:step*10]
            result=engine.update(m,c,ck,opt,cfg,sample,step,'checks',budget)
            if parallel.leader():write(dest/f'update{step}.json',result)
        engine.save(dest/f'checkpoint{steps}.pt',m,ck,opt,cfg,steps,manifest,pool)
        parallel.barrier()

def gradient_summary(config,output):
    cfg=engine.config(config);dest=Path(output);checks={}
    a=torch.load(dest/'dispatch_reference/checkpoint1.pt',map_location='cpu',weights_only=False,mmap=True);b=torch.load(dest/'dispatch_optimized/checkpoint1.pt',map_location='cpu',weights_only=False,mmap=True)
    for key in ['root_model','body_model','denoiser_aux','optimizer','rng_by_rank']:checks[key]=identity(a[key])==identity(b[key])
    for key in ['loss','clip','master_delta','gradient_norms']:
        checks[key]=read(dest/'dispatch_reference/update1.json')[key]==read(dest/'dispatch_optimized/update1.json')[key]
    del a,b;gc.collect()
    history=[read(dest/'history'/f'update{i}.json') for i in [1,2]]
    teachers=[[r for r in step['receipts'] if r['supervision']=='same_state_teacher'] for step in history]
    out=dict(kernel_backward_switch_exact=all(checks.values()),checks=checks,history_actual_updates=2,history_label_exposures=sum(map(len,teachers)),history_loss_before=history[0]['loss'],history_loss_next_update=history[1]['loss'],history_teacher_components=[[r['loss']['values'] for r in rs] for rs in teachers],history_real_master_deltas=[h['master_delta'] for h in history],scope='finite mixture of8GT+2 qualified same-state teacher; two repeated-batch updates; not complete stage coverage, not autonomous improvement, not two collector generations')
    write(dest/'RESULT.json',out);assert out['kernel_backward_switch_exact'];print('GRADIENT_SWITCH_EXACT',flush=True)

def distributed_summary(config,output):
    cfg=engine.config(config);dest=Path(output);protocol=read(dest/'PROTOCOL.json')
    cp=dest/'four_gpu/checkpoint1.pt';reference=dest.parent/'gradient_checks/dispatch_reference/checkpoint1.pt'
    parent=torch.load(cfg['parent'],map_location='cpu',weights_only=False,mmap=True)
    a=torch.load(reference,map_location='cpu',weights_only=False,mmap=True);b=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
    ra=read(reference.parent/'update1.json');rb=read(cp.parent/'update1.json')
    sample_keys=['id','input_hash','target_hash','noise_hash']
    same=all(all(x[k]==y[k] for k in sample_keys) for x,y in zip(ra['receipts'],rb['receipts'])) and len(ra['receipts'])==len(rb['receipts'])==10
    dot=aa=bb=err=0.
    def optimizer_values(c):
        o=c['optimizer'];masters=o['master_parameters'];values={}
        models={**{'root_model.'+k:v for k,v in c['root_model'].items()},**{'body_model.'+k:v for k,v in c['body_model'].items()},**c['denoiser_aux']}
        for i,s in enumerate(o['compute_signature']):
            if str(i) in masters:values[str(i)]=masters[str(i)]
            else:values[str(i)]=models[s['name'].removeprefix('module.')]
        return values
    pvals=optimizer_values(parent);avals=optimizer_values(a);bvals=optimizer_values(b)
    for k,p in pvals.items():
        x=(avals[k].float()-p.float()).flatten().double();y=(bvals[k].float()-p.float()).flatten().double()
        dot+=float(x@y);aa+=float(x@x);bb+=float(y@y);err+=float(((x-y)**2).sum())
    cosine=dot/math.sqrt(aa*bb);rel=math.sqrt(err/max(aa,1e-30));lossdiff=abs(ra['loss']-rb['loss'])
    result=dict(samples_exact=same,global_batch=10,ranks=4,loss_absolute_difference=lossdiff,master_update_cosine=cosine,master_update_relative_l2=rel,wall_seconds_four=rb['seconds'],wall_seconds_reference=ra['seconds'],phase_seconds_by_rank=rb['phase_seconds_by_rank'],first_step_not_steady_throughput=True,protocol=protocol)
    result['passed']=same and lossdiff<=protocol['loss_absolute_tolerance'] and cosine>=protocol['master_update_cosine_min'] and rel<=protocol['master_update_relative_l2_max']
    result['rank_equality_checkpoint2']='pending'
    second=dest/'four_gpu/checkpoint2.pt'
    if second.exists():
        c=torch.load(second,map_location='cpu',weights_only=False,mmap=True);hs=c['framework']['rank_state_hashes']
        result['rank_equality_checkpoint2']=bool(len(hs)==4 and len(set(hs))==1);result['resume_step2']=c['framework']['stage_step']==2 and len(c['rng_by_rank'])==4
        result['passed']=result['passed'] and result['rank_equality_checkpoint2'] and result['resume_step2']
        continuous=dest/'four_continuous/checkpoint2.pt'
        if continuous.exists():
            d=torch.load(continuous,map_location='cpu',weights_only=False,mmap=True)
            keys=['root_model','body_model','denoiser_aux','optimizer','rng_by_rank','step']
            result['continuous_vs_resumed_exact']={key:identity(c[key])==identity(d[key]) for key in keys}
            result['passed']=result['passed'] and all(result['continuous_vs_resumed_exact'].values())
    write(dest/'RESULT.json',result);print('DISTRIBUTED',result,flush=True);return result

def graph_contract(config,output):
    cfg=engine.config(config);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'RESULT.json').exists():return read(dest/'RESULT.json')
    with Budget(cfg['budget_root']).timed('targeted_root_decoder_graph'):
        m,c,ck=runtime.load(cfg['parent'],True);raw=m.denoiser.model;plan=read(Path(cfg['budget_root'])/'B/data/MANIFEST.json')
        r=next(r for r in plan['plan'] if r['H']==52 and r['F']==0 and r['condition']=='root_trajectory')
        f,receipt=data.build_gt(m,c,r,dict(cfg,text_dropout=0.,constraint_dropout=0.));raw.train();roots=[]
        hook=raw.root_model.register_forward_hook(lambda module,args,out:roots.append(out))
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,detail,terms=losses.compute(raw,c,f,cfg['loss'],True)
        hook.remove();root=roots[-1];prediction=detail['prediction']
        body_to_root=torch.autograd.grad(prediction[:,:,20:].float().square().mean(),root,allow_unused=True,retain_graph=True)[0]
        rootnorm=0. if body_to_root is None else float(body_to_root.norm())
        assert rootnorm==0.,rootnorm
        goal_to_root=torch.autograd.grad(terms['goal'],root,retain_graph=True)[0]
        decoded_to_prediction=torch.autograd.grad(terms['decoded_body'],prediction,retain_graph=True)[0]
        assert goal_to_root.norm()>0 and decoded_to_prediction[:,:,20:].norm()>0
        loss.backward();assert all(p.grad is None and not p.requires_grad for p in m.autoencoder.parameters())
        result=dict(passed=True,body_prediction_to_root_output_norm=rootnorm,decoded_goal_to_root_output_norm=float(goal_to_root.norm()),decoded_body_to_body_latent_norm=float(decoded_to_prediction[:,:,20:].norm()),root_output_dtype=str(root.dtype),codec_frozen=True,codec_inputs_differentiable=True,optimizer_updates=0,receipt=receipt,scope='actual root output and decoder paths; nonempty same-source root goals; no claim all body/FK terms must have nonzero root gradients')
        write(dest/'RESULT.json',result);print('GRAPH_CONTRACT', {k:v for k,v in result.items() if k!='receipt'},flush=True);return result

def paired_check(config,output):
    """Same exposed samples/noise before versus after an actual stopped run.

    Train-mode prediction/codec losses only; no optimizer updates. This cannot
    substitute for autonomous quality or justify overruling a stop gate.
    """
    from .io import rows
    import numpy as np
    cfg=engine.config(config);dest=Path(output);protocol=read(dest/'PROTOCOL.json');run=Path(protocol['run'])
    if (dest/'RESULT.json').exists():return read(dest/'RESULT.json')
    receipts=[r for step in rows(run/'train/train.jsonl') for r in step['receipts']]
    assert len(receipts)==80
    checkpoints={'before':run/'train/checkpoint0.pt','after':run/'train/checkpoint8.pt'};results={}
    with Budget(cfg['budget_root']).timed('paired_learning_after_stop',closing=True):
      for name,cp in checkpoints.items():
        m,c,ck=runtime.load(cp,False);raw=m.denoiser.model;out=[]
        with torch.no_grad():
          for r in receipts:
            f,receipt=engine.build(m,c,ck,r,cfg);raw.train();m.autoencoder.eval();torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
            with torch.autocast('cuda',dtype=torch.bfloat16):value,info,_=losses.compute(raw,c,f,cfg['loss'])
            for k in ['input_hash','target_hash','noise_hash']:assert receipt[k]==r[k],(r['id'],k)
            out.append(dict(id=r['id'],source=r['supervision'],stage=r.get('stage_key'),values=info['values'],total=float(value),input_hash=receipt['input_hash'],target_hash=receipt['target_hash']))
        results[name]=out;write(dest/(name+'.json'),out);del m,c,ck,raw;gc.collect();torch.cuda.empty_cache()
    summary={}
    for source in ['GT','same_state_teacher']:
        indices=[i for i,r in enumerate(receipts) if r['supervision']==source];a=[results['before'][i] for i in indices];b=[results['after'][i] for i in indices]
        summary[source]=dict(n=len(indices),total_before=float(np.mean([r['total'] for r in a])),total_after=float(np.mean([r['total'] for r in b])),samples_lower=sum(y['total']<x['total'] for x,y in zip(a,b)),components={k:dict(before=float(np.mean([r['values'][k] for r in a])),after=float(np.mean([r['values'][k] for r in b]))) for k in a[0]['values']})
    result=dict(summary=summary,optimizer_updates=0,sample_count=80,inputs_targets_noise_exact=True,scope='previously exposed TRAIN paired loss; not heldout loss or autonomous improvement; STOP unchanged');write(dest/'RESULT.json',result);print('PAIRED_CHECK',summary,flush=True);return result
