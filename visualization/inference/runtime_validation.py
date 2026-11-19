"""GPU acceptance: offline parity, isolated sessions, rewind and retry semantics."""
import json
from pathlib import Path
import numpy as np
import torch
from motion_mamba2.model.decoder_history import generate_with_committed_decode

def generate(model, client, motion, text, caption, seed):
    client.reset_memory()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    length=motion.shape[1]
    return generate_with_committed_decode(
        model, [caption], length, decoder_prefix=None, num_denoising_steps=10,
        cfg_weight=(2.0, 2.0), text_feat=text,
        text_pad_mask=torch.ones(text.shape[:2], device=motion.device, dtype=torch.bool),
        pad_mask=torch.ones(1,length,device=motion.device,dtype=torch.bool),
        motion_mask=torch.zeros_like(motion,dtype=torch.bool),
        observed_motion=torch.zeros_like(motion),
        first_heading_angle=torch.zeros(1,device=motion.device),
        crop_history_length=0,progress_bar=lambda values:values)


def validate(runtime, output):
    from bootstrap import warm_embedding
    text=warm_embedding('cuda')
    row={'caption':'A person continuously walks straight ahead with a natural gait.'}
    mask=torch.ones(1,1,dtype=torch.bool,device='cuda')
    m,c=runtime.model,runtime.client
    with torch.inference_mode():
        motion=torch.zeros(1,104,414,device='cuda')
        pred=generate(m,c,motion,text,row['caption'],20260917)
        offline=np.asarray(runtime.converter.dict_to_qpos(m.motion_rep.inverse(pred,is_normalized=True),device='cuda',numpy=True)).reshape(-1,36)
    a=runtime.new_session(20260917,2)
    def step(s,prompt='same',feature=text):
        return runtime.step(s,feature,mask,prompt,s.windows,s.revision)
    a0=step(a)
    b=runtime.new_session(97,2);step(b)
    a1=step(a)
    online=np.concatenate([a0['qpos'],a1['qpos']])
    error=float(np.max(np.abs(online-offline)))
    print('PARITY_PER_WINDOW',[float(np.max(np.abs(online[i:i+52]-offline[i:i+52]))) for i in [0,52]],flush=True)
    np.savez_compressed(Path(output).with_suffix('.npz'),offline=offline,online=online)
    # Compare interleaved versus isolated execution independently of parity.
    solo=runtime.new_session(20260917,2)
    solo0=step(solo);solo1=step(solo)
    isolation=float(np.max(np.abs(np.asarray(a1['qpos'])-np.asarray(solo1['qpos']))))
    runtime.rewind(a,1,a.revision)
    replay=step(a)
    rewind=float(np.max(np.abs(np.asarray(replay['qpos'])-np.asarray(a1['qpos']))))
    again=runtime.step(a,text,mask,'same',1,a.revision)
    assert again is a.last_result and a.windows==2
    runtime.publish(a,2,a.revision)
    try:runtime.rewind(a,1,a.revision)
    except ValueError:published_protected=True
    else:published_protected=False
    assert isolation<1e-6 and rewind<1e-6 and published_protected
    # parity tolerance allows fp32 recentering/rounding but not trajectory drift.
    assert error<1e-3, f'offline/incremental qpos max error {error}'
    # Cross the W4 FIFO boundary; long-term state must survive KV eviction.
    with torch.inference_mode():
        motion6=torch.zeros(1,312,414,device='cuda')
        pred6=generate(m,c,motion6,text,row['caption'],20260917)
        offline6=np.asarray(runtime.converter.dict_to_qpos(m.motion_rep.inverse(pred6,is_normalized=True),device='cuda',numpy=True)).reshape(-1,36)
    six=runtime.new_session(20260917,6)
    chunks=[]
    for _ in range(6):
        value=step(six);chunks.append(value['qpos'])
        runtime.publish(six,six.windows,six.revision)
    long_error=float(np.max(np.abs(np.concatenate(chunks)-offline6)))
    assert long_error<1e-3, f'W4 eviction parity error {long_error}'
    assert value['cache_chunk_ids']==[2,3,4,5]
    result=dict(status='PASS',offline_parity_max_abs=error,
                interleaved_session_max_abs=isolation,rewind_replay_max_abs=rewind,
                retry_does_not_double_commit=True,published_rewind_rejected=published_protected,
                six_window_parity_max_abs=long_error,w6_cache_chunk_ids=value['cache_chunk_ids'],
                note='Same cached text, seed and 10 denoising steps; 104-frame offline vs 2 incremental windows.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print('ONLINE_RUNTIME_VALIDATED',result,flush=True)
