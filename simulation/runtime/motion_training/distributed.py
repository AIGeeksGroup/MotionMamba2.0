"""Explicit data parallelism for the shared microbatch engine.

Global batch stays ten, including on four/eight ranks. Unequal local sample
counts are summed, never averaged by rank. Old single-rank numerics stay intact.
"""
import contextlib,os
import torch
import torch.distributed as dist

def setup():
    if int(os.environ.get('WORLD_SIZE','1'))>1 and not dist.is_initialized():
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl',device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
def world():return dist.get_world_size() if dist.is_initialized() else 1
def rank():return dist.get_rank() if dist.is_initialized() else 0
def leader():return rank()==0
def barrier():
    if world()>1:dist.barrier()
def gather(value):
    if world()==1:return [value]
    result=[None]*world();dist.all_gather_object(result,value);return result
def timed(budget,name,closing=False):
    return budget.timed(name,closing=closing,devices=world()) if leader() else contextlib.nullcontext()
def reduce_gradients(raw):
    if world()==1:return
    params=list(raw.parameters());device=params[0].device
    exists=torch.tensor([p.grad is not None for p in params],device=device,dtype=torch.int32)
    dist.all_reduce(exists,op=dist.ReduceOp.SUM)
    selected=[p for p,n in zip(params,exists.tolist()) if n]
    # Bounded buckets avoid allocating a second whole-model FP32 gradient.
    bucket=[];size=0
    def flush(ps):
        flat=torch.cat([(p.grad.detach().float() if p.grad is not None else torch.zeros_like(p,dtype=torch.float32)).flatten() for p in ps])
        dist.all_reduce(flat,op=dist.ReduceOp.SUM);offset=0
        for p in ps:
            p.grad=flat[offset:offset+p.numel()].reshape_as(p).to(p.dtype).clone();offset+=p.numel()
    for p in selected:
        bucket.append(p);size+=p.numel()
        if size>=4_000_000:flush(bucket);bucket=[];size=0
    if bucket:flush(bucket)
