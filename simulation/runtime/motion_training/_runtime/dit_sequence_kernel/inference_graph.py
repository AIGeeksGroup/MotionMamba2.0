"""Bounded read-only stack graphs with explicit input and state updates.

Commit/restore/RNG remain outside capture. Unsupported masks use eager.
Graph scratch is held on the module, never in serializable session state.
"""
import copy,time
import torch

def tensors(value):
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,dict):
        for v in value.values():yield from tensors(v)
    elif isinstance(value,(list,tuple)):
        for v in value:yield from tensors(v)
    elif hasattr(value,'__dict__'):
        yield from tensors(vars(value))


def eager(stack, x, condition, kw):
    """Capture unchanged serial CFG branches in one graph. Never batch GEMMs."""
    groups=kw.get('_serial_cfg_groups',1)
    if groups==1:
        clean={k:v for k,v in kw.items() if k!='_serial_cfg_groups'}
        return stack._forward_eager(x,condition,**clean)
    assert x.shape[0] % groups == 0 and not kw.get('commit',False)
    assert not kw.get('compute_pending',True) and not kw.get('collect_diagnostics',True)
    batch=x.shape[0]//groups
    outputs=[]
    from contextlib import nullcontext
    scope=shared_kv_scope(stack) if getattr(stack,'inference_graph_shared_kv',False) else nullcontext()
    with scope:
        for start in range(0,x.shape[0],batch):
            part={k:v for k,v in kw.items() if k!='_serial_cfg_groups'}
            for key in ('current_mask','attention_mask','content_memory_condition'):
                if part.get(key) is not None:part[key]=part[key][start:start+batch]
            outputs.append(stack._forward_eager(x[start:start+batch],condition[start:start+batch],**part)[0])
    return torch.cat(outputs,dim=0),None,{'layers':[]}


from contextlib import contextmanager

@contextmanager
def shared_kv_scope(stack):
    """Compute shared sink/history once *inside every replay*, not just capture.

    Python caches are reset for each warmup/capture. The first branch's
    projection/cat nodes stay in the graph and update buffers every replay;
    later serial branches consume those same buffers. Durable state is untouched.
    """
    old=[]
    scope=object()
    names=('inference_cache','_inference_read_scope','_inference_sink_cache','_inference_history_cache')
    missing=object()
    try:
        for layer in stack.layers:
            att=layer.mixer.attention
            old.append((att,tuple(getattr(att,n,missing) for n in names)))
            att.inference_cache=True
            att._inference_read_scope=scope
            att._inference_sink_cache=None
            att._inference_history_cache=None
        yield
    finally:
        for att,values in old:
            for n,v in zip(names,values):
                if v is missing:delattr(att,n)
                else:setattr(att,n,v)

class ReadOnlyStackGraphs:
    def __init__(self,limit=8):
        self.entries={};self.limit=limit;self.events=[];self.replays=0;self.fallbacks=0
        self.validated_state=None;self.validated_version=None;self.state_is_dense=False

    def call(self,stack,x,condition,kw):
        # An explicit all-true attention mask has the same original SDPA path
        # as None. Validate on the boundary; never conflate it with recurrent masks.
        mask=kw.get('attention_mask')
        if mask is not None and (mask.shape!=x.shape[:2] or mask.dtype!=torch.bool or mask.device!=x.device):
            self.fallbacks+=1
            return eager(stack,x,condition,kw)
        if mask is not None and bool(mask.all()):
            kw=dict(kw,attention_mask=None)
        state=kw['state']
        if (not x.is_cuda or stack.training or torch.is_grad_enabled() or kw.get('commit',False)
            or kw.get('compute_pending',True) or kw.get('collect_diagnostics',True)
            or kw.get('current_mask') is not None or kw.get('attention_mask') is not None
            or kw.get('content_memory_read_view') is not None
            or any(l.mixer.record_memory_read_debug or l._forward_hooks or l._forward_pre_hooks for l in stack.layers)):
            return eager(stack,x,condition,kw)
        source_version=tuple((s.commit_version,s.attention_cache.chunk_ids) for s in state.layers)
        # Once per restored committed state, outside capture; no input checks deleted.
        if self.validated_state is not state or self.validated_version!=source_version:
            self.state_is_dense=all(bool(c.token_valid.all()) for s in state.layers for c in s.attention_cache.chunks)
            self.validated_state=state;self.validated_version=source_version
        if not self.state_is_dense:return eager(stack,x,condition,kw)
        fast=getattr(stack, 'inference_fast_graph_checks', False)
        from .graph_checks import parameter_versions, tensors as fast_tensors
        flatten=fast_tensors if fast else tensors
        state_tensors=list(flatten(state))
        extras={k:v for k,v in kw.items() if k!='state'}
        versions=(parameter_versions(stack) if fast else tuple((id(p),p._version,p.dtype,p.device) for p in stack.parameters()))
        flags=tuple((getattr(l.mixer._mamba_wrapper,'inference_fused_adt',False),
                     getattr(l.mixer,'inference_prepared_attention',False),
                     getattr(l.mixer,'inference_dual_attention',False),
                     getattr(l,'inference_fused_adaln',False), getattr(l,'inference_fused_pointwise',False)) for l in stack.layers)
        signature=tuple((tuple(t.shape),t.dtype,t.device) for t in [x,condition]+state_tensors+list(flatten(extras)))
        scalar_options=tuple((k,v) for k,v in extras.items() if not isinstance(v,torch.Tensor))
        key=(signature,versions,flags,scalar_options,getattr(stack,'inference_graph_shared_kv',False),
             tuple(tuple(c.num_tokens for c in s.attention_cache.chunks) for s in state.layers))
        entry=self.entries.get(key)
        if entry is None:
            if len(self.entries)>=self.limit:
                # Bound memory; eviction does not alter durable session state.
                self.entries.pop(next(iter(self.entries)))
            start=time.perf_counter();allocated_before=torch.cuda.memory_allocated();reserved_before=torch.cuda.memory_reserved()
            sx=x.clone();sc=condition.clone();skw=copy.deepcopy(kw)
            old=[]
            for layer in stack.layers:
                att=layer.mixer.attention
                old.append((att,getattr(att,'_validated_graph_all_valid',False),getattr(att,'inference_cache',False)))
                att._validated_graph_all_valid=True
                # Rebuild history inside graph after static state buffers update.
                att.inference_cache=False
            try:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):eager(stack,sx,sc,skw)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=eager(stack,sx,sc,skw)[0]
            finally:
                for att,valid,caching in old:
                    att._validated_graph_all_valid=valid;att.inference_cache=caching
            entry={'x':sx,'condition':sc,'kw':skw,'out':out,'graph':graph,'source_state':state,'source_version':source_version}
            self.entries[key]=entry
            self.events.append({'prepare_seconds':time.perf_counter()-start,'buckets':len(self.entries),
                                'input_shape':list(x.shape),'history_chunks':len(state.layers[0].attention_cache.chunks),
                                'allocated_delta':torch.cuda.memory_allocated()-allocated_before,
                                'reserved_delta':torch.cuda.memory_reserved()-reserved_before})
        else:
            entry['x'].copy_(x);entry['condition'].copy_(condition)
            if entry['source_state'] is not state or entry['source_version']!=source_version:
                for a,b in zip(flatten(entry['kw']['state']),state_tensors):a.copy_(b)
                entry['source_state']=state;entry['source_version']=source_version
            for k,v in extras.items():
                if isinstance(v,torch.Tensor):entry['kw'][k].copy_(v)
        entry['graph'].replay();self.replays+=1
        # CFG serial branches must own their outputs; next replay overwrites scratch.
        return entry['out'].clone(),None,{'layers':[]}
