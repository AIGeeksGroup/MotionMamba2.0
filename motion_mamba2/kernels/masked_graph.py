"""Install capture-safe sparse attention masks without modifying committed state."""
import types
import torch
def graph_allowed_mask(*,key_valid,query_valid,prefix_tokens,causal_within_chunk):
    batch,q_len=query_valid.shape;kv_len=key_valid.shape[1]
    allowed=key_valid[:,None,:].expand(batch,q_len,kv_len)
    if causal_within_chunk:
        q=torch.arange(q_len,device=key_valid.device)[:,None]
        k=torch.arange(kv_len,device=key_valid.device)[None,:]
        allowed=allowed & (k<=prefix_tokens+q)[None]
    allowed=allowed & query_valid[:,:,None]
    if kv_len:
        allowed=allowed.clone();allowed[:,:,0] |= ~query_valid
    return allowed
def install_masked_graphs(client):
    from motion_mamba2.kernels import cuda_graph,masked_cuda_graph
    cuda_graph.ReadOnlyStackGraphs=masked_cuda_graph.ReadOnlyStackGraphs
    for block in (client.controller.root_block,client.controller.body_block):
        stack=block.sequence_stack
        if getattr(stack,'inference_masked_graph',False):continue
        if hasattr(stack,'_inference_graphs'):del stack._inference_graphs
        for layer in stack.layers:
            attention=layer.mixer.attention;original=attention._build_allowed_mask
            def build(self,_original=original,**kwargs):
                if getattr(self,'_masked_graph_capture',False):return graph_allowed_mask(**kwargs)
                return _original(**kwargs)
            attention._build_allowed_mask=types.MethodType(build,attention)
        stack.inference_masked_graph=True
