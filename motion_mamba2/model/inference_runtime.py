"""Read-only acceleration controls; history commits remain outside CUDA Graphs."""
def configure_acceleration(client,settings):
    enabled=settings.get('acceleration','eager')=='cuda_graph'
    if settings.get('acceleration') not in ['eager','cuda_graph']:raise ValueError('Unknown acceleration option')
    if enabled:
        from motion_mamba2.kernels.masked_graph import install_masked_graphs
        install_masked_graphs(client)
    for block in [client.controller.root_block,client.controller.body_block]:
        block.inference_no_diagnostics=enabled
        block.inference_no_pending=enabled
        block.inference_cfg_parallel=False
        block.inference_no_commit_diagnostics=enabled
        block.inference_serial_cfg_graph=enabled
        stack=block.sequence_stack
        stack.inference_graph=enabled;stack.inference_graph_shared_kv=enabled;stack.inference_fast_graph_checks=enabled
        if hasattr(stack,'_inference_graphs'):stack._inference_graphs.limit=settings.get('graph_cache_capacity',64)
        for layer in stack.layers:
            mixer=layer.mixer
            if mixer._mamba_wrapper is not None:mixer._mamba_wrapper.inference_fused_adt=enabled
            mixer.attention.inference_cache=enabled;mixer.inference_prepared_attention=enabled
            mixer.inference_dual_attention=False;layer.inference_fused_adaln=False;layer.inference_fused_pointwise=enabled
