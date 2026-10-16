"""Service-only bounded graph cache and complete ordinary-session warmup."""
import importlib
import shutil


def configure_graph_capacity(client, capacity=16):
    for block in (client.controller.root_block, client.controller.body_block):
        stack = block.sequence_stack
        module = importlib.import_module(stack.__class__.__module__.rsplit('.', 1)[0]+'.inference_graph')
        if not hasattr(stack, '_inference_graphs'):
            stack._inference_graphs = module.ReadOnlyStackGraphs(limit=capacity)
        else:
            stack._inference_graphs.limit = capacity


def warm_ordinary_session(runtime, text, mask, seed=202610084):
    session = runtime.new_session(seed)
    try:
        for window in range(6):
            runtime.step(session, text, mask, 'warm', window, 0)
            runtime.publish(session, window+1, 0)
    finally:
        if hasattr(session, 'archive'):
            shutil.rmtree(session.archive, ignore_errors=True)
