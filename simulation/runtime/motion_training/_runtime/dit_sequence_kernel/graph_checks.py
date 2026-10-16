"""Same graph safety signatures, without recursive generator chains or names.

No parameter/state validation is cached or omitted. Reassignments, shared
modules, in-place weight changes, shape/dtype changes are checked on every call.
"""
import torch

def parameter_versions(root):
    pending=[root];modules=set();parameters=set();result=[]
    while pending:
        module=pending.pop()
        if module is None or id(module) in modules:continue
        modules.add(id(module))
        for p in module._parameters.values():
            if p is not None and id(p) not in parameters:
                parameters.add(id(p))
                result.append((id(p),p._version,p.dtype,p.device))
        pending.extend(reversed(tuple(module._modules.values())))
    return tuple(result)

def tensors(value):
    pending=[value]
    while pending:
        x=pending.pop()
        if isinstance(x,torch.Tensor):yield x
        elif isinstance(x,dict):pending.extend(reversed(tuple(x.values())))
        elif isinstance(x,(list,tuple)):pending.extend(reversed(x))
        elif hasattr(x,'__dict__'):pending.append(vars(x))
