"""FP32 AdamW master parameters for low-precision compute weights.

Keep kernel-facing model parameters unchanged in dtype. Gradients are copied
after DDP reduction/accumulation/clipping, and only the compute copy is rounded
after an update. FP32 master values and moments persist across resume.
"""
import torch

MASTER_CONTRACT = 'fp32_master_v1'


class FP32MasterAdamW:
    def __init__(self, parameters, *, parameter_names=None, **kwargs):
        self.compute_parameters = list(parameters)
        if not self.compute_parameters or len({id(p) for p in self.compute_parameters}) != len(self.compute_parameters):
            raise ValueError('Expected nonempty unique parameters')
        if any(not p.requires_grad or not p.is_floating_point() for p in self.compute_parameters):
            raise ValueError('Optimizer accepts only trainable floating point parameters')
        names = list(parameter_names) if parameter_names is not None else [str(i) for i in range(len(self.compute_parameters))]
        if len(names) != len(self.compute_parameters) or len(set(names)) != len(names):
            raise ValueError('Parameter names must be unique and complete')
        self.signature = [dict(name=n, shape=list(p.shape), dtype=str(p.dtype))
                          for n, p in zip(names, self.compute_parameters)]
        self.masters = {i: torch.nn.Parameter(p.detach().float().clone())
                        for i, p in enumerate(self.compute_parameters)
                        if p.dtype in (torch.float16, torch.bfloat16)}
        self.optimizer = torch.optim.AdamW(
            [self.masters.get(i, p) for i, p in enumerate(self.compute_parameters)], **kwargs)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    @torch.no_grad()
    def sync_master_from_model(self):
        """Call after an init checkpoint load, before the first optimizer step."""
        if self.optimizer.state:
            raise RuntimeError('Cannot discard accumulated master values after optimizer updates')
        for i, master in self.masters.items():
            master.copy_(self.compute_parameters[i])

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for i in self.masters:
            parameter = self.compute_parameters[i]
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.detach_()
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures are unsupported; reduce and clip model gradients before step')
        for i, master in self.masters.items():
            gradient = self.compute_parameters[i].grad
            if gradient is not None and gradient.is_sparse:
                raise ValueError('Sparse gradients are unsupported')
            master.grad = None if gradient is None else gradient.detach().float()
        result = self.optimizer.step()
        for i, master in self.masters.items():
            self.compute_parameters[i].copy_(master)
        return result

    def state_dict(self):
        result = self.optimizer.state_dict()
        result.update(precision_contract=MASTER_CONTRACT, compute_signature=self.signature,
                      master_parameters={str(i): p.detach() for i, p in self.masters.items()})
        return result

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        if state_dict.get('precision_contract') != MASTER_CONTRACT:
            raise ValueError('Cannot resume legacy AdamW as FP32-master AdamW; start a separate initialized run')
        if state_dict.get('compute_signature') != self.signature:
            raise ValueError('Optimizer compute parameter identity, shape or dtype changed')
        saved = state_dict.get('master_parameters', {})
        if set(saved) != {str(i) for i in self.masters}:
            raise ValueError('Missing or extra master parameters')
        for i, master in self.masters.items():
            value = saved[str(i)]
            compute = self.compute_parameters[i]
            if value.dtype != torch.float32 or value.shape != master.shape or not torch.isfinite(value).all():
                raise ValueError('Invalid saved master parameter')
            if not torch.equal(value.to(compute), compute):
                raise ValueError('Saved master does not round to the loaded model checkpoint')
        self.optimizer.load_state_dict({k: state_dict[k] for k in ['state', 'param_groups']})
        for i, master in self.masters.items():
            master.copy_(saved[str(i)])
            for key in ['exp_avg', 'exp_avg_sq']:
                value = self.optimizer.state.get(master, {}).get(key)
                if value is not None and value.dtype != torch.float32:
                    raise ValueError('Master optimizer moments must remain FP32')


def create_adamw(parameters, *, precision='legacy', parameter_names=None, **kwargs):
    if precision == 'legacy':
        return torch.optim.AdamW(parameters, **kwargs)
    if precision == MASTER_CONTRACT:
        return FP32MasterAdamW(parameters, parameter_names=parameter_names, **kwargs)
    raise ValueError(f'Unknown optimizer precision: {precision}')
