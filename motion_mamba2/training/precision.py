'Training precision.'
import torch
FP32_BOUNDARIES = 'g1_root_output_geometry_fp32_v1'

def unwrap(raw):
    return raw.module if hasattr(raw, 'module') else raw

def configure_training_precision(raw, codec, contract):
    if contract != FP32_BOUNDARIES:
        raise ValueError(f'Unknown G1 training precision contract: {contract}')
    model = unwrap(raw)
    if contract == FP32_BOUNDARIES:
        from motion_mamba2.model.dit import MotionDiTBlock
        if not isinstance(model.root_model, MotionDiTBlock):
            raise TypeError('FP32 root output requires the MotionMamba2 DiT block')
        if any((p.dtype != torch.float32 for p in model.root_model.output_linear.parameters())):
            raise ValueError('Root output parameters must already be FP32; do not recast optimizer parameters')
        if any((p.requires_grad for p in codec.parameters())):
            raise ValueError('This contract requires a frozen codec')
        if any((p.dtype != torch.float32 for p in codec.parameters() if p.is_floating_point())):
            raise ValueError('Frozen codec parameters must already be FP32')
    model.root_model.output_projection_fp32 = contract == FP32_BOUNDARIES
    model.training_precision_contract = contract
    return contract

def prediction_loss_context(raw, prediction):
    contract = getattr(unwrap(raw), 'training_precision_contract', None)
    if contract != FP32_BOUNDARIES:
        raise ValueError(f'Unknown G1 training precision contract: {contract}')
    if prediction.dtype != torch.float32:
        raise ValueError('G1 hybrid prediction must be FP32 before physical losses')
    return torch.autocast(device_type=prediction.device.type, enabled=False)
