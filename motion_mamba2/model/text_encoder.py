'Text encoder.'
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import torch

class ExactMotionMamba2TextEncoder:

    def __init__(self, *, base_mirror: str, mntp_adapter: str, supervised_adapter: str, device: str='cuda:0') -> None:
        if device.startswith('cuda') and torch.cuda.device_count() != 1:
            raise RuntimeError('Motion Mamba 2.0 LLM2Vec replicates its 8B encoder across all visible GPUs. Set CUDA_VISIBLE_DEVICES to one physical GPU and use cuda:0.')
        from motion_mamba2.model.language_encoder import LLM2Vec
        from peft import PeftModel
        self.model = LLM2Vec.from_pretrained(base_model_name_or_path=str(Path(base_mirror).resolve()), peft_model_name_or_path=str(Path(mntp_adapter).resolve()), merge_peft=True, torch_dtype=torch.bfloat16)
        self.model.model = PeftModel.from_pretrained(self.model.model, str(Path(supervised_adapter).resolve()))
        self.model.model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
        self._device = device
        self.to(device)
        self.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def __call__(self, texts: Sequence[str] | str):
        is_string = isinstance(texts, str)
        values = [texts] if is_string else list(texts)
        with torch.inference_mode():
            features = self.model.encode(values, batch_size=1, show_progress_bar=False, device=self._device)
        features = torch.as_tensor(features, device=self._device)[:, None, :]
        lengths = [1] * len(values)
        if is_string:
            return (features[0], lengths[0])
        return (features, lengths)

    def to(self, device: torch.device | str | None=None, dtype: torch.dtype | None=None) -> 'ExactMotionMamba2TextEncoder':
        if device is not None and dtype is not None:
            self.model = self.model.to(device=device, dtype=dtype)
        elif device is not None:
            self.model = self.model.to(device)
        elif dtype is not None:
            self.model = self.model.to(dtype=dtype)
        if device is not None:
            self._device = str(device)
        return self

    def eval(self) -> 'ExactMotionMamba2TextEncoder':
        self.model.eval()
        return self

    def get_device(self):
        return self.model.model.device
__all__ = ['ExactMotionMamba2TextEncoder']
