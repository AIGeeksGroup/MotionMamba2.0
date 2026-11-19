"""Capture frozen FP32 B1 codec operations; never batch their matrix multiplies.

Only all-real, uniform, patch-aligned training inputs are supported. No model,
optimizer, diffusion, decoder or streaming runtime is captured or patched.
"""
import torch


class FrozenCodecGraphs:
    def __init__(self, converter):
        self.converter = converter
        self.graphs = {}
        self.replays = 0

    def _serial(self, motion, pad):
        tokens, masks = [], []
        encoder = self.converter.autoencoder.encoder
        module = encoder.seqTransEncoder
        original = module.forward
        # The causal encoder uses the standard causal attention mask.
        # Provide that same flag instead of rediscovering it with a GPU->CPU bool.
        def explicit_causal(*args, **kwargs):
            kwargs['is_causal'] = bool(encoder.is_causal)
            return original(*args, **kwargs)
        module.forward = explicit_causal
        try:
            for i in range(motion.shape[0]):
                p = pad[i:i+1]
                a, b = self.converter.get_hybrid_motion_from_explicit(motion[i:i+1], p.sum(1), p)
                tokens.append(a)
                masks.append(b)
        finally:
            module.forward = original
        return torch.cat(tokens), torch.cat(masks)

    def encode(self, motion):
        assert not torch.is_grad_enabled(), 'Frozen encoding requires no grad'
        assert not torch.is_autocast_enabled('cuda'), 'Keep the original FP32 codec boundary'
        assert not self.converter.autoencoder.training, 'Codec must remain frozen/eval'
        assert motion.dtype == torch.float32 and motion.is_cuda
        assert motion.shape[-1] == 414 and motion.shape[1] % 4 == 0
        key = (tuple(motion.shape), motion.device.index)
        if key not in self.graphs:
            static = motion.clone().contiguous()
            pad = torch.ones(static.shape[:2], dtype=torch.bool, device=static.device)
            stream = torch.cuda.Stream(device=static.device)
            stream.wait_stream(torch.cuda.current_stream(static.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._serial(static, pad)
            torch.cuda.current_stream(static.device).wait_stream(stream)
            torch.cuda.synchronize(static.device)
            graph = torch.cuda.CUDAGraph()
            # Captures B sequential B1 calls, retaining each original GEMM shape.
            with torch.cuda.graph(graph, stream=stream):
                output = self._serial(static, pad)
            torch.cuda.current_stream(static.device).wait_stream(stream)
            self.graphs[key] = (static, pad, graph, output)
        static, pad, graph, output = self.graphs[key]
        static.copy_(motion)
        graph.replay()
        self.replays += 1
        # Graph storage is overwritten on replay; callers retain distinct outputs.
        return output[0].clone(), output[1].clone()

    def uniform_history(self, motion):
        assert motion.ndim == 3 and motion.shape[1] > 0 and motion.shape[1] % 4 == 0
        chunks = []
        for start in range(0, motion.shape[1], 52):
            end = min(start+52, motion.shape[1])
            begin = max(0, start-52)
            value, _ = self.encode(motion[:, begin:end])
            chunks.append(value[:, -(end-start)//4:])
        return torch.cat(chunks, 1)
