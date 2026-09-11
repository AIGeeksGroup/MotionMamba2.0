"""Frozen original HumanML3D text-motion evaluator and benchmark metrics."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy import linalg
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


POS_ENUMERATOR = {
    "VERB": 0, "NOUN": 1, "DET": 2, "ADP": 3, "NUM": 4,
    "AUX": 5, "PRON": 6, "ADJ": 7, "ADV": 8, "Loc_VIP": 9,
    "Body_VIP": 10, "Obj_VIP": 11, "Act_VIP": 12, "Desc_VIP": 13,
    "OTHER": 14,
}
VIP = {
    "Loc_VIP": ("left", "right", "clockwise", "counterclockwise", "anticlockwise", "forward", "back", "backward", "up", "down", "straight", "curve"),
    "Body_VIP": ("arm", "chin", "foot", "feet", "face", "hand", "mouth", "leg", "waist", "eye", "knee", "shoulder", "thigh"),
    "Obj_VIP": ("stair", "dumbbell", "chair", "window", "floor", "car", "ball", "handrail", "baseball", "basketball"),
    "Act_VIP": ("walk", "run", "swing", "pick", "bring", "kick", "put", "squat", "throw", "hop", "dance", "jump", "turn", "stumble", "stop", "sit", "lift", "lower", "raise", "wash", "stand", "kneel", "stroll", "rub", "bend", "balance", "flap", "jog", "shuffle", "lean", "rotate", "spin", "spread", "climb"),
    "Desc_VIP": ("slowly", "carefully", "fast", "careful", "slow", "quickly", "happy", "angry", "sad", "happily", "angrily", "sadly"),
}


class WordVectorizer:
    def __init__(self, root: str | Path):
        root = Path(root)
        vectors = np.load(root / "our_vab_data.npy")
        with (root / "our_vab_words.pkl").open("rb") as handle:
            words = pickle.load(handle)
        with (root / "our_vab_idx.pkl").open("rb") as handle:
            indices = pickle.load(handle)
        self.word2vec = {word: vectors[indices[word]] for word in words}

    def __getitem__(self, token: str) -> tuple[np.ndarray, np.ndarray]:
        word, pos = token.split("/", 1)
        if word not in self.word2vec:
            word, pos = "unk", "OTHER"
        else:
            for vip_pos, vocabulary in VIP.items():
                if word in vocabulary:
                    pos = vip_pos
                    break
        one_hot = np.zeros(len(POS_ENUMERATOR), dtype=np.float32)
        one_hot[POS_ENUMERATOR.get(pos, POS_ENUMERATOR["OTHER"])] = 1.0
        return self.word2vec[word].astype(np.float32, copy=False), one_hot

    def encode(self, tokens: list[str], max_text_length: int = 20):
        tokens = tokens[:max_text_length]
        tokens = ["sos/OTHER", *tokens, "eos/OTHER"]
        length = len(tokens)
        tokens += ["unk/OTHER"] * (max_text_length + 2 - len(tokens))
        word, pos = zip(*(self[token] for token in tokens))
        return np.stack(word), np.stack(pos), length


class MovementEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(259, 512, 4, 2, 1), nn.Dropout(0.2, inplace=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 512, 4, 2, 1), nn.Dropout(0.2, inplace=True), nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(512, 512)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        return self.out_net(self.main(motion.permute(0, 2, 1)).permute(0, 2, 1))


class BiGRUEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *, text: bool):
        super().__init__()
        self.text = text
        if text:
            self.pos_emb = nn.Linear(len(POS_ENUMERATOR), input_size)
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden_size, 512),
        )
        self.hidden = nn.Parameter(torch.empty(2, 1, hidden_size), requires_grad=False)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor, pos: torch.Tensor | None = None):
        if self.text:
            if pos is None:
                raise ValueError("text encoder requires POS one-hot features")
            values = values + self.pos_emb(pos)
        embedded = self.input_emb(values)
        # Caption lengths are not sorted with the motion batch order. Use
        # unsorted packing, matching the official wrapper's per-example output.
        packed = pack_padded_sequence(embedded, lengths.cpu().tolist(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed, self.hidden.repeat(1, len(values), 1))
        return self.output_net(torch.cat((hidden[0], hidden[1]), dim=-1))


class HumanML3DOfficialEvaluator:
    def __init__(self, assets: str | Path, device: str | torch.device = "cuda"):
        self.assets = Path(assets)
        self.device = torch.device(device)
        self.mean = np.load(self.assets / "stats/mean.npy").astype(np.float32)
        self.std = np.load(self.assets / "stats/std.npy").astype(np.float32)
        self.vectorizer = WordVectorizer(self.assets / "glove")
        self.movement = MovementEncoder()
        self.text = BiGRUEncoder(300, 512, text=True)
        self.motion = BiGRUEncoder(512, 1024, text=False)
        tensors = load_file(self.assets / "model.safetensors")
        for prefix, module in (("movement_encoder.", self.movement), ("text_encoder.", self.text), ("motion_encoder.", self.motion)):
            state = {key.removeprefix(prefix): value for key, value in tensors.items() if key.startswith(prefix)}
            module.load_state_dict(state, strict=True)
            module.to(self.device).eval()

    @torch.inference_mode()
    def embed(self, motions: np.ndarray, motion_lengths: np.ndarray, tokens: list[list[str]] | None = None):
        order = np.argsort(motion_lengths)[::-1].copy()
        normalized = (motions[order] - self.mean) / self.std
        motion_tensor = torch.from_numpy(normalized).to(self.device, dtype=torch.float32)
        lengths = torch.from_numpy(motion_lengths[order].copy()).long().to(self.device)
        movements = self.movement(motion_tensor[..., :-4])
        motion_embedding = self.motion(movements, lengths // 4)
        if tokens is None:
            return None, motion_embedding.cpu().numpy(), order
        encoded = [self.vectorizer.encode(tokens[index]) for index in order]
        words = torch.from_numpy(np.stack([item[0] for item in encoded])).to(self.device)
        pos = torch.from_numpy(np.stack([item[1] for item in encoded])).to(self.device)
        caption_lengths = torch.tensor([item[2] for item in encoded], device=self.device)
        text_embedding = self.text(words, caption_lengths, pos)
        return text_embedding.cpu().numpy(), motion_embedding.cpu().numpy(), order


def r_precision(text: np.ndarray, motion: np.ndarray, top_k: int = 3) -> np.ndarray:
    squared = -2 * text @ motion.T + np.square(text).sum(1, keepdims=True) + np.square(motion).sum(1)
    ranking = np.argsort(np.sqrt(np.maximum(squared, 0.0)), axis=1)
    correct = ranking == np.arange(len(text))[:, None]
    return np.logical_or.accumulate(correct[:, :top_k], axis=1).sum(axis=0)


def frechet_distance(first: np.ndarray, second: np.ndarray, eps: float = 1e-6) -> float:
    mu1, mu2 = first.mean(0), second.mean(0)
    sigma1, sigma2 = np.cov(first, rowvar=False), np.cov(second, rowvar=False)
    covariance, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covariance).all():
        offset = np.eye(len(mu1)) * eps
        covariance = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covariance):
        covariance = covariance.real
    difference = mu1 - mu2
    return float(difference.dot(difference) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covariance))
