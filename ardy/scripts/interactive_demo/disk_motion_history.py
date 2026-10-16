"""Append-only source-frame history with a bounded in-memory window cache.

The original viewer indexes absolute frames; keep that API while putting old
windows on disk. Explicit large exports may allocate; normal playback does not.
"""
from collections import OrderedDict
from pathlib import Path
import threading
import numpy as np
import torch


class DiskMotionHistory:
    def __init__(self, directory, name, window=52, cache_windows=2):
        self.directory = Path(directory)/name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.window = window
        self.cache_windows = cache_windows
        self.cache = OrderedDict()
        self.count = 0
        self.tail_shape = ()
        self.lock = threading.RLock()

    @property
    def shape(self):
        return (1, self.count*self.window, *self.tail_shape)

    def append(self, tensor):
        array = tensor.detach().cpu().numpy().copy()
        if array.shape[:2] != (1, self.window):
            raise ValueError('history append must be a single complete source window')
        with self.lock:
            if self.count and array.shape[2:] != self.tail_shape:
                raise ValueError('history shape changed')
            self.tail_shape = array.shape[2:]
            np.save(self.directory/f'{self.count:08d}.npy', array)
            self.cache[self.count] = array
            self.count += 1
            self._trim()

    def _trim(self):
        while len(self.cache) > self.cache_windows:
            self.cache.popitem(last=False)

    def replace_tail(self, window_index, tensor):
        """Replace one full window and invalidate its generated descendants."""
        array = tensor.detach().cpu().numpy().copy()
        with self.lock:
            if not 0 <= window_index < self.count or array.shape != (1, self.window, *self.tail_shape):
                raise ValueError('invalid history replacement')
            np.save(self.directory/f'{window_index:08d}.npy', array)
            for index in range(window_index + 1, self.count):
                (self.directory/f'{index:08d}.npy').unlink(missing_ok=True)
            self.count = window_index + 1
            self.cache = OrderedDict((k, v) for k, v in self.cache.items() if k < window_index)
            self.cache[window_index] = array
            self._trim()

    def _window(self, index):
        if index not in self.cache:
            self.cache[index] = np.load(self.directory/f'{index:08d}.npy')
        self.cache.move_to_end(index)
        self._trim()
        return self.cache[index]

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) < 2:
            raise IndexError('use explicit batch and source-frame indices')
        batch, frame, *tail = index
        with self.lock:
            if isinstance(frame, (int, np.integer)):
                if frame < 0:
                    frame += self.shape[1]
                if not 0 <= frame < self.shape[1]:
                    raise IndexError(frame)
                array = self._window(frame//self.window)[(batch, frame%self.window, *tail)]
            elif isinstance(frame, slice):
                indices = range(*frame.indices(self.shape[1]))
                parts = [self._window(i//self.window)[:, i%self.window:i%self.window+1] for i in indices]
                array = (np.concatenate(parts, axis=1) if parts else np.empty((1,0,*self.tail_shape), dtype=np.float32))
                array = array[(batch, slice(None), *tail)]
            else:
                raise IndexError('source-frame index must be int or slice')
            return torch.from_numpy(np.array(array, copy=True))
