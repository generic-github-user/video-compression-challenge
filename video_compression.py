from arithmeticcoding import *
from collections import Counter
from datasets import load_dataset
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from pprint import pprint
from rich import print as rprint
from rich.pretty import pprint as pprint2
from tensorflow import keras
from typing import Tuple, Callable
from video_compression_core import load_data, get_paths, get_data, to_binary, build_model
import base64
import fire
import io
import itertools
import math
import multiprocessing as mp
import numpy as np
import pickle
import shutil
import tqdm
import zlib

submission_path = Path('./compression_challenge_submission')
# print(gpus := tf.config.list_physical_devices('GPU'))
# if len(gpus) > 0:
    # tf.config.experimental.set_memory_growth(gpus[0], True)

ds = load_data()
paths = get_paths()

def compression_rate(f: Callable[np.ndarray, bytes], data: np.ndarray, **kwargs) -> float:
    r = (a := len(data.tobytes())) / (b := len(f(data, **kwargs)))
    print(f'{a} bytes -> {b} bytes')
    return r

def zlib_compression_rate(data: np.ndarray) -> float:
    return len(s := data.tobytes()) / len(zlib.compress(s, level=9))

class ContextDataset(keras.utils.PyDataset):
    def __init__(self, samples: np.ndarray, dims: Tuple[int, ...] = None, k: int = 2, t: int = 5,
                 binary: bool = True, batch_size: int = 32, m: int = 10, shuffle: bool = True,
                 reshuffle: bool = True, limit: int = 5000, extra_bit=True,
                 debug=True, **kwargs):
        super().__init__(**kwargs)
        self.samples = samples
        if dims is None:
            dims = samples.shape[1:]
        self.dims = dims
        self.k = k
        self.t = t
        self.m, self.binary = m, binary
        self.batch_size = batch_size
        self.limit = limit
        self.extra_bit = extra_bit
        
        self.shuffle, self.reshuffle = shuffle, reshuffle
        if self.shuffle:
            self.p = np.random.permutation(self.n_batches())[:len(self)]
        # TODO: specially mark out-of-bounds pixels
        # THE `-1` is VERY IMPORTANT, DO NOT REMOVE
        self.padded = np.pad(self.samples[:, :-1, :, :], [(0, 0), (self.t, 0), (self.k, self.k), (self.k, self.k)],
            mode='constant', constant_values=(2**m if extra_bit else 0))

    def n_batches(self) -> int:
        return math.ceil(self.samples.size / self.batch_size)

    def __len__(self) -> int:
        return min(self.n_batches(), self.limit)

    def get_context(self, pos: Tuple[int, ...]) -> np.ndarray:
        i, j, k, l = pos
        h = self.k * 2 + 1
        r = self.padded[i, j:j+self.t, k:k+h, l:l+h]
        if self.binary:
            r = to_binary(r, self.m + int(self.extra_bit))
        else:
            r = r.astype(np.float32) / 1024.0
        return r

    def get_single(self, idx: int) -> Tuple[np.ndarray, int]:
        i, j, k, l = np.unravel_index(idx, self.samples.shape, order='F')
        return (self.get_context((i, j, k, l)), self.samples[i, j, k, l])

    def materialize(self) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self) * self.batch_size
        s = (n, self.t, self.k*2+1, self.k*2+1)
        xs, ys = np.empty(s + (self.m + int(self.extra_bit),) if self.binary else s), np.empty((n, 1))
        for i in range(len(self) * self.batch_size):
            xs[i], ys[i] = self.get_single(i)
        return xs, ys

    def __getitem__(self, idx: int):
        if self.shuffle:
            idx = self.p[idx]
        t_, height, width = self.dims
        window_shape = (1, self.t, self.k*2+1, self.k*2+1)
        r = range(i := idx*self.batch_size, i+self.batch_size)
        xs, ys = zip(*[self.get_single(i) for i in r])
        return np.stack(xs, axis=0), np.array(ys)

    def on_epoch_end(self):
        if self.reshuffle:
            self.p = np.random.permutation(self.n_batches())[:len(self)]

def fit_model(data: np.ndarray, n: int = 1000, t: int = 20, k: int = 2, m: int = 10,
                binary: bool = True, extra_bit: bool = True,
                h1: int = 100, h2: int = 100, lstm_layers: int = 2,
                epochs: int = 100, activation: str = 'tanh') -> 'tf.keras.Model':
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    j = k * 2 + 1
    tf.keras.backend.clear_session()
    m2 = m + int(extra_bit)
    sample_shape = (t, j * j * (m2 if binary else 1)) # m2 ** int(binary)
    model = build_model(t, j, m, m2, h1, h2, sample_shape)
    model.summary()
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./training_checkpoint.weights.h5",
        save_weights_only=True,
        save_freq="epoch",
    )
    model.fit(ContextDataset(data,
                             (1200, 8, 16), k, t, True, 32,
                             shuffle=True, reshuffle=True, limit=n, extra_bit=extra_bit),
              epochs=epochs, callbacks=[model_checkpoint_callback])
    return model

def compress(M: 'tf.keras.Model', data: np.ndarray, **kwargs) -> bytes:
    import tensorflow as tf
    encoder = ArithmeticEncoder(32, stream := BitOutputStream(io.BytesIO()))
    dataset = ContextDataset(data, **kwargs)
    frame_shape = data.shape[2:]
    for n, t in np.ndindex(*data.shape[:2]):
        raw_freqs = tf.nn.softmax(M(np.stack([dataset.get_context((n, t, x, y))
                    for x, y in np.ndindex(*frame_shape)], axis=0)), axis=1).numpy().reshape(frame_shape + (1024,))
        scaled_freqs = (raw_freqs * (2 ** 16)).round().astype(int) + 1
        for x, y in np.ndindex(*frame_shape):
            freqs = SimpleFrequencyTable(scaled_freqs[x, y]) # TODO
            encoder.write(freqs, data[n, t, x, y])
    # papering over an incredibly cursed bug
    for i in range(16 * 8):
        encoder.write(freqs, 0) # eof
    encoder.finish()
    return stream.output.getvalue()

def compress_wrapped(xs: np.ndarray, p: str) -> float:
    import tensorflow as tf # ??
    tf.config.set_visible_devices([], 'GPU')

    t = 5
    m2 = 11
    model = build_model(t, 5, 10, m2, 100, 100, (t, 5 * 5 * m2))
    model.load_weights('./model_checkpoint.weights.h5')
    compressed = compress(model, xs[np.newaxis, ...])
    (submission_path / Path(p).name).write_bytes(compressed)

    print(ratio := len(xs.tobytes()) / len(compressed))
    return ratio

def compress_parallel(n_processes: int, data: np.ndarray) -> float:
    ctx = mp.get_context("spawn")
    submission_path.mkdir(exist_ok=True)
    with ctx.Pool(n_processes) as pool:
        ratios = tqdm.tqdm(pool.starmap(compress_wrapped, zip(data, paths[:len(data)])),
                             total=data.shape[0])
        ratios = list(ratios)
    for p in ['decompress.py', 'video_compression_core.py',
              'arithmeticcoding.py', 'model_checkpoint.weights.h5']:
        shutil.copy(f'./{p}', submission_path / p)
    shutil.make_archive(str(submission_path), 'zip', str(submission_path))
    return np.mean(ratios)

def train():
    model = fit_model(get_data(5000), n=3000, t=5, k=2, lstm_layers=1, h1=100, h2=100,
                      binary=True, activation='relu', epochs=100)
    model.save_weights('./model_checkpoint.weights.h5')
    model.save('./checkpoint.keras')

def test_compression():
    s = get_data(5000)
    # for i in range(1):
        # print(len(c := compress(model, s[i:i+1, :1200], t=5)))
    print(f'Average compression rate: {compress_parallel(30, s[:, :])}')

if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'test_compression': test_compression
    })
