```python
from arithmeticcoding import *
import io
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from rich import print as rprint
from rich.pretty import pprint as pprint2
from pprint import pprint
from datasets import load_dataset
import itertools
from collections import Counter
import numba
import pickle
import zlib
import base64
import tensorflow as tf
from tensorflow import keras
import math
import tqdm
```

    /home/annaa/.cache/pypoetry/virtualenvs/compression-challenge-m8JjhSn8-py3.11/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    2024-06-07 17:00:45.515308: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-06-07 17:00:47.293713: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



```python
def compression_rate(f: Callable[np.ndarray, bytes], data: np.ndarray, **kwargs) -> float:
  return len(data.tobytes()) / len(f(data, **kwargs))

def zlib_compression_rate(data: np.ndarray) -> float:
  return len(s := data.tobytes()) / len(zlib.compress(s, level=9))
```


```python
ds = load_dataset('commaai/commavq', num_proc=16)
# ds = load_dataset('commaai/commavq', streaming=True)
```


```python
list(itertools.islice(ds['0'], 1))
```




    [{'path': '3b41c0fa8959aea6c118e5714f412a2e_13.npy'}]




```python
def get_data(n: int = 10) -> np.ndarray:
    # return np.stack([np.load('./data/' + p) for p in ds['0'][:n]['path']])
    return np.stack([np.load('./data/' + x['path']) for x in itertools.islice(ds['0'], n)])
print(get_data().shape)
```

    (10, 1200, 8, 16)



```python
# adapted from https://stackoverflow.com/a/22227898
def to_binary(d: np.array, m: int) -> np.array:
    return (((d.ravel()[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int).reshape((*d.shape, m))

print(to_binary(np.arange(25).reshape((5, 5)), 5))
```

    [[[0 0 0 0 0]
      [0 0 0 0 1]
      [0 0 0 1 0]
      [0 0 0 1 1]
      [0 0 1 0 0]]
    
     [[0 0 1 0 1]
      [0 0 1 1 0]
      [0 0 1 1 1]
      [0 1 0 0 0]
      [0 1 0 0 1]]
    
     [[0 1 0 1 0]
      [0 1 0 1 1]
      [0 1 1 0 0]
      [0 1 1 0 1]
      [0 1 1 1 0]]
    
     [[0 1 1 1 1]
      [1 0 0 0 0]
      [1 0 0 0 1]
      [1 0 0 1 0]
      [1 0 0 1 1]]
    
     [[1 0 1 0 0]
      [1 0 1 0 1]
      [1 0 1 1 0]
      [1 0 1 1 1]
      [1 1 0 0 0]]]



```python
class ContextDataset(keras.utils.PyDataset):
    def __init__(self, samples: np.ndarray, dims: Tuple[int, ...] = None, k: int = 2, t: int = 10,
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
            # self.p = np.random.permutation(len(self))
            self.p = np.random.permutation(self.n_batches())[:len(self)]
        # if debug: print(self.p)
        # TODO: specially mark out-of-bounds pixels
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
        # print(i, j, k, l)
        return (self.get_context((i, j, k, l)), self.samples[i, j, k, l])

    def materialize(self) -> Tuple[np.ndarray, np.ndarray]:
        # n = self.samples.size
        n = len(self) * self.batch_size
        s = (n, self.t, self.k*2+1, self.k*2+1)
        xs, ys = np.empty(s + (self.m + int(self.extra_bit),) if self.binary else s), np.empty((n, 1))
        for i in range(len(self) * self.batch_size):
            xs[i], ys[i] = self.get_single(i) # TODO: shuffle
        return xs, ys

    def __getitem__(self, idx: int):
        if self.shuffle:
            idx = self.p[idx]
        t_, height, width = self.dims
        window_shape = (1, self.t, self.k*2+1, self.k*2+1)
        # return sliding_window_view(np.pad(self.samples[:, :-1, :, :], [(0, 0), (self.t, 0), (self.k, self.k), (self.k, self.k)]),
        #                           window_shape).reshape((self.samples.size, *window_shape[1:]))[(i:=idx*self.batch_size):i+self.batch_size]
        r = range(i := idx*self.batch_size, i+self.batch_size)
        xs, ys = zip(*[self.get_single(i) for i in r])
        return np.stack(xs, axis=0), np.array(ys)

    def on_epoch_end(self):
        if self.reshuffle:
            self.p = np.random.permutation(self.n_batches())[:len(self)]
```


```python
def fit_model(data: np.ndarray, n: int = 1000, t: int = 20, k: int = 2, m: int = 10,
                binary: bool = True, extra_bit: bool = True,
                h1: int = 100, h2: int = 100, lstm_layers: int = 2,
                epochs: int = 100, activation: str = 'tanh') -> tf.keras.Model:
    j = k * 2 + 1
    tf.keras.backend.clear_session()
    m2 = m + int(extra_bit)
    sample_shape = (t, j * j * (m2 if binary else 1)) # m2 ** int(binary)
    model = tf.keras.models.Sequential([
        # keras.layers.Flatten(input_shape=(10, 5, 5, 10)),
        # keras.layers.LSTM(50, input_shape=sample_shape, return_sequences=False),
        # convolve over time?
        keras.layers.Input((t, j, j) + ((m2,) if binary else ())),
        # keras.layers.Conv3D(16, 3, activation='relu'),
        keras.layers.Reshape(sample_shape),
        *[keras.layers.LSTM(h1, return_sequences=True) for _ in range(lstm_layers-1)],
        keras.layers.LSTM(h1),
        # keras.layers.Flatten(),
        # keras.layers.Dense(30, activation='relu'),
        # keras.layers.Dense(h2, activation=activation),
        keras.layers.Dense(h2, activation=activation),
        keras.layers.Dense(2 ** m, activation=None)
    ])
    model.summary()
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    
    model.fit(ContextDataset(data,
                             # np.diff(get_data(1000), axis=1),
                             (1200, 8, 16), k, t, True, 32,
                             shuffle=True, reshuffle=True, limit=n, extra_bit=extra_bit), epochs=epochs)
    # xs, ys = ContextDataset(get_data(1), (1200, 8, 16), 2, 10, True, 32, shuffle=True, limit=20).materialize()
    # model.fit(tf.constant(xs), tf.constant(ys), epochs=20, shuffle=True)
    return model

model = fit_model(get_data(1000), n=50, t=5, k=2, lstm_layers=1, h1=50, h2=100, binary=True, activation='relu', epochs=800)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ reshape (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">275</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">65,200</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │         <span style="color: #00af00; text-decoration-color: #00af00">5,100</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)           │       <span style="color: #00af00; text-decoration-color: #00af00">103,424</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">173,724</span> (678.61 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">173,724</span> (678.61 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    Epoch 1/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 12ms/step - accuracy: 0.0022 - loss: 6.9299
    Epoch 2/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0014 - loss: 6.9207
    Epoch 3/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.0000e+00 - loss: 6.9118
    Epoch 4/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.0038 - loss: 6.9045
    Epoch 5/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.0025 - loss: 6.9172
    Epoch 6/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0037 - loss: 6.9013
    Epoch 7/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.0037 - loss: 6.8759
    Epoch 8/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0030 - loss: 6.8917
    Epoch 9/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0042 - loss: 6.8863
    Epoch 10/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0024 - loss: 6.8675
    Epoch 11/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0026 - loss: 6.8548
    Epoch 12/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0028 - loss: 6.8723
    Epoch 13/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.0022 - loss: 6.8854
    Epoch 14/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0018 - loss: 6.8338
    Epoch 15/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0037 - loss: 6.8592
    Epoch 16/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.0013 - loss: 6.8752
    Epoch 17/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0057 - loss: 6.8301
    Epoch 18/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0029 - loss: 6.8488
    Epoch 19/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - accuracy: 0.0014 - loss: 6.8741  
    Epoch 20/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 5.6462e-04 - loss: 6.8729
    Epoch 21/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0019 - loss: 6.8547
    Epoch 22/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 7.1727e-04 - loss: 6.8581
    Epoch 23/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0062 - loss: 6.8565
    Epoch 24/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0029 - loss: 6.8692
    Epoch 25/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0021 - loss: 6.8754
    Epoch 26/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0012 - loss: 6.8197 
    Epoch 27/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.0061 - loss: 6.8476
    Epoch 28/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0045 - loss: 6.8209
    Epoch 29/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0029 - loss: 6.8692
    Epoch 30/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 1.6278e-04 - loss: 6.8377
    Epoch 31/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.0014 - loss: 6.8558
    Epoch 32/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.0020 - loss: 6.8727
    Epoch 33/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0019 - loss: 6.8281
    Epoch 34/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0014 - loss: 6.8684
    Epoch 35/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0018 - loss: 6.8337
    Epoch 36/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.0073 - loss: 6.7947
    Epoch 37/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.0015 - loss: 6.8346
    Epoch 38/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.0061 - loss: 6.8257
    Epoch 39/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 15ms/step - accuracy: 0.0058 - loss: 6.8419
    Epoch 40/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0058 - loss: 6.8135
    Epoch 41/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0041 - loss: 6.8539
    Epoch 42/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0052 - loss: 6.7948
    Epoch 43/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 5.7013e-04 - loss: 6.8016
    Epoch 44/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 4.2420e-04 - loss: 6.8704
    Epoch 45/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0066 - loss: 6.7860
    Epoch 46/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0048 - loss: 6.8324
    Epoch 47/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0048 - loss: 6.7837
    Epoch 48/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0042 - loss: 6.7986
    Epoch 49/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0048 - loss: 6.7815
    Epoch 50/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0040 - loss: 6.7692
    Epoch 51/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0051 - loss: 6.8069
    Epoch 52/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0026 - loss: 6.7406
    Epoch 53/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0059 - loss: 6.7900
    Epoch 54/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0028 - loss: 6.8054
    Epoch 55/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0025 - loss: 6.7958
    Epoch 56/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 5.7220e-04 - loss: 6.8113
    Epoch 57/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0019 - loss: 6.7786
    Epoch 58/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0069 - loss: 6.7495
    Epoch 59/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.0062 - loss: 6.7525
    Epoch 60/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0035 - loss: 6.7281
    Epoch 61/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0082 - loss: 6.7258
    Epoch 62/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0025 - loss: 6.7313
    Epoch 63/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0066 - loss: 6.7080
    Epoch 64/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0049 - loss: 6.6687
    Epoch 65/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0044 - loss: 6.7578
    Epoch 66/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0082 - loss: 6.6886
    Epoch 67/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0057 - loss: 6.7242
    Epoch 68/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0026 - loss: 6.6474
    Epoch 69/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0087 - loss: 6.6417
    Epoch 70/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0099 - loss: 6.6726
    Epoch 71/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0028 - loss: 6.6596
    Epoch 72/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0048 - loss: 6.6273
    Epoch 73/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0072 - loss: 6.6042
    Epoch 74/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0072 - loss: 6.5940
    Epoch 75/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0132 - loss: 6.6318
    Epoch 76/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0123 - loss: 6.5479
    Epoch 77/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0193 - loss: 6.6063
    Epoch 78/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0138 - loss: 6.5787
    Epoch 79/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0132 - loss: 6.5570
    Epoch 80/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0099 - loss: 6.6288
    Epoch 81/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0143 - loss: 6.6034
    Epoch 82/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0148 - loss: 6.5234
    Epoch 83/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0089 - loss: 6.5723
    Epoch 84/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0102 - loss: 6.6224
    Epoch 85/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0172 - loss: 6.5610
    Epoch 86/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - accuracy: 0.0131 - loss: 6.5578
    Epoch 87/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0192 - loss: 6.5166
    Epoch 88/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0193 - loss: 6.5405
    Epoch 89/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0218 - loss: 6.4847
    Epoch 90/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.0252 - loss: 6.4955
    Epoch 91/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0188 - loss: 6.5277
    Epoch 92/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0254 - loss: 6.4697
    Epoch 93/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0223 - loss: 6.4609
    Epoch 94/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0161 - loss: 6.4585
    Epoch 95/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0229 - loss: 6.4050
    Epoch 96/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0270 - loss: 6.4491
    Epoch 97/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0192 - loss: 6.4873
    Epoch 98/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0158 - loss: 6.5101
    Epoch 99/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0240 - loss: 6.4187
    Epoch 100/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0336 - loss: 6.4598
    Epoch 101/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0333 - loss: 6.3806
    Epoch 102/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0293 - loss: 6.3490
    Epoch 103/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0374 - loss: 6.3258
    Epoch 104/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0319 - loss: 6.3415
    Epoch 105/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0262 - loss: 6.3508
    Epoch 106/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.0366 - loss: 6.3441
    Epoch 107/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0434 - loss: 6.3913
    Epoch 108/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0389 - loss: 6.3069
    Epoch 109/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0475 - loss: 6.3419
    Epoch 110/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0305 - loss: 6.3308
    Epoch 111/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0603 - loss: 6.1517
    Epoch 112/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.0455 - loss: 6.2255
    Epoch 113/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0595 - loss: 6.1270
    Epoch 114/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0627 - loss: 6.2490
    Epoch 115/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0696 - loss: 6.1911
    Epoch 116/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0545 - loss: 6.0888
    Epoch 117/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0522 - loss: 6.3312
    Epoch 118/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0626 - loss: 6.2081
    Epoch 119/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.0721 - loss: 6.2494
    Epoch 120/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.0574 - loss: 6.2716
    Epoch 121/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0578 - loss: 6.3451
    Epoch 122/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0773 - loss: 6.2125
    Epoch 123/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0615 - loss: 6.2882
    Epoch 124/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.0800 - loss: 6.1488
    Epoch 125/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0649 - loss: 6.1939
    Epoch 126/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0800 - loss: 6.2002
    Epoch 127/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0868 - loss: 6.0750
    Epoch 128/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0766 - loss: 6.3338
    Epoch 129/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0984 - loss: 6.0721
    Epoch 130/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0877 - loss: 6.2288
    Epoch 131/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1003 - loss: 6.0979
    Epoch 132/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0987 - loss: 6.0385
    Epoch 133/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0991 - loss: 6.0079
    Epoch 134/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - accuracy: 0.1030 - loss: 6.0097
    Epoch 135/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1202 - loss: 5.8950
    Epoch 136/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1289 - loss: 5.9955
    Epoch 137/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1017 - loss: 5.9508
    Epoch 138/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.0962 - loss: 5.9865
    Epoch 139/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1109 - loss: 5.9504
    Epoch 140/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - accuracy: 0.1138 - loss: 5.9731
    Epoch 141/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1280 - loss: 5.9393
    Epoch 142/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1189 - loss: 5.9315
    Epoch 143/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1366 - loss: 5.9277
    Epoch 144/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1245 - loss: 5.9207
    Epoch 145/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1368 - loss: 5.9651
    Epoch 146/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1298 - loss: 5.8857
    Epoch 147/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1332 - loss: 5.8490
    Epoch 148/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1438 - loss: 5.9516
    Epoch 149/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1305 - loss: 5.9128
    Epoch 150/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1447 - loss: 5.9464
    Epoch 151/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1464 - loss: 5.9732
    Epoch 152/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2021 - loss: 5.7368
    Epoch 153/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.1419 - loss: 5.9584
    Epoch 154/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.1078 - loss: 5.9971
    Epoch 155/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1573 - loss: 5.8353
    Epoch 156/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1303 - loss: 6.0947
    Epoch 157/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1689 - loss: 5.8455
    Epoch 158/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1755 - loss: 5.8866
    Epoch 159/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1609 - loss: 5.8552
    Epoch 160/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1720 - loss: 5.7765
    Epoch 161/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.1688 - loss: 5.8860
    Epoch 162/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1819 - loss: 5.8344
    Epoch 163/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1912 - loss: 5.7895
    Epoch 164/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1878 - loss: 5.7630
    Epoch 165/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1452 - loss: 6.0141
    Epoch 166/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1913 - loss: 5.7044
    Epoch 167/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2100 - loss: 5.6398
    Epoch 168/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1993 - loss: 5.7327
    Epoch 169/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2103 - loss: 5.7399
    Epoch 170/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1817 - loss: 5.7344
    Epoch 171/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1845 - loss: 5.8379
    Epoch 172/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2061 - loss: 5.7237
    Epoch 173/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2072 - loss: 5.7623
    Epoch 174/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.1975 - loss: 5.8685
    Epoch 175/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2225 - loss: 5.8438
    Epoch 176/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2765 - loss: 5.4793
    Epoch 177/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2269 - loss: 5.6584
    Epoch 178/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2436 - loss: 5.5999
    Epoch 179/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2348 - loss: 5.6227
    Epoch 180/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2230 - loss: 5.8840
    Epoch 181/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2334 - loss: 5.7418
    Epoch 182/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - accuracy: 0.2329 - loss: 5.7152
    Epoch 183/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2491 - loss: 5.7727
    Epoch 184/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2708 - loss: 5.5189
    Epoch 185/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2503 - loss: 5.6075
    Epoch 186/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2741 - loss: 5.4554
    Epoch 187/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2887 - loss: 5.4532
    Epoch 188/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3111 - loss: 5.4688
    Epoch 189/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2695 - loss: 5.5024
    Epoch 190/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2547 - loss: 5.7596
    Epoch 191/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2392 - loss: 5.7165
    Epoch 192/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2749 - loss: 5.5720
    Epoch 193/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2344 - loss: 5.7013
    Epoch 194/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3095 - loss: 5.3930
    Epoch 195/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2745 - loss: 5.5373
    Epoch 196/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3162 - loss: 5.3269
    Epoch 197/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.2794 - loss: 5.7364
    Epoch 198/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.2692 - loss: 5.6449
    Epoch 199/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.2949 - loss: 5.4901
    Epoch 200/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2694 - loss: 5.6537
    Epoch 201/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2607 - loss: 5.7391
    Epoch 202/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3216 - loss: 5.3803
    Epoch 203/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3024 - loss: 5.4048
    Epoch 204/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3196 - loss: 5.3334
    Epoch 205/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2810 - loss: 5.6293
    Epoch 206/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3145 - loss: 5.4861
    Epoch 207/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2815 - loss: 5.5970
    Epoch 208/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2815 - loss: 5.5505
    Epoch 209/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3144 - loss: 5.4857
    Epoch 210/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3629 - loss: 5.1872
    Epoch 211/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3292 - loss: 5.3331
    Epoch 212/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2858 - loss: 5.5825
    Epoch 213/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3873 - loss: 5.0852
    Epoch 214/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.3274 - loss: 5.4917
    Epoch 215/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.2871 - loss: 5.5786
    Epoch 216/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3375 - loss: 5.2666
    Epoch 217/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3067 - loss: 5.4215
    Epoch 218/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.2949 - loss: 5.6630
    Epoch 219/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3334 - loss: 5.3105
    Epoch 220/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3041 - loss: 5.5662
    Epoch 221/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - accuracy: 0.3343 - loss: 5.3894
    Epoch 222/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3486 - loss: 5.2947
    Epoch 223/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3283 - loss: 5.3511
    Epoch 224/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3322 - loss: 5.3591
    Epoch 225/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3225 - loss: 5.5641
    Epoch 226/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3402 - loss: 5.3585
    Epoch 227/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3539 - loss: 5.3078
    Epoch 228/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3487 - loss: 5.1836
    Epoch 229/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3697 - loss: 5.1459
    Epoch 230/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.3398 - loss: 5.2720
    Epoch 231/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3343 - loss: 5.3695
    Epoch 232/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3255 - loss: 5.4211
    Epoch 233/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3592 - loss: 5.1373
    Epoch 234/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3491 - loss: 5.2565
    Epoch 235/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3529 - loss: 5.3326
    Epoch 236/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3424 - loss: 5.4548
    Epoch 237/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3311 - loss: 5.4875
    Epoch 238/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.4098 - loss: 5.0110
    Epoch 239/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3711 - loss: 5.1302
    Epoch 240/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3702 - loss: 5.1460
    Epoch 241/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4010 - loss: 4.9597
    Epoch 242/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3311 - loss: 5.3699
    Epoch 243/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3670 - loss: 5.1740
    Epoch 244/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3730 - loss: 5.2289
    Epoch 245/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3278 - loss: 5.4599
    Epoch 246/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3988 - loss: 5.0180
    Epoch 247/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - accuracy: 0.3743 - loss: 5.0911
    Epoch 248/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3430 - loss: 5.2537
    Epoch 249/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3681 - loss: 5.2569
    Epoch 250/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3544 - loss: 5.3713
    Epoch 251/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3723 - loss: 5.1187
    Epoch 252/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3683 - loss: 5.2109
    Epoch 253/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3208 - loss: 5.4142
    Epoch 254/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3443 - loss: 5.3768
    Epoch 255/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3928 - loss: 5.0737
    Epoch 256/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3380 - loss: 5.3554
    Epoch 257/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3696 - loss: 5.2220
    Epoch 258/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3680 - loss: 5.2067
    Epoch 259/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3710 - loss: 5.1693
    Epoch 260/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3915 - loss: 5.0473
    Epoch 261/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3727 - loss: 5.1522
    Epoch 262/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3829 - loss: 5.1746
    Epoch 263/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.4197 - loss: 4.9450
    Epoch 264/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3186 - loss: 5.4979
    Epoch 265/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3630 - loss: 5.2454
    Epoch 266/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3598 - loss: 5.1886
    Epoch 267/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3805 - loss: 5.1627
    Epoch 268/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.4079 - loss: 5.0243
    Epoch 269/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3600 - loss: 5.3112
    Epoch 270/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3386 - loss: 5.2527
    Epoch 271/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3521 - loss: 5.3034
    Epoch 272/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.4210 - loss: 4.8014
    Epoch 273/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3807 - loss: 5.0381
    Epoch 274/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3871 - loss: 4.9726
    Epoch 275/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3569 - loss: 5.2082
    Epoch 276/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3850 - loss: 5.0516
    Epoch 277/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3711 - loss: 5.2548
    Epoch 278/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.4098 - loss: 4.9202
    Epoch 279/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3560 - loss: 5.2593
    Epoch 280/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3909 - loss: 5.0222
    Epoch 281/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3921 - loss: 5.1262
    Epoch 282/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3791 - loss: 5.1310
    Epoch 283/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.4092 - loss: 4.9583
    Epoch 284/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3835 - loss: 5.1467
    Epoch 285/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3237 - loss: 5.3913
    Epoch 286/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - accuracy: 0.3936 - loss: 4.9591
    Epoch 287/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4320 - loss: 4.9183
    Epoch 288/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3994 - loss: 5.0413
    Epoch 289/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4000 - loss: 5.0133
    Epoch 290/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3780 - loss: 5.0938
    Epoch 291/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3578 - loss: 5.2637
    Epoch 292/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3374 - loss: 5.3573
    Epoch 293/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3247 - loss: 5.3657
    Epoch 294/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3744 - loss: 5.0555
    Epoch 295/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4048 - loss: 4.9338
    Epoch 296/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3752 - loss: 5.0893
    Epoch 297/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4131 - loss: 4.9439
    Epoch 298/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3762 - loss: 5.0897
    Epoch 299/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - accuracy: 0.4325 - loss: 4.7617
    Epoch 300/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - accuracy: 0.3700 - loss: 5.1125
    Epoch 301/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - accuracy: 0.4062 - loss: 5.0230
    Epoch 302/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - accuracy: 0.4073 - loss: 4.9273
    Epoch 303/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4058 - loss: 4.7827
    Epoch 304/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4271 - loss: 4.8427
    Epoch 305/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4307 - loss: 4.7554
    Epoch 306/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4171 - loss: 4.8581
    Epoch 307/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3345 - loss: 5.2563
    Epoch 308/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3741 - loss: 5.1874
    Epoch 309/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3477 - loss: 5.2396
    Epoch 310/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3912 - loss: 4.9958
    Epoch 311/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4098 - loss: 4.9219
    Epoch 312/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3767 - loss: 5.0690
    Epoch 313/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3897 - loss: 5.0672
    Epoch 314/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3641 - loss: 5.1805
    Epoch 315/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4115 - loss: 4.9975
    Epoch 316/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4207 - loss: 4.8219
    Epoch 317/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3751 - loss: 5.0916
    Epoch 318/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3783 - loss: 5.0936
    Epoch 319/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3764 - loss: 5.1704
    Epoch 320/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3443 - loss: 5.3028
    Epoch 321/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3897 - loss: 5.1276
    Epoch 322/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.3989 - loss: 4.9002
    Epoch 323/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3705 - loss: 5.1048
    Epoch 324/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4365 - loss: 4.7779
    Epoch 325/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4017 - loss: 4.9726
    Epoch 326/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3658 - loss: 5.0882
    Epoch 327/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3778 - loss: 5.1192
    Epoch 328/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3930 - loss: 4.9661
    Epoch 329/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3963 - loss: 4.8823
    Epoch 330/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4247 - loss: 4.7502
    Epoch 331/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3743 - loss: 5.1192
    Epoch 332/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3939 - loss: 4.9108
    Epoch 333/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3963 - loss: 4.9131
    Epoch 334/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4546 - loss: 4.6318
    Epoch 335/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4016 - loss: 4.9635
    Epoch 336/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3758 - loss: 5.0263
    Epoch 337/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4201 - loss: 4.8248
    Epoch 338/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3709 - loss: 5.1120
    Epoch 339/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4098 - loss: 4.9370
    Epoch 340/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3673 - loss: 5.0870
    Epoch 341/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3852 - loss: 4.9651
    Epoch 342/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4062 - loss: 4.9166
    Epoch 343/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4373 - loss: 4.7115
    Epoch 344/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3737 - loss: 4.9752
    Epoch 345/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3568 - loss: 5.1881
    Epoch 346/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4099 - loss: 4.8289
    Epoch 347/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3385 - loss: 5.2522
    Epoch 348/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4138 - loss: 4.8285
    Epoch 349/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4099 - loss: 4.8355
    Epoch 350/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3731 - loss: 5.0824
    Epoch 351/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3942 - loss: 4.9261
    Epoch 352/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3325 - loss: 5.2010
    Epoch 353/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3712 - loss: 5.0614
    Epoch 354/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4090 - loss: 4.8356
    Epoch 355/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4228 - loss: 4.6700
    Epoch 356/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3944 - loss: 4.9241
    Epoch 357/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3960 - loss: 4.8885
    Epoch 358/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3928 - loss: 5.0356
    Epoch 359/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4143 - loss: 4.7532
    Epoch 360/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3716 - loss: 5.1003
    Epoch 361/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3750 - loss: 5.1030
    Epoch 362/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4008 - loss: 4.8469
    Epoch 363/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4164 - loss: 4.8065
    Epoch 364/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3891 - loss: 4.9502
    Epoch 365/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4069 - loss: 4.8275
    Epoch 366/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.3885 - loss: 4.9446
    Epoch 367/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4185 - loss: 4.7835
    Epoch 368/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4189 - loss: 4.8181
    Epoch 369/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4524 - loss: 4.5994
    Epoch 370/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4038 - loss: 4.8615
    Epoch 371/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3585 - loss: 5.1231
    Epoch 372/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4238 - loss: 4.6859
    Epoch 373/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3905 - loss: 4.9052
    Epoch 374/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4273 - loss: 4.7905
    Epoch 375/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3639 - loss: 5.0424
    Epoch 376/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3545 - loss: 5.0886
    Epoch 377/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4016 - loss: 4.8598
    Epoch 378/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4117 - loss: 4.7878
    Epoch 379/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3516 - loss: 5.0795
    Epoch 380/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3896 - loss: 4.8612
    Epoch 381/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4178 - loss: 4.7145
    Epoch 382/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4183 - loss: 4.7321
    Epoch 383/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3832 - loss: 4.9907
    Epoch 384/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4026 - loss: 4.8628
    Epoch 385/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4024 - loss: 4.8185
    Epoch 386/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3915 - loss: 4.8845
    Epoch 387/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3940 - loss: 4.9610
    Epoch 388/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4062 - loss: 4.8859
    Epoch 389/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.4008 - loss: 4.8687
    Epoch 390/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.3949 - loss: 4.9093
    Epoch 391/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.4029 - loss: 4.9253
    Epoch 392/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.4012 - loss: 4.7532
    Epoch 393/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3842 - loss: 4.9688
    Epoch 394/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3828 - loss: 4.9266
    Epoch 395/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3838 - loss: 4.8576
    Epoch 396/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4341 - loss: 4.6301
    Epoch 397/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3771 - loss: 4.9391
    Epoch 398/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3598 - loss: 5.0313
    Epoch 399/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3389 - loss: 5.2234
    Epoch 400/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3751 - loss: 4.9572
    Epoch 401/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3868 - loss: 4.8336
    Epoch 402/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3712 - loss: 5.0354
    Epoch 403/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3923 - loss: 4.8983
    Epoch 404/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3790 - loss: 5.0413
    Epoch 405/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3775 - loss: 4.8798
    Epoch 406/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4186 - loss: 4.7911
    Epoch 407/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3971 - loss: 4.9063
    Epoch 408/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4326 - loss: 4.6053
    Epoch 409/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3861 - loss: 5.0307
    Epoch 410/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3737 - loss: 4.9949
    Epoch 411/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3790 - loss: 4.9936
    Epoch 412/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4094 - loss: 4.8404
    Epoch 413/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3793 - loss: 4.9105
    Epoch 414/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3972 - loss: 4.8402
    Epoch 415/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4049 - loss: 4.8808
    Epoch 416/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4220 - loss: 4.7091
    Epoch 417/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4092 - loss: 4.8096
    Epoch 418/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4166 - loss: 4.7560
    Epoch 419/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3892 - loss: 4.9733
    Epoch 420/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3884 - loss: 4.9570
    Epoch 421/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4037 - loss: 4.8398
    Epoch 422/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4087 - loss: 4.7141
    Epoch 423/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3858 - loss: 4.9328
    Epoch 424/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4042 - loss: 4.8322
    Epoch 425/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4463 - loss: 4.5424
    Epoch 426/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3828 - loss: 4.9095
    Epoch 427/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4232 - loss: 4.6408
    Epoch 428/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4130 - loss: 4.7591
    Epoch 429/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3841 - loss: 4.8897
    Epoch 430/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4033 - loss: 4.7538
    Epoch 431/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3793 - loss: 5.0209
    Epoch 432/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3938 - loss: 4.8433
    Epoch 433/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4146 - loss: 4.7994
    Epoch 434/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3436 - loss: 5.1666
    Epoch 435/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3367 - loss: 5.2175
    Epoch 436/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3895 - loss: 4.8968
    Epoch 437/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3867 - loss: 4.9264
    Epoch 438/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3987 - loss: 4.8099
    Epoch 439/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3490 - loss: 5.0676
    Epoch 440/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3872 - loss: 4.9790
    Epoch 441/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3947 - loss: 4.7849
    Epoch 442/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3571 - loss: 5.1232
    Epoch 443/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3811 - loss: 4.9169
    Epoch 444/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3758 - loss: 4.9668
    Epoch 445/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3893 - loss: 4.9585
    Epoch 446/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4023 - loss: 4.7440
    Epoch 447/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4212 - loss: 4.6674
    Epoch 448/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3932 - loss: 4.8034
    Epoch 449/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4198 - loss: 4.6900
    Epoch 450/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3810 - loss: 4.8649
    Epoch 451/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3728 - loss: 4.9833
    Epoch 452/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4109 - loss: 4.7252
    Epoch 453/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4007 - loss: 4.7622
    Epoch 454/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3899 - loss: 4.8444
    Epoch 455/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4118 - loss: 4.5984
    Epoch 456/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 15ms/step - accuracy: 0.3430 - loss: 5.0776
    Epoch 457/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4251 - loss: 4.6227
    Epoch 458/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3885 - loss: 4.8841
    Epoch 459/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3939 - loss: 4.8700
    Epoch 460/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3934 - loss: 4.8777
    Epoch 461/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4096 - loss: 4.6698
    Epoch 462/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4116 - loss: 4.7103
    Epoch 463/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3987 - loss: 4.7987
    Epoch 464/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3802 - loss: 4.8948
    Epoch 465/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4468 - loss: 4.5619
    Epoch 466/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3885 - loss: 4.8679
    Epoch 467/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3942 - loss: 4.8144
    Epoch 468/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3909 - loss: 4.8025
    Epoch 469/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3723 - loss: 4.9352
    Epoch 470/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3798 - loss: 4.8701
    Epoch 471/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3523 - loss: 5.0662
    Epoch 472/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4242 - loss: 4.6976
    Epoch 473/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4180 - loss: 4.7065
    Epoch 474/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3773 - loss: 4.9198
    Epoch 475/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3632 - loss: 4.9719
    Epoch 476/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3755 - loss: 4.8705
    Epoch 477/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4102 - loss: 4.7013
    Epoch 478/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3826 - loss: 4.9440
    Epoch 479/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4166 - loss: 4.6506
    Epoch 480/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3615 - loss: 4.9560
    Epoch 481/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3765 - loss: 4.9666
    Epoch 482/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3802 - loss: 4.9751
    Epoch 483/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4340 - loss: 4.5199
    Epoch 484/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4591 - loss: 4.3649
    Epoch 485/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3929 - loss: 4.7463
    Epoch 486/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3998 - loss: 4.7813
    Epoch 487/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3698 - loss: 4.9824
    Epoch 488/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4079 - loss: 4.7380
    Epoch 489/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3570 - loss: 5.0230
    Epoch 490/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4230 - loss: 4.5932
    Epoch 491/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3953 - loss: 4.7375
    Epoch 492/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3815 - loss: 4.7239
    Epoch 493/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3738 - loss: 4.9266
    Epoch 494/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3919 - loss: 4.7700
    Epoch 495/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3896 - loss: 4.8275
    Epoch 496/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3866 - loss: 4.8607
    Epoch 497/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3962 - loss: 4.7492
    Epoch 498/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4040 - loss: 4.7219
    Epoch 499/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4159 - loss: 4.5239
    Epoch 500/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3724 - loss: 4.8883
    Epoch 501/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4519 - loss: 4.4696
    Epoch 502/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4062 - loss: 4.6900
    Epoch 503/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4003 - loss: 4.7600
    Epoch 504/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3931 - loss: 4.7630
    Epoch 505/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4504 - loss: 4.4607
    Epoch 506/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4193 - loss: 4.5832
    Epoch 507/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4172 - loss: 4.6104
    Epoch 508/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4063 - loss: 4.5511
    Epoch 509/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4044 - loss: 4.7267
    Epoch 510/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3834 - loss: 4.8511
    Epoch 511/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4042 - loss: 4.6629
    Epoch 512/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4006 - loss: 4.7370
    Epoch 513/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4088 - loss: 4.6725
    Epoch 514/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3962 - loss: 4.7561
    Epoch 515/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3982 - loss: 4.7148
    Epoch 516/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4041 - loss: 4.8221
    Epoch 517/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3465 - loss: 5.1155
    Epoch 518/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3747 - loss: 4.9025
    Epoch 519/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4155 - loss: 4.6228
    Epoch 520/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3887 - loss: 4.8612
    Epoch 521/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3874 - loss: 4.8421
    Epoch 522/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3861 - loss: 4.8782
    Epoch 523/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3822 - loss: 4.7405
    Epoch 524/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4124 - loss: 4.6522
    Epoch 525/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4131 - loss: 4.6354
    Epoch 526/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3708 - loss: 4.9382
    Epoch 527/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4119 - loss: 4.6907
    Epoch 528/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4312 - loss: 4.4946
    Epoch 529/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4237 - loss: 4.5807
    Epoch 530/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4238 - loss: 4.5969
    Epoch 531/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3905 - loss: 4.7279
    Epoch 532/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4168 - loss: 4.5890
    Epoch 533/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4262 - loss: 4.5021
    Epoch 534/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4299 - loss: 4.5505
    Epoch 535/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3973 - loss: 4.6966
    Epoch 536/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3864 - loss: 4.7232
    Epoch 537/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3789 - loss: 4.7283
    Epoch 538/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3742 - loss: 4.8504
    Epoch 539/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3812 - loss: 4.8383
    Epoch 540/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4235 - loss: 4.6273
    Epoch 541/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3873 - loss: 4.7394
    Epoch 542/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4123 - loss: 4.5881
    Epoch 543/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4661 - loss: 4.2963
    Epoch 544/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4203 - loss: 4.6403
    Epoch 545/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3831 - loss: 4.7900
    Epoch 546/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4051 - loss: 4.6959
    Epoch 547/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3771 - loss: 4.8009
    Epoch 548/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3590 - loss: 4.9769
    Epoch 549/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4118 - loss: 4.6462
    Epoch 550/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4093 - loss: 4.6875
    Epoch 551/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3809 - loss: 4.8225
    Epoch 552/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4156 - loss: 4.5434
    Epoch 553/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3806 - loss: 4.8442
    Epoch 554/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4306 - loss: 4.4729
    Epoch 555/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4295 - loss: 4.5965
    Epoch 556/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3793 - loss: 4.7222
    Epoch 557/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - accuracy: 0.3805 - loss: 4.8299
    Epoch 558/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 18ms/step - accuracy: 0.3830 - loss: 4.8089
    Epoch 559/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 15ms/step - accuracy: 0.3790 - loss: 4.7705
    Epoch 560/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 15ms/step - accuracy: 0.3998 - loss: 4.7239
    Epoch 561/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3826 - loss: 4.7054
    Epoch 562/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4256 - loss: 4.5773
    Epoch 563/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3800 - loss: 4.7828
    Epoch 564/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4250 - loss: 4.5562
    Epoch 565/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3747 - loss: 4.8342
    Epoch 566/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4527 - loss: 4.3331
    Epoch 567/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3682 - loss: 4.8632
    Epoch 568/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3880 - loss: 4.7495
    Epoch 569/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4175 - loss: 4.5730
    Epoch 570/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3826 - loss: 4.7537
    Epoch 571/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3963 - loss: 4.7016
    Epoch 572/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3726 - loss: 4.8808
    Epoch 573/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3841 - loss: 4.7723
    Epoch 574/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4261 - loss: 4.5324
    Epoch 575/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4262 - loss: 4.5229
    Epoch 576/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3458 - loss: 4.9992
    Epoch 577/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3783 - loss: 4.7473
    Epoch 578/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4231 - loss: 4.5159
    Epoch 579/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4029 - loss: 4.7585
    Epoch 580/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4086 - loss: 4.6043
    Epoch 581/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4136 - loss: 4.5649
    Epoch 582/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4054 - loss: 4.6033
    Epoch 583/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3808 - loss: 4.7869
    Epoch 584/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3966 - loss: 4.6890
    Epoch 585/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4090 - loss: 4.6682
    Epoch 586/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4232 - loss: 4.5923
    Epoch 587/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3916 - loss: 4.7346
    Epoch 588/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3673 - loss: 4.8639
    Epoch 589/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4242 - loss: 4.5587
    Epoch 590/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3686 - loss: 4.8720
    Epoch 591/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4187 - loss: 4.5185
    Epoch 592/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3686 - loss: 4.8578
    Epoch 593/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4029 - loss: 4.5690
    Epoch 594/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 18ms/step - accuracy: 0.4119 - loss: 4.5615
    Epoch 595/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 15ms/step - accuracy: 0.3858 - loss: 4.6868
    Epoch 596/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3797 - loss: 4.7463
    Epoch 597/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3916 - loss: 4.6946
    Epoch 598/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3725 - loss: 4.8382
    Epoch 599/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4148 - loss: 4.6036
    Epoch 600/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4238 - loss: 4.5386
    Epoch 601/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3569 - loss: 4.8741
    Epoch 602/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3868 - loss: 4.7785
    Epoch 603/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3972 - loss: 4.6384
    Epoch 604/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3882 - loss: 4.6938
    Epoch 605/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3680 - loss: 4.8655
    Epoch 606/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3893 - loss: 4.7516
    Epoch 607/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3647 - loss: 4.8974
    Epoch 608/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4146 - loss: 4.4670
    Epoch 609/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4216 - loss: 4.5885
    Epoch 610/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3777 - loss: 4.8732
    Epoch 611/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3789 - loss: 4.7128
    Epoch 612/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4122 - loss: 4.5767
    Epoch 613/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4174 - loss: 4.5911
    Epoch 614/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4088 - loss: 4.5376
    Epoch 615/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4175 - loss: 4.5323
    Epoch 616/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4632 - loss: 4.2294
    Epoch 617/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4086 - loss: 4.6161
    Epoch 618/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3766 - loss: 4.8189
    Epoch 619/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3788 - loss: 4.8231
    Epoch 620/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4102 - loss: 4.5644
    Epoch 621/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4231 - loss: 4.5313
    Epoch 622/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4010 - loss: 4.5830
    Epoch 623/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3887 - loss: 4.7643
    Epoch 624/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4264 - loss: 4.4968
    Epoch 625/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3890 - loss: 4.6825
    Epoch 626/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3833 - loss: 4.7528
    Epoch 627/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3812 - loss: 4.7848
    Epoch 628/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4234 - loss: 4.4631
    Epoch 629/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4114 - loss: 4.5808
    Epoch 630/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4350 - loss: 4.5119
    Epoch 631/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3667 - loss: 4.8169
    Epoch 632/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4290 - loss: 4.4820
    Epoch 633/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4026 - loss: 4.5759
    Epoch 634/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3773 - loss: 4.6689
    Epoch 635/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.4139 - loss: 4.5308
    Epoch 636/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 15ms/step - accuracy: 0.4110 - loss: 4.5723
    Epoch 637/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4216 - loss: 4.5624
    Epoch 638/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3646 - loss: 4.8764
    Epoch 639/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4325 - loss: 4.3132
    Epoch 640/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3655 - loss: 4.8538
    Epoch 641/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3884 - loss: 4.6356
    Epoch 642/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4107 - loss: 4.6078
    Epoch 643/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4298 - loss: 4.4293
    Epoch 644/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3964 - loss: 4.7742
    Epoch 645/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3902 - loss: 4.6902
    Epoch 646/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3802 - loss: 4.8125
    Epoch 647/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3838 - loss: 4.6581
    Epoch 648/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4123 - loss: 4.5315
    Epoch 649/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4122 - loss: 4.5369
    Epoch 650/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3813 - loss: 4.7168
    Epoch 651/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4057 - loss: 4.5547
    Epoch 652/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3842 - loss: 4.6796
    Epoch 653/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4181 - loss: 4.5348
    Epoch 654/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4120 - loss: 4.5793
    Epoch 655/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3965 - loss: 4.5085
    Epoch 656/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3794 - loss: 4.8186
    Epoch 657/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4136 - loss: 4.5036
    Epoch 658/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.4260 - loss: 4.4715
    Epoch 659/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3511 - loss: 4.9627
    Epoch 660/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3672 - loss: 4.7938
    Epoch 661/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3755 - loss: 4.7693
    Epoch 662/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3697 - loss: 4.7592
    Epoch 663/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3859 - loss: 4.5690
    Epoch 664/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3729 - loss: 4.8017
    Epoch 665/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4359 - loss: 4.3480
    Epoch 666/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4119 - loss: 4.5553
    Epoch 667/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4394 - loss: 4.4546
    Epoch 668/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4182 - loss: 4.3734
    Epoch 669/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3737 - loss: 4.7233
    Epoch 670/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4379 - loss: 4.3329
    Epoch 671/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4019 - loss: 4.5405
    Epoch 672/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3677 - loss: 4.8481
    Epoch 673/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4121 - loss: 4.5994
    Epoch 674/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3905 - loss: 4.6927
    Epoch 675/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 15ms/step - accuracy: 0.4259 - loss: 4.5027
    Epoch 676/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3943 - loss: 4.6035
    Epoch 677/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - accuracy: 0.3766 - loss: 4.8602
    Epoch 678/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4065 - loss: 4.5346
    Epoch 679/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4335 - loss: 4.4224
    Epoch 680/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 15ms/step - accuracy: 0.3997 - loss: 4.5903
    Epoch 681/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3670 - loss: 4.7464
    Epoch 682/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4305 - loss: 4.4272
    Epoch 683/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4054 - loss: 4.5836
    Epoch 684/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3644 - loss: 4.7642
    Epoch 685/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3870 - loss: 4.5729
    Epoch 686/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3875 - loss: 4.6024
    Epoch 687/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3716 - loss: 4.7660
    Epoch 688/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3710 - loss: 4.7644
    Epoch 689/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4123 - loss: 4.5378
    Epoch 690/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3867 - loss: 4.6368
    Epoch 691/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4370 - loss: 4.3959
    Epoch 692/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3914 - loss: 4.5619
    Epoch 693/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4031 - loss: 4.5349
    Epoch 694/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3990 - loss: 4.5458
    Epoch 695/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3571 - loss: 4.8400
    Epoch 696/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4456 - loss: 4.2648
    Epoch 697/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4099 - loss: 4.4767
    Epoch 698/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4361 - loss: 4.3850
    Epoch 699/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4351 - loss: 4.3257
    Epoch 700/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4233 - loss: 4.4132
    Epoch 701/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4145 - loss: 4.5837
    Epoch 702/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3631 - loss: 4.7352
    Epoch 703/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4400 - loss: 4.3412
    Epoch 704/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3868 - loss: 4.7319
    Epoch 705/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3852 - loss: 4.6895
    Epoch 706/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3724 - loss: 4.6445
    Epoch 707/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4666 - loss: 4.2201
    Epoch 708/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3916 - loss: 4.6395
    Epoch 709/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3980 - loss: 4.5179
    Epoch 710/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3850 - loss: 4.7090
    Epoch 711/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3790 - loss: 4.5844
    Epoch 712/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4099 - loss: 4.5463
    Epoch 713/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3765 - loss: 4.7086
    Epoch 714/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3961 - loss: 4.5721
    Epoch 715/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3826 - loss: 4.7022
    Epoch 716/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3851 - loss: 4.6590
    Epoch 717/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3587 - loss: 4.7465
    Epoch 718/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4417 - loss: 4.3107
    Epoch 719/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4597 - loss: 4.1424
    Epoch 720/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4364 - loss: 4.3768
    Epoch 721/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3429 - loss: 4.9782
    Epoch 722/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4016 - loss: 4.5919
    Epoch 723/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4254 - loss: 4.4261
    Epoch 724/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4070 - loss: 4.5848
    Epoch 725/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3982 - loss: 4.6914
    Epoch 726/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4157 - loss: 4.4067
    Epoch 727/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4228 - loss: 4.3945
    Epoch 728/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4272 - loss: 4.3352
    Epoch 729/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4259 - loss: 4.4197
    Epoch 730/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3977 - loss: 4.6258
    Epoch 731/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3787 - loss: 4.6306
    Epoch 732/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4200 - loss: 4.5186
    Epoch 733/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3597 - loss: 4.8164
    Epoch 734/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3764 - loss: 4.7101
    Epoch 735/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3539 - loss: 4.7705
    Epoch 736/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3917 - loss: 4.5667
    Epoch 737/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3387 - loss: 4.9756
    Epoch 738/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4217 - loss: 4.4785
    Epoch 739/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4045 - loss: 4.6211
    Epoch 740/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3913 - loss: 4.5351
    Epoch 741/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3873 - loss: 4.5553
    Epoch 742/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4026 - loss: 4.4329
    Epoch 743/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3934 - loss: 4.6692
    Epoch 744/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4370 - loss: 4.3859
    Epoch 745/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4073 - loss: 4.4826
    Epoch 746/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3858 - loss: 4.6404
    Epoch 747/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3827 - loss: 4.7018
    Epoch 748/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4316 - loss: 4.4441
    Epoch 749/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3693 - loss: 4.7818
    Epoch 750/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3771 - loss: 4.7544
    Epoch 751/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3802 - loss: 4.6722
    Epoch 752/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4166 - loss: 4.5326
    Epoch 753/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3990 - loss: 4.5600
    Epoch 754/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3985 - loss: 4.5674
    Epoch 755/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3783 - loss: 4.6844
    Epoch 756/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3717 - loss: 4.8002
    Epoch 757/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3923 - loss: 4.6125
    Epoch 758/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4241 - loss: 4.4321
    Epoch 759/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3795 - loss: 4.6642
    Epoch 760/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4307 - loss: 4.4112
    Epoch 761/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3816 - loss: 4.6716
    Epoch 762/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3583 - loss: 4.8352
    Epoch 763/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3891 - loss: 4.7518
    Epoch 764/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4075 - loss: 4.4632
    Epoch 765/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4334 - loss: 4.3349
    Epoch 766/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3725 - loss: 4.7693
    Epoch 767/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4429 - loss: 4.2562
    Epoch 768/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3593 - loss: 4.7019
    Epoch 769/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4714 - loss: 4.1216
    Epoch 770/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3953 - loss: 4.5822
    Epoch 771/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4353 - loss: 4.3194
    Epoch 772/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4029 - loss: 4.5728
    Epoch 773/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3715 - loss: 4.7315
    Epoch 774/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4379 - loss: 4.2583
    Epoch 775/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3912 - loss: 4.6003
    Epoch 776/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4179 - loss: 4.4250
    Epoch 777/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3598 - loss: 4.8418
    Epoch 778/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4230 - loss: 4.3404
    Epoch 779/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.4050 - loss: 4.4387
    Epoch 780/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4421 - loss: 4.2662
    Epoch 781/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3986 - loss: 4.4802
    Epoch 782/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3760 - loss: 4.6569
    Epoch 783/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.3755 - loss: 4.6034
    Epoch 784/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4225 - loss: 4.3980
    Epoch 785/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3641 - loss: 4.6793
    Epoch 786/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3837 - loss: 4.5458
    Epoch 787/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4149 - loss: 4.4541
    Epoch 788/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.4414 - loss: 4.3241
    Epoch 789/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3951 - loss: 4.4840
    Epoch 790/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.4224 - loss: 4.4266
    Epoch 791/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3881 - loss: 4.6637
    Epoch 792/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3533 - loss: 4.8431
    Epoch 793/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.4034 - loss: 4.5012
    Epoch 794/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - accuracy: 0.3969 - loss: 4.5700
    Epoch 795/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - accuracy: 0.4119 - loss: 4.4039
    Epoch 796/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - accuracy: 0.3571 - loss: 4.7962
    Epoch 797/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.4204 - loss: 4.4110
    Epoch 798/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3744 - loss: 4.7069
    Epoch 799/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - accuracy: 0.3612 - loss: 4.7861
    Epoch 800/800
    [1m50/50[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.3975 - loss: 4.4652



```python
plt.plot(model.history.history['loss'])
```




    [<matplotlib.lines.Line2D at 0x7fd8bad49b50>]




    
![png](video-compression-new_files/video-compression-new_8_1.png)
    



```python
p_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
```


```python
tf.config.list_physical_devices()
```




    [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]




```python
x = np.ndindex(5, 5)
list(x)
print(list(x))
```

    []



```python
p_model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ sequential (<span style="color: #0087ff; text-decoration-color: #0087ff">Sequential</span>)         │ ?                      │       <span style="color: #00af00; text-decoration-color: #00af00">173,724</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ softmax (<span style="color: #0087ff; text-decoration-color: #0087ff">Softmax</span>)               │ ?                      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">173,724</span> (678.61 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">173,724</span> (678.61 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
def compress(M: tf.keras.Model, data: np.ndarray, **kwargs) -> bytes:
    encoder = ArithmeticEncoder(32, stream := BitOutputStream(io.BytesIO()))
    dataset = ContextDataset(data, **kwargs)
    frame_shape = data.shape[2:]
    for n, t in tqdm.tqdm(list(np.ndindex(*data.shape[:2]))):
        # print(n, t)
        raw_freqs = tf.nn.softmax(M(np.stack([dataset.get_context((n, t, x, y))
                    for x, y in np.ndindex(*frame_shape)], axis=0)), axis=1).numpy().reshape(frame_shape + (1024,))
        # raw_freqs = tf.nn.softmax(np.random.normal(0, 1, (8 * 16, 1024)), axis=1).numpy().reshape(frame_shape + (1024,))
        scaled_freqs = (raw_freqs * (2 ** 16)).round().astype(int) + 1
        # print(raw_freqs.shape)
        # print(np.sum(raw_freqs, axis=1))
        for x, y in np.ndindex(*frame_shape):
            freqs = SimpleFrequencyTable(scaled_freqs[x, y]) # TODO
            # print(raw_freqs[x, y].max())
            # print(type(int(data[n, t, x, y])))
            encoder.write(freqs, data[n, t, x, y])
    encoder.finish()
    return stream.output.getvalue()

s = get_data(5)
for i in range(1):
    print(len(c := compress(model, s[i:i+1, :1200], t=5)))
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1200/1200 [01:21<00:00, 14.65it/s]

    144549


    



```python
model.save_weights('./model_checkpoint.weights.h5')
```


```python
len(_)
```




    107795




```python
# get_data(1).dtype
# freqs
# get_data(1)[0, 0, 0, 0] ^ 20
np.array([1.1, 2.2, 3.3]).round().astype(int)
```




    array([1, 2, 3])




```python
model.history
```




    <keras.src.callbacks.history.History at 0x7ff001dd1b90>




```python
d = ContextDataset(get_data(100), (1200, 8, 16), 3, 10, True, 32, shuffle=True, limit=3000)
model.evaluate(ds)
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[18], line 1
    ----> 1 d = ContextDataset(get_data(100), (1200, 8, 16), 3, 10, True, 32, shuffle=True, limit=3000)
          2 model.evaluate(ds)


    Cell In[3], line 3, in get_data(n)
          2 def get_data(n: int = 10) -> np.ndarray:
    ----> 3     return np.stack([np.load('./data/' + p) for p in ds['0'][:n]['path']])


    Cell In[12], line 53, in ContextDataset.__getitem__(self, idx)
         51 def __getitem__(self, idx: int):
         52     if self.shuffle:
    ---> 53         idx = self.p[idx]
         54     t_, height, width = self.dims
         55     window_shape = (1, self.t, self.k*2+1, self.k*2+1)


    IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices



```python
ds[1]
```




    (array([[[[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 0, ..., 1, 0, 0],
               [0, 1, 1, ..., 1, 1, 1],
               [0, 1, 0, ..., 1, 1, 0],
               ...,
               [0, 1, 0, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 1, 0],
               [0, 0, 0, ..., 0, 1, 0],
               [0, 0, 1, ..., 0, 1, 1],
               ...,
               [0, 1, 0, ..., 1, 0, 0],
               [0, 0, 0, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 0, 1],
               ...,
               [0, 1, 0, ..., 1, 0, 1],
               [0, 0, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 1, ..., 1, 1, 1],
               [0, 1, 1, ..., 1, 1, 1],
               [0, 1, 0, ..., 0, 1, 0],
               ...,
               [0, 1, 0, ..., 1, 0, 1],
               [0, 1, 1, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 1, 0],
               [0, 0, 1, ..., 1, 1, 0],
               [0, 0, 1, ..., 1, 0, 0],
               ...,
               [0, 1, 0, ..., 1, 0, 0],
               [0, 0, 0, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 1, 1, 0],
               ...,
               [0, 1, 0, ..., 1, 0, 1],
               [0, 0, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 1, ..., 1, 1, 0],
               [0, 1, 0, ..., 1, 0, 0],
               [0, 0, 1, ..., 1, 1, 0],
               ...,
               [0, 0, 0, ..., 0, 0, 1],
               [0, 0, 1, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 1, 0],
               [0, 0, 1, ..., 1, 1, 0],
               [0, 0, 1, ..., 1, 0, 0],
               ...,
               [0, 1, 0, ..., 1, 0, 0],
               [0, 1, 1, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 1, 1, 0],
               ...,
               [0, 1, 0, ..., 1, 0, 1],
               [0, 1, 0, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             ...,
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 0, 1, 1],
               [0, 1, 1, ..., 1, 1, 1],
               [0, 0, 0, ..., 1, 0, 1],
               ...,
               [0, 0, 0, ..., 0, 0, 1],
               [0, 1, 1, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 1, 0],
               [0, 0, 1, ..., 1, 1, 0],
               [0, 0, 0, ..., 1, 1, 0],
               ...,
               [0, 1, 0, ..., 1, 0, 0],
               [0, 1, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 0, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 1, 1, 0],
               ...,
               [0, 1, 0, ..., 1, 0, 1],
               [0, 1, 0, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 0, 1, 1],
               [0, 0, 0, ..., 1, 1, 1],
               [0, 1, 0, ..., 0, 0, 1],
               ...,
               [0, 0, 0, ..., 0, 0, 1],
               [0, 1, 1, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 1, 0, 1],
               [0, 0, 1, ..., 1, 1, 0],
               [0, 1, 1, ..., 1, 1, 0],
               ...,
               [0, 0, 0, ..., 0, 1, 0],
               [0, 1, 1, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 0, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 1, 0, 0],
               ...,
               [0, 1, 0, ..., 1, 0, 1],
               [0, 0, 1, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 1, 0, 1],
               [0, 1, 0, ..., 0, 1, 0],
               ...,
               [0, 0, 0, ..., 0, 0, 1],
               [0, 1, 1, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 1, 0, 0],
               [0, 0, 1, ..., 1, 1, 0],
               [0, 0, 0, ..., 1, 1, 0],
               ...,
               [0, 1, 0, ..., 0, 1, 1],
               [0, 1, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 0, 0],
               [0, 1, 0, ..., 0, 1, 1],
               [0, 0, 0, ..., 1, 1, 0],
               ...,
               [0, 1, 0, ..., 1, 0, 1],
               [0, 1, 0, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]]],
     
     
     
            [[[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 1, ..., 1, 0, 0],
               [0, 0, 0, ..., 1, 1, 0],
               [0, 1, 0, ..., 1, 1, 0],
               ...,
               [0, 1, 1, ..., 0, 0, 0],
               [0, 0, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 0, 1],
               [0, 1, 1, ..., 1, 0, 0],
               [0, 0, 0, ..., 0, 0, 1],
               ...,
               [0, 1, 0, ..., 0, 1, 0],
               [0, 1, 0, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 1, 0],
               [0, 1, 1, ..., 1, 1, 0],
               [0, 0, 0, ..., 0, 0, 0],
               ...,
               [0, 1, 1, ..., 1, 0, 1],
               [0, 0, 0, ..., 1, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 1, ..., 1, 0, 0],
               [0, 0, 0, ..., 1, 1, 0],
               [0, 1, 0, ..., 1, 0, 0],
               ...,
               [0, 0, 1, ..., 0, 0, 1],
               [0, 0, 1, ..., 0, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 0, 1],
               [0, 1, 1, ..., 1, 0, 0],
               [0, 0, 0, ..., 0, 0, 1],
               ...,
               [0, 0, 0, ..., 1, 0, 1],
               [0, 0, 1, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 1, 0, 0],
               [0, 1, 1, ..., 1, 1, 0],
               [0, 0, 0, ..., 0, 0, 1],
               ...,
               [0, 1, 0, ..., 1, 1, 1],
               [0, 0, 1, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 0, 0, 0],
               [0, 0, 0, ..., 1, 1, 0],
               [0, 1, 0, ..., 0, 1, 0],
               ...,
               [0, 0, 0, ..., 1, 0, 0],
               [0, 1, 1, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 0, 1],
               [0, 1, 1, ..., 1, 0, 0],
               [0, 0, 0, ..., 0, 0, 1],
               ...,
               [0, 1, 1, ..., 1, 1, 0],
               [0, 0, 1, ..., 1, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 1, 0, 0],
               [0, 0, 1, ..., 0, 0, 1],
               [0, 1, 1, ..., 0, 0, 0],
               ...,
               [0, 0, 1, ..., 0, 1, 0],
               [0, 1, 0, ..., 1, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             ...,
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 1, 0, 0],
               [0, 1, 0, ..., 1, 1, 1],
               [0, 0, 0, ..., 1, 0, 1],
               ...,
               [0, 1, 1, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 1, 1],
               [0, 1, 0, ..., 1, 1, 0],
               [0, 0, 1, ..., 0, 0, 1],
               ...,
               [0, 1, 0, ..., 0, 1, 0],
               [0, 0, 0, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 1, 0],
               [0, 1, 0, ..., 0, 1, 0],
               ...,
               [0, 0, 1, ..., 1, 0, 1],
               [0, 0, 1, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 1, 0, 0],
               [0, 0, 0, ..., 1, 1, 0],
               [0, 0, 1, ..., 0, 0, 1],
               ...,
               [0, 1, 1, ..., 0, 0, 1],
               [0, 1, 1, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 1, 1],
               [0, 1, 0, ..., 1, 1, 0],
               [0, 0, 1, ..., 0, 0, 1],
               ...,
               [0, 1, 0, ..., 0, 1, 0],
               [0, 1, 0, ..., 0, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 0, 0, 0],
               [0, 1, 0, ..., 0, 1, 1],
               [0, 1, 0, ..., 0, 1, 0],
               ...,
               [0, 0, 1, ..., 0, 0, 1],
               [0, 0, 1, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 1, 0, 0],
               [0, 1, 1, ..., 0, 0, 1],
               [0, 0, 0, ..., 1, 1, 1],
               ...,
               [0, 1, 0, ..., 0, 1, 1],
               [0, 1, 0, ..., 1, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 0, 1],
               [0, 1, 1, ..., 1, 0, 1],
               [0, 0, 1, ..., 0, 0, 1],
               ...,
               [0, 0, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 0, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 0, 0, 0],
               [0, 0, 1, ..., 0, 0, 0],
               [0, 0, 1, ..., 1, 1, 0],
               ...,
               [0, 1, 1, ..., 0, 0, 1],
               [0, 0, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]]],
     
     
     
            [[[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 0, 1, 1],
               [0, 1, 1, ..., 1, 1, 1],
               ...,
               [0, 1, 0, ..., 0, 0, 1],
               [0, 1, 0, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 1, 0, 1],
               [0, 1, 0, ..., 0, 0, 0],
               [0, 1, 0, ..., 0, 0, 0],
               ...,
               [0, 0, 1, ..., 0, 1, 1],
               [0, 1, 0, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 1, 1],
               [0, 1, 0, ..., 0, 1, 0],
               [0, 1, 1, ..., 1, 0, 0],
               ...,
               [0, 0, 1, ..., 0, 0, 0],
               [0, 0, 0, ..., 1, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 1, 1, 1],
               [0, 1, 1, ..., 1, 1, 1],
               [0, 1, 0, ..., 0, 1, 0],
               ...,
               [0, 1, 0, ..., 0, 0, 1],
               [0, 0, 1, ..., 0, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 0, 1],
               [0, 1, 1, ..., 1, 1, 1],
               [0, 1, 0, ..., 0, 0, 1],
               ...,
               [0, 0, 0, ..., 1, 0, 0],
               [0, 1, 0, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 1, ..., 1, 1, 0],
               [0, 1, 0, ..., 0, 0, 1],
               [0, 1, 1, ..., 1, 0, 0],
               ...,
               [0, 0, 1, ..., 1, 1, 0],
               [0, 1, 0, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 1, 1, 0],
               [0, 1, 1, ..., 1, 1, 1],
               [0, 1, 1, ..., 1, 0, 0],
               ...,
               [0, 1, 0, ..., 1, 0, 0],
               [0, 1, 0, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 1, 0, 0],
               [0, 0, 0, ..., 1, 0, 1],
               [0, 1, 0, ..., 0, 0, 1],
               ...,
               [0, 1, 0, ..., 0, 1, 0],
               [0, 1, 0, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 0, 0, 1],
               [0, 1, 0, ..., 0, 0, 1],
               [0, 1, 1, ..., 1, 0, 0],
               ...,
               [0, 0, 1, ..., 0, 0, 0],
               [0, 1, 1, ..., 0, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             ...,
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 1, ..., 1, 1, 1],
               [0, 0, 0, ..., 0, 0, 0],
               [0, 0, 1, ..., 0, 0, 1],
               ...,
               [0, 1, 0, ..., 0, 0, 1],
               [0, 1, 1, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 1, 0],
               [0, 1, 0, ..., 0, 1, 0],
               [0, 0, 0, ..., 0, 1, 0],
               ...,
               [0, 1, 1, ..., 0, 1, 1],
               [0, 0, 0, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 1, ..., 1, 1, 0],
               [0, 0, 0, ..., 1, 0, 1],
               [0, 0, 1, ..., 0, 1, 1],
               ...,
               [0, 1, 1, ..., 1, 1, 0],
               [0, 1, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 0, 0, 0],
               [0, 1, 0, ..., 1, 1, 1],
               [0, 1, 1, ..., 1, 0, 1],
               ...,
               [0, 1, 0, ..., 1, 1, 0],
               [0, 1, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 1, 1, 1],
               [0, 1, 0, ..., 0, 0, 1],
               [0, 1, 0, ..., 1, 1, 1],
               ...,
               [0, 0, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 1, ..., 1, 1, 0],
               [0, 1, 1, ..., 0, 1, 1],
               [0, 0, 1, ..., 0, 1, 1],
               ...,
               [0, 0, 1, ..., 1, 0, 0],
               [0, 1, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 1, 0, 1],
               [0, 1, 1, ..., 1, 0, 0],
               [0, 1, 0, ..., 1, 1, 0],
               ...,
               [0, 1, 1, ..., 0, 1, 0],
               [0, 1, 0, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 1, 1],
               [0, 1, 0, ..., 1, 1, 1],
               ...,
               [0, 1, 1, ..., 0, 1, 1],
               [0, 0, 0, ..., 0, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 0, 1, 1],
               [0, 0, 0, ..., 1, 0, 1],
               [0, 0, 1, ..., 0, 1, 1],
               ...,
               [0, 0, 1, ..., 1, 1, 1],
               [0, 0, 0, ..., 1, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]]],
     
     
     
            ...,
     
     
     
            [[[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 0, 0, 0],
               [0, 1, 1, ..., 0, 0, 1],
               [0, 0, 0, ..., 0, 0, 1],
               ...,
               [0, 1, 1, ..., 0, 0, 1],
               [0, 0, 1, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 1, 0, 1],
               [0, 0, 1, ..., 0, 1, 0],
               [0, 1, 1, ..., 1, 0, 0],
               ...,
               [0, 1, 1, ..., 0, 0, 1],
               [0, 1, 1, ..., 1, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 0, 1, 1],
               [0, 0, 1, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 1, 0],
               ...,
               [0, 1, 1, ..., 1, 0, 1],
               [0, 1, 1, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 1, 1, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 1, 1],
               ...,
               [0, 1, 1, ..., 0, 0, 0],
               [0, 0, 0, ..., 1, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 1, 0, 1],
               [0, 0, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 1, 1, 1],
               ...,
               [0, 1, 1, ..., 0, 0, 1],
               [0, 1, 1, ..., 1, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 0, 1, 1],
               [0, 1, 0, ..., 1, 0, 0],
               [0, 0, 1, ..., 1, 1, 1],
               ...,
               [0, 0, 0, ..., 0, 0, 0],
               [0, 1, 0, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 1, 1, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [0, 0, 1, ..., 1, 0, 0],
               ...,
               [0, 1, 1, ..., 0, 0, 1],
               [0, 1, 1, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 1, 0, 1],
               [0, 1, 1, ..., 0, 1, 1],
               [0, 1, 1, ..., 1, 0, 0],
               ...,
               [0, 1, 1, ..., 0, 1, 0],
               [0, 1, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 0, 1, 1],
               [0, 0, 0, ..., 1, 0, 1],
               [0, 0, 0, ..., 1, 0, 0],
               ...,
               [0, 0, 0, ..., 0, 0, 0],
               [0, 1, 1, ..., 0, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             ...,
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 1, 0, 1],
               [0, 1, 0, ..., 0, 1, 1],
               [0, 1, 1, ..., 1, 0, 0],
               ...,
               [0, 1, 0, ..., 0, 0, 1],
               [0, 0, 1, ..., 1, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 1, ..., 1, 1, 0],
               [0, 0, 1, ..., 0, 1, 0],
               [0, 1, 0, ..., 0, 1, 1],
               ...,
               [0, 0, 1, ..., 0, 0, 1],
               [0, 1, 1, ..., 1, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 1, ..., 1, 1, 0],
               [0, 0, 1, ..., 0, 0, 0],
               [0, 0, 1, ..., 1, 1, 1],
               ...,
               [0, 0, 1, ..., 1, 1, 0],
               [0, 1, 0, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 0, ..., 1, 1, 1],
               [0, 1, 0, ..., 0, 1, 1],
               [0, 1, 0, ..., 1, 0, 1],
               ...,
               [0, 1, 0, ..., 1, 0, 1],
               [0, 1, 0, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 1, 0, 1],
               [0, 1, 0, ..., 0, 0, 1],
               [0, 0, 1, ..., 0, 0, 1],
               ...,
               [0, 0, 1, ..., 1, 0, 1],
               [0, 1, 1, ..., 1, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 1, 1, 0],
               [0, 1, 0, ..., 0, 0, 1],
               [0, 1, 0, ..., 0, 0, 1],
               ...,
               [0, 1, 0, ..., 1, 0, 1],
               [0, 0, 0, ..., 1, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 0, ..., 1, 1, 1],
               [0, 1, 0, ..., 0, 1, 1],
               [0, 0, 0, ..., 1, 0, 1],
               ...,
               [0, 1, 0, ..., 0, 0, 0],
               [0, 1, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 1, 1, 0],
               [0, 0, 1, ..., 1, 1, 1],
               [0, 1, 0, ..., 1, 1, 0],
               ...,
               [0, 1, 0, ..., 1, 1, 1],
               [0, 1, 1, ..., 1, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 1, 1, 0],
               [0, 1, 0, ..., 1, 0, 0],
               [0, 1, 1, ..., 1, 0, 0],
               ...,
               [0, 0, 1, ..., 0, 0, 0],
               [0, 0, 1, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]]]],
     
     
     
            [[[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 1, 1, 0],
               [0, 1, 0, ..., 1, 1, 1],
               [0, 1, 0, ..., 1, 1, 0],
               ...,
               [0, 1, 0, ..., 0, 0, 0],
               [0, 0, 1, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 1, 0],
               [0, 1, 0, ..., 1, 1, 0],
               [0, 1, 0, ..., 1, 0, 1],
               ...,
               [0, 1, 0, ..., 1, 1, 0],
               [0, 0, 0, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 1, 0, 0],
               [0, 0, 1, ..., 0, 0, 1],
               [0, 1, 1, ..., 0, 0, 1],
               ...,
               [0, 0, 0, ..., 0, 0, 0],
               [0, 1, 0, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 1, 1, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [0, 1, 0, ..., 1, 1, 0],
               ...,
               [0, 0, 1, ..., 0, 0, 1],
               [0, 0, 1, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 0, 0],
               ...,
               [0, 0, 0, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 1, 1, 1],
               [0, 0, 1, ..., 0, 0, 1],
               [0, 0, 1, ..., 1, 0, 0],
               ...,
               [0, 1, 0, ..., 0, 1, 1],
               [0, 1, 0, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 1, 0, 0],
               [0, 1, 0, ..., 1, 0, 1],
               ...,
               [0, 0, 0, ..., 1, 0, 0],
               [0, 0, 0, ..., 1, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 0, 1],
               [0, 0, 1, ..., 0, 0, 1],
               [0, 0, 1, ..., 1, 0, 1],
               ...,
               [0, 0, 0, ..., 0, 0, 1],
               [0, 0, 0, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 1, ..., 1, 0, 1],
               [0, 0, 1, ..., 0, 0, 1],
               [0, 0, 0, ..., 0, 1, 1],
               ...,
               [0, 0, 0, ..., 0, 0, 0],
               [0, 0, 1, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             ...,
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 1, 1, 1],
               [0, 1, 1, ..., 1, 0, 0],
               [0, 1, 1, ..., 1, 0, 0],
               ...,
               [0, 0, 1, ..., 1, 1, 1],
               [0, 0, 1, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 1, 1, 0],
               [0, 1, 0, ..., 1, 0, 0],
               [0, 0, 1, ..., 0, 0, 0],
               ...,
               [0, 1, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 1, 1],
               ...,
               [0, 0, 0, ..., 0, 1, 1],
               [0, 0, 1, ..., 1, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 1, ..., 1, 1, 1],
               [0, 0, 1, ..., 0, 0, 1],
               [0, 0, 1, ..., 0, 0, 1],
               ...,
               [0, 1, 0, ..., 0, 0, 1],
               [0, 0, 1, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 0, 1],
               [0, 1, 0, ..., 0, 0, 1],
               [0, 1, 0, ..., 1, 0, 1],
               ...,
               [0, 0, 1, ..., 0, 1, 1],
               [0, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 0, 0],
               [0, 1, 0, ..., 1, 1, 1],
               [0, 0, 0, ..., 0, 1, 1],
               ...,
               [0, 0, 1, ..., 0, 0, 1],
               [0, 0, 0, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 0, 1, 1],
               [0, 0, 1, ..., 0, 0, 0],
               [0, 0, 1, ..., 1, 1, 1],
               ...,
               [0, 1, 0, ..., 0, 0, 0],
               [0, 1, 0, ..., 0, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 0, ..., 0, 0, 1],
               [0, 1, 0, ..., 1, 0, 0],
               [0, 0, 0, ..., 1, 0, 1],
               ...,
               [0, 0, 1, ..., 0, 0, 1],
               [0, 1, 0, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 1, ..., 0, 0, 1],
               [0, 0, 0, ..., 1, 0, 0],
               [0, 0, 0, ..., 0, 1, 1],
               ...,
               [0, 0, 0, ..., 0, 1, 0],
               [0, 0, 0, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]]],
     
     
     
            [[[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 1, 1, 1],
               [0, 0, 0, ..., 1, 1, 1],
               [0, 0, 0, ..., 1, 0, 1],
               ...,
               [0, 0, 0, ..., 0, 1, 0],
               [0, 0, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 1, ..., 1, 1, 1],
               [0, 1, 0, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 1, 0],
               ...,
               [0, 0, 0, ..., 1, 1, 0],
               [0, 1, 0, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 1, 1],
               [0, 0, 0, ..., 0, 1, 1],
               [0, 1, 1, ..., 0, 0, 0],
               ...,
               [0, 1, 1, ..., 1, 0, 1],
               [0, 0, 0, ..., 1, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 1, 1, ..., 1, 1, 0],
               [0, 0, 0, ..., 1, 1, 1],
               [0, 1, 1, ..., 1, 1, 0],
               ...,
               [0, 1, 1, ..., 0, 0, 0],
               [0, 0, 1, ..., 1, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 0, 0, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [0, 1, 0, ..., 1, 0, 1],
               ...,
               [0, 0, 1, ..., 0, 0, 1],
               [0, 0, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 1, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 0, 1],
               [0, 0, 1, ..., 0, 1, 1],
               ...,
               [0, 0, 1, ..., 1, 1, 1],
               [0, 0, 0, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 1, 1, 0],
               [0, 0, 0, ..., 0, 1, 0],
               [0, 0, 1, ..., 0, 1, 0],
               ...,
               [0, 1, 0, ..., 1, 1, 1],
               [0, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 1, 0, 0],
               [0, 1, 1, ..., 0, 1, 0],
               [0, 0, 1, ..., 1, 0, 0],
               ...,
               [0, 0, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 1, 0, 0],
               [0, 0, 0, ..., 0, 0, 1],
               [0, 0, 1, ..., 0, 1, 1],
               ...,
               [0, 1, 1, ..., 1, 0, 1],
               [0, 0, 0, ..., 0, 0, 1],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             ...,
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 1, ..., 1, 0, 0],
               [0, 0, 1, ..., 0, 1, 1],
               [0, 0, 0, ..., 0, 1, 1],
               ...,
               [0, 0, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 1, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 1, ..., 1, 1, 0],
               [0, 1, 0, ..., 0, 1, 0],
               [0, 1, 0, ..., 0, 1, 0],
               ...,
               [0, 1, 1, ..., 1, 1, 0],
               [0, 0, 0, ..., 0, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 1, 0, 0],
               [0, 1, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 1, 0],
               ...,
               [0, 1, 0, ..., 1, 0, 0],
               [0, 0, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 0, 1, 0],
               [0, 0, 0, ..., 1, 0, 0],
               [0, 0, 0, ..., 0, 1, 1],
               ...,
               [0, 0, 1, ..., 1, 1, 0],
               [0, 1, 0, ..., 1, 1, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 0, ..., 1, 0, 1],
               [0, 1, 0, ..., 0, 1, 0],
               [0, 0, 1, ..., 1, 0, 1],
               ...,
               [0, 1, 0, ..., 1, 0, 1],
               [0, 1, 1, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 1, 0, 0],
               [0, 0, 1, ..., 1, 1, 1],
               [0, 1, 1, ..., 0, 1, 1],
               ...,
               [0, 0, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]],
     
     
             [[[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               ...,
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              ...,
     
              [[0, 0, 0, ..., 0, 1, 0],
               [0, 1, 1, ..., 0, 0, 0],
               [0, 1, 0, ..., 1, 1, 1],
               ...,
               [0, 0, 1, ..., 1, 0, 0],
               [0, 1, 1, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 0, 1, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 1, 0],
               [0, 1, 1, ..., 1, 0, 1],
               ...,
               [0, 1, 0, ..., 1, 0, 1],
               [0, 1, 1, ..., 1, 1, 1],
               [1, 0, 0, ..., 0, 0, 0]],
     
              [[0, 1, 1, ..., 1, 0, 0],
               [0, 1, 1, ..., 0, 1, 1],
               [0, 0, 1, ..., 1, 0, 0],
               ...,
               [0, 1, 1, ..., 0, 1, 0],
               [0, 0, 0, ..., 1, 0, 0],
               [1, 0, 0, ..., 0, 0, 0]]]]]),
     array([948, 566, 638, 360, 946, 365, 105, 709, 676, 119, 303, 397, 377,
            802,  75, 838, 956, 676,   2, 425, 613, 752, 642, 738, 764, 153,
            746, 197, 758,   0, 560, 220], dtype=int16))




```python
tf.constant(xs)
```




    <tf.Tensor: shape=(153600, 10, 5, 5), dtype=float64, numpy=
    array([[[[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            ...,
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]]],
    
    
           [[[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            ...,
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0., 811., 793., 251.],
             [  0.,   0., 908., 713., 136.],
             [  0.,   0.,  51., 327., 939.]]],
    
    
           [[[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            ...,
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0., 811., 793., 251.],
             [  0.,   0., 908., 713., 136.],
             [  0.,   0.,  51., 327., 939.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0., 476., 793., 251.],
             [  0.,   0., 908., 713., 136.],
             [  0.,   0.,  51., 327., 939.]]],
    
    
           ...,
    
    
           [[[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            ...,
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]]],
    
    
           [[[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            ...,
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]]],
    
    
           [[[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            ...,
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]],
    
            [[  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.],
             [  0.,   0.,   0.,   0.,   0.]]]])>




```python
model(d[0])
```




    <tf.Tensor: shape=(16, 1024), dtype=float32, numpy=
    array([[ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [-0.01412737, -0.0019212 , -0.05573326, ...,  0.04152815,
            -0.04518569,  0.00353482],
           [-0.01141175,  0.0176482 , -0.10198461, ...,  0.09202049,
            -0.05361081,  0.00360572],
           ...,
           [ 0.03148152,  0.08828445, -0.15718943, ..., -0.0191272 ,
            -0.08934174, -0.01622619],
           [ 0.06562139,  0.12046035, -0.12340311, ..., -0.00264132,
            -0.18297282,  0.01407764],
           [ 0.05176689,  0.11401519, -0.09553595, ..., -0.0093105 ,
            -0.22289763, -0.00746004]], dtype=float32)>




```python
# ContextDataset(get_data(1), (1200, 8, 16), 3, 10, False, 1600)[0]
```


```python
d = ContextDataset(get_data(1), (1200, 8, 16), 3, 10, False, 16)
print(len(d))
z = d[0]#[768000//16-10]
# plt.imshow(z[2].reshape((10, 25)))
print(z[1])
plt.imshow(z[0][2, 1])
```

    9600
    [811 476 476 476 811 811 811 811 738 811 644 644 644 644 644 644]





    <matplotlib.image.AxesImage at 0x7f0b88eb9e50>




    
![png](video-compression-new_files/video-compression-new_23_2.png)
    



```python
io.BytesIO().write(b'test')
```




    4




```python
x = ArithmeticEncoder(12, stream := BitOutputStream(io.BytesIO()))
# x = ArithmeticEncoder(32, stream := BitOutputStream(open('./test-stream', 'wb')))
# s = 'an anabolic abacus and an apple ascend ' * 10
s = 'ab' * 40
s_ = list(set(s))
data = [s_.index(c) for c in s]
freqs = SimpleFrequencyTable([s.count(s_[i]) for i in range(len(s_))])
# freqs = SimpleFrequencyTable([1] * len(s_))
for c in data:
    x.write(freqs, c)
x.finish()
```


```python
s_
```




    [' ', 'd', 'u', 's', 'n', 'b', 'c', 'a', 'p', 'e', 'l']




```python
print(freqs.frequencies)
print(data)
```

    [30, 10, 10, 10, 20, 10, 10, 50, 20, 10, 10]
    [7, 5, 7, 6, 2, 3, 0, 7, 4, 1, 0, 7, 4, 0, 7, 8, 8, 10, 9, 7, 5, 7, 6, 2, 3, 0, 7, 4, 1, 0, 7, 4, 0, 7, 8, 8, 10, 9, 7, 5, 7, 6, 2, 3, 0, 7, 4, 1, 0, 7, 4, 0, 7, 8, 8, 10, 9, 7, 5, 7, 6, 2, 3, 0, 7, 4, 1, 0, 7, 4, 0, 7, 8, 8, 10, 9, 7, 5, 7, 6, 2, 3, 0, 7, 4, 1, 0, 7, 4, 0, 7, 8, 8, 10, 9, 7, 5, 7, 6, 2, 3, 0, 7, 4, 1, 0, 7, 4, 0, 7, 8, 8, 10, 9, 7, 5, 7, 6, 2, 3, 0, 7, 4, 1, 0, 7, 4, 0, 7, 8, 8, 10, 9, 7, 5, 7, 6, 2, 3, 0, 7, 4, 1, 0, 7, 4, 0, 7, 8, 8, 10, 9, 7, 5, 7, 6, 2, 3, 0, 7, 4, 1, 0, 7, 4, 0, 7, 8, 8, 10, 9, 7, 5, 7, 6, 2, 3, 0, 7, 4, 1, 0, 7, 4, 0, 7, 8, 8, 10, 9]



```python
# stream.close()
# print(stream.output.read())
print(r := stream.output.getvalue())
stream.output.seek(0)
y = ArithmeticDecoder(12, BitInputStream(stream.output))
r2 = ''
for i in range(80):
    # print(y.read(freqs))
    r2 += s_[y.read(freqs)]
print(r2)
```

    b'UUUUUUUUUU'
    abababababababababababababababababababababababababababababababababababababababab



```python
(d := ContextDataset(get_data(1), (1200, 8, 16), 2, 10, False, 16, limit=21))[1][0].shape
print(d.limit)
print(d.materialize()[0].shape)
print(d.materialize())
```

    21
    (336, 10, 5, 5)
    (array([[[[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            ...,
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]]],
    
    
           [[[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            ...,
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.79199219, 0.77441406, 0.24511719],
             [0.        , 0.        , 0.88671875, 0.69628906, 0.1328125 ],
             [0.        , 0.        , 0.04980469, 0.31933594, 0.91699219]]],
    
    
           [[[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            ...,
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.79199219, 0.77441406, 0.24511719],
             [0.        , 0.        , 0.88671875, 0.69628906, 0.1328125 ],
             [0.        , 0.        , 0.04980469, 0.31933594, 0.91699219]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.46484375, 0.77441406, 0.24511719],
             [0.        , 0.        , 0.88671875, 0.69628906, 0.1328125 ],
             [0.        , 0.        , 0.04980469, 0.31933594, 0.91699219]]],
    
    
           ...,
    
    
           [[[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.95996094, 0.43554688],
             [0.        , 0.        , 0.92871094, 0.6640625 , 0.62207031],
             [0.        , 0.        , 0.96777344, 0.10839844, 0.52929688]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.95996094, 0.43554688],
             [0.        , 0.        , 0.92871094, 0.58886719, 0.25292969],
             [0.        , 0.        , 0.96777344, 0.10839844, 0.52929688]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.95996094, 0.43554688],
             [0.        , 0.        , 0.92871094, 0.6640625 , 0.25292969],
             [0.        , 0.        , 0.96777344, 0.10839844, 0.52929688]],
    
            ...,
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.55273438, 0.78613281, 0.05761719],
             [0.        , 0.        , 0.92871094, 0.29980469, 0.88476562],
             [0.        , 0.        , 0.53320312, 0.59179688, 0.54492188]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.78613281, 0.05761719],
             [0.        , 0.        , 0.92871094, 0.29980469, 0.88476562],
             [0.        , 0.        , 0.53320312, 0.59179688, 0.54492188]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.55273438, 0.78613281, 0.05761719],
             [0.        , 0.        , 0.92871094, 0.29980469, 0.88476562],
             [0.        , 0.        , 0.53320312, 0.59179688, 0.31933594]]],
    
    
           [[[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.95996094, 0.43554688],
             [0.        , 0.        , 0.92871094, 0.58886719, 0.25292969],
             [0.        , 0.        , 0.96777344, 0.10839844, 0.52929688]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.95996094, 0.43554688],
             [0.        , 0.        , 0.92871094, 0.6640625 , 0.25292969],
             [0.        , 0.        , 0.96777344, 0.10839844, 0.52929688]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.95996094, 0.43554688],
             [0.        , 0.        , 0.92871094, 0.6640625 , 0.30566406],
             [0.        , 0.        , 0.96777344, 0.07324219, 0.52929688]],
    
            ...,
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.78613281, 0.05761719],
             [0.        , 0.        , 0.92871094, 0.29980469, 0.88476562],
             [0.        , 0.        , 0.53320312, 0.59179688, 0.54492188]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.55273438, 0.78613281, 0.05761719],
             [0.        , 0.        , 0.92871094, 0.29980469, 0.88476562],
             [0.        , 0.        , 0.53320312, 0.59179688, 0.31933594]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.78613281, 0.43554688],
             [0.        , 0.        , 0.48242188, 0.29980469, 0.88476562],
             [0.        , 0.        , 0.48144531, 0.59179688, 0.69140625]]],
    
    
           [[[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.95996094, 0.43554688],
             [0.        , 0.        , 0.92871094, 0.6640625 , 0.25292969],
             [0.        , 0.        , 0.96777344, 0.10839844, 0.52929688]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.95996094, 0.43554688],
             [0.        , 0.        , 0.92871094, 0.6640625 , 0.30566406],
             [0.        , 0.        , 0.96777344, 0.07324219, 0.52929688]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.78613281, 0.43554688],
             [0.        , 0.        , 0.92871094, 0.6640625 , 0.30566406],
             [0.        , 0.        , 0.96777344, 0.07324219, 0.52929688]],
    
            ...,
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.55273438, 0.78613281, 0.05761719],
             [0.        , 0.        , 0.92871094, 0.29980469, 0.88476562],
             [0.        , 0.        , 0.53320312, 0.59179688, 0.31933594]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.60058594, 0.78613281, 0.43554688],
             [0.        , 0.        , 0.48242188, 0.29980469, 0.88476562],
             [0.        , 0.        , 0.48144531, 0.59179688, 0.69140625]],
    
            [[0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        ],
             [0.        , 0.        , 0.55273438, 0.78613281, 0.05761719],
             [0.        , 0.        , 0.92871094, 0.29980469, 0.88476562],
             [0.        , 0.        , 0.48144531, 0.59179688, 0.39941406]]]]), array([[811.],
           [476.],
           [476.],
           [476.],
           [811.],
           [811.],
           [811.],
           [811.],
           [738.],
           [811.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [644.],
           [469.],
           [644.],
           [644.],
           [806.],
           [469.],
           [469.],
           [469.],
           [469.],
           [469.],
           [695.],
           [688.],
           [695.],
           [695.],
           [688.],
           [806.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [695.],
           [695.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [688.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [567.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [567.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [688.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [619.],
           [423.],
           [423.],
           [494.],
           [940.],
           [940.],
           [940.],
           [613.],
           [613.],
           [613.],
           [613.],
           [666.],
           [666.],
           [666.],
           [666.],
           [666.],
           [666.],
           [666.],
           [666.],
           [666.],
           [445.],
           [721.],
           [721.],
           [721.],
           [721.],
           [ 53.],
           [ 53.],
           [ 53.],
           [ 53.],
           [ 53.],
           [199.],
           [199.],
           [199.],
           [199.],
           [198.],
           [198.],
           [908.],
           [813.],
           [813.],
           [813.],
           [813.],
           [313.],
           [313.],
           [313.],
           [313.],
           [313.],
           [313.],
           [875.],
           [427.],
           [938.],
           [938.],
           [938.],
           [660.],
           [399.],
           [399.],
           [775.],
           [775.],
           [775.],
           [879.],
           [775.],
           [ 83.],
           [ 30.],
           [ 30.],
           [875.],
           [ 56.],
           [ 19.],
           [ 19.],
           [ 19.],
           [533.],
           [492.],
           [391.],
           [809.],
           [809.],
           [528.],
           [528.],
           [534.],
           [470.],
           [685.],
           [685.],
           [967.],
           [967.],
           [306.],
           [417.],
           [741.],
           [969.],
           [604.],
           [560.],
           [560.],
           [560.],
           [791.],
           [879.],
           [ 83.],
           [665.],
           [296.],
           [322.],
           [884.],
           [ 70.],
           [737.],
           [135.],
           [472.],
           [781.],
           [263.],
           [535.],
           [ 93.],
           [322.],
           [322.],
           [ 70.],
           [208.],
           [775.],
           [ 30.],
           [ 30.],
           [762.],
           [524.],
           [306.],
           [306.],
           [306.],
           [824.],
           [960.],
           [603.],
           [652.],
           [434.],
           [434.],
           [739.],
           [739.],
           [443.],
           [443.],
           [198.],
           [986.],
           [422.],
           [ 26.],
           [ 26.],
           [935.],
           [585.],
           [724.],
           [705.],
           [960.],
           [237.],
           [237.],
           [237.],
           [477.],
           [976.],
           [ 90.],
           [ 90.],
           [988.],
           [378.],
           [474.],
           [474.],
           [  7.],
           [982.],
           [304.],
           [304.],
           [429.],
           [531.],
           [531.],
           [855.],
           [128.],
           [740.],
           [487.],
           [538.],
           [470.],
           [873.],
           [427.],
           [875.],
           [779.],
           [354.],
           [136.],
           [429.],
           [429.],
           [537.],
           [795.],
           [107.],
           [638.],
           [746.],
           [746.],
           [182.],
           [ 22.],
           [ 22.],
           [397.],
           [397.],
           [237.],
           [997.],
           [237.],
           [991.],
           [472.],
           [751.],
           [136.],
           [284.],
           [284.],
           [284.],
           [284.],
           [284.],
           [ 93.],
           [ 93.],
           [ 93.],
           [ 93.],
           [ 93.],
           [ 93.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [615.],
           [566.],
           [615.],
           [566.],
           [615.],
           [566.],
           [566.]]))

