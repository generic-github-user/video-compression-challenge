```python
from pathlib import Path

list(Path('/').glob('./root/**/778a08cdcd6718b73373d7d5fa7c3104_9.npy'))

# tokens = np.load('~/.cache/huggingface/datasets/downloads/' + ds['0'][0]['path']) # first segment from the first data shard
```


```python
d3 = donut2
for i in range(3):
  d3 = np.diff(d3, n=1, axis=i%2)
plt.imshow(d3)
print(d3)
```

    [[ 0  0  0 ... -1  0  0]
     [ 0  0  0 ...  0  0  0]
     [ 0  0  0 ...  0  0  0]
     ...
     [ 0  1 -1 ...  0  2  0]
     [ 0 -2  2 ...  0 -1  0]
     [ 0  1 -1 ...  0  0  0]]



    
![png](video-compression_files/video-compression_1_1.png)
    



```python
xx
circle
```




    array([[20000, 19801, 19604, ..., 19409, 19604, 19801],
           [19801, 19602, 19405, ..., 19210, 19405, 19602],
           [19604, 19405, 19208, ..., 19013, 19208, 19405],
           ...,
           [19409, 19210, 19013, ..., 18818, 19013, 19210],
           [19604, 19405, 19208, ..., 19013, 19208, 19405],
           [19801, 19602, 19405, ..., 19210, 19405, 19602]])




```python
CompressedData = Tuple[np.ndarray, np.ndarray, np.ndarray]
```


```python
np.concatenate([[[1, 2, 3], [7, 8, 9]], [[4, 5, 6]]], axis=0)
```




    array([[1, 2, 3],
           [7, 8, 9],
           [4, 5, 6]])




```python
np.array([1, 2, 3]).argmax()
```




    2




```python
# @numba.njit
def compression_round(data: np.ndarray, d: np.ndarray, dims: int, sample_ratio: float = 1.0) -> CompressedData:
  # counts = Counter(itertools.chain.from_iterable(for (a, b) in itertools.product()))
  # frequencies = {((a, b), delta): count for ((a, b), delta, count)}
  u = np.unique(data[:, -1])
  # z = itertools.chain.from_iterable([((a, b), delta, count) for (delta, count) in
  #                                    zip(*np.unique(axis=0, return_counts=True))]
  #                                   for (a, b) in itertools.product(u, u))
  l = data.shape[0]
  if l == 1:
    return data, d, False
  if sample_ratio < 1.0:
    #  index = index[np.random.randint(l ** 2, size=int((l ** 2) * sample_ratio))]
    index = np.random.randint(l, size=(int((l ** 2) * sample_ratio), 2))
  else:
    index = np.array(list(np.ndindex(l, l)))
  # print('index: ', index[:10])
  # can we somehow use a convolution (or similar transform) to compute this more efficiently?
  y = np.stack([np.concatenate([data[j, :dims] - data[i, :dims], [data[i, -1]], [data[j, -1]]]) for i, j in index if i != j], axis=0)
  # print(y)
  z, counts = np.unique(y, axis=0, return_counts=True)
  if counts.max() == 1:
    return data, d, False

  best = z[counts.argmax()]
  # print(z, counts)
  # print(best)
  # print(data, d)
  print(best, counts.max())
  rule = best[2:]
  left, right = rule
  delta = best[:dims]
  # print(rule, delta)
  # for row in data[data[:, -1] == left]:
  new_rule = np.concatenate([[np.max(d[:, 0]) + 1], rule, delta])
  merged_mask = np.full((l,), True)
  for i in range(l):
    if (row := data[i])[-1] == left and merged_mask[i]:
        # print(row)
      # print(np.hstack([data[:, -1, np.newaxis], data[:, :dims] - row[:dims]]))
      mask = np.logical_not(np.logical_and(data[:, -1] == right, (data[:, :dims] - row[:dims] == delta).all(axis=1)))
      mask_size = np.logical_not(mask).astype(int).sum()
      assert mask_size <= 1, mask
      if mask_size == 1:
        # data = data[mask]
        merged_mask = np.logical_and(merged_mask, mask)
        data[i, -1] = new_rule[0]
        # l = data.shape[0] # TODO
  data = data[merged_mask]
  d = np.append(d, new_rule[np.newaxis, ...], axis=0)
  # d = d[np.isin(d[:, 0], data[:, -1])]
  return data, d, True
```


```python
q = np.array([[[1, 2, 3], [4, 5, 6]], [[3, 2, 3], [7, 5, 6]]])
print(np.diff(q, axis=0))
print(q[0, 0, :])
```

    [[[2 0 0]
      [3 0 0]]]
    [1 2 3]



```python
def compress_array(data: np.ndarray, difference: bool = True, **kwargs) -> CompressedData:
  dims = len(data.shape)
  if difference:
    for i in  range(dims):
      data = np.diff(data, n=1, axis=i)
  print(data)
  plt.imshow(data)
  # root =
  dx, dy = data.shape
  xs, ys = np.mgrid[:dx, :dy]
  unique = np.unique(data, return_inverse=True)[1].reshape((dx, dy))
  vals, idx = np.unique(data, return_index=True)

  data = np.column_stack((xs.ravel(), ys.ravel(), unique.ravel()))
  d = np.hstack((np.arange(0, vals.size)[..., np.newaxis],
                                      data.flatten()[idx, np.newaxis],
                                      np.full((vals.size, dims + 1), -1))).astype(int)
  print(data, d)
  flag = True
  for i in range(50):
    data, d, flag = compression_round(data, d, dims, **kwargs)
    if not flag:
      break
  return data, d

with np.printoptions(threshold=5000):
  # print(compress_array(np.full((20, 20), 5), sample_ratio=0.2))
  print(compress_array(donut, difference=True))
  # print(compress_array(d3))

# TODO: prune unused rules, flatten deeply nested rules (?)
# TODO: handle overlap between matches
# TODO: use MCTS?
```

    [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0. -1.  0.  1.  0.  0. -1.  0.  1.  0.  0. -1.  0.  1.]
     [ 0.  1.  0. -1.  0.  0.  1.  0. -1.  0.  0.  1.  0. -1.]
     [-1.  0.  0.  0.  1. -1.  0.  0.  0.  1. -1.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 1.  0.  0.  0. -1.  1.  0.  0.  0. -1.  1.  0.  0.  0.]
     [ 0. -1.  0.  1.  0.  0. -1.  0.  1.  0.  0. -1.  0.  1.]
     [ 0.  1.  0. -1.  0.  0.  1.  0. -1.  0.  0.  1.  0. -1.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    [[ 0  0  1]
     [ 0  1  1]
     [ 0  2  1]
     [ 0  3  1]
     [ 0  4  1]
     [ 0  5  1]
     [ 0  6  1]
     [ 0  7  1]
     [ 0  8  1]
     [ 0  9  1]
     [ 0 10  1]
     [ 0 11  1]
     [ 0 12  1]
     [ 0 13  1]
     [ 1  0  1]
     [ 1  1  1]
     [ 1  2  1]
     [ 1  3  1]
     [ 1  4  1]
     [ 1  5  1]
     [ 1  6  1]
     [ 1  7  1]
     [ 1  8  1]
     [ 1  9  1]
     [ 1 10  1]
     [ 1 11  1]
     [ 1 12  1]
     [ 1 13  1]
     [ 2  0  1]
     [ 2  1  1]
     [ 2  2  1]
     [ 2  3  1]
     [ 2  4  1]
     [ 2  5  1]
     [ 2  6  1]
     [ 2  7  1]
     [ 2  8  1]
     [ 2  9  1]
     [ 2 10  1]
     [ 2 11  1]
     [ 2 12  1]
     [ 2 13  1]
     [ 3  0  1]
     [ 3  1  1]
     [ 3  2  1]
     [ 3  3  1]
     [ 3  4  1]
     [ 3  5  1]
     [ 3  6  1]
     [ 3  7  1]
     [ 3  8  1]
     [ 3  9  1]
     [ 3 10  1]
     [ 3 11  1]
     [ 3 12  1]
     [ 3 13  1]
     [ 4  0  1]
     [ 4  1  0]
     [ 4  2  1]
     [ 4  3  2]
     [ 4  4  1]
     [ 4  5  1]
     [ 4  6  0]
     [ 4  7  1]
     [ 4  8  2]
     [ 4  9  1]
     [ 4 10  1]
     [ 4 11  0]
     [ 4 12  1]
     [ 4 13  2]
     [ 5  0  1]
     [ 5  1  2]
     [ 5  2  1]
     [ 5  3  0]
     [ 5  4  1]
     [ 5  5  1]
     [ 5  6  2]
     [ 5  7  1]
     [ 5  8  0]
     [ 5  9  1]
     [ 5 10  1]
     [ 5 11  2]
     [ 5 12  1]
     [ 5 13  0]
     [ 6  0  0]
     [ 6  1  1]
     [ 6  2  1]
     [ 6  3  1]
     [ 6  4  2]
     [ 6  5  0]
     [ 6  6  1]
     [ 6  7  1]
     [ 6  8  1]
     [ 6  9  2]
     [ 6 10  0]
     [ 6 11  1]
     [ 6 12  1]
     [ 6 13  1]
     [ 7  0  1]
     [ 7  1  1]
     [ 7  2  1]
     [ 7  3  1]
     [ 7  4  1]
     [ 7  5  1]
     [ 7  6  1]
     [ 7  7  1]
     [ 7  8  1]
     [ 7  9  1]
     [ 7 10  1]
     [ 7 11  1]
     [ 7 12  1]
     [ 7 13  1]
     [ 8  0  2]
     [ 8  1  1]
     [ 8  2  1]
     [ 8  3  1]
     [ 8  4  0]
     [ 8  5  2]
     [ 8  6  1]
     [ 8  7  1]
     [ 8  8  1]
     [ 8  9  0]
     [ 8 10  2]
     [ 8 11  1]
     [ 8 12  1]
     [ 8 13  1]
     [ 9  0  1]
     [ 9  1  0]
     [ 9  2  1]
     [ 9  3  2]
     [ 9  4  1]
     [ 9  5  1]
     [ 9  6  0]
     [ 9  7  1]
     [ 9  8  2]
     [ 9  9  1]
     [ 9 10  1]
     [ 9 11  0]
     [ 9 12  1]
     [ 9 13  2]
     [10  0  1]
     [10  1  2]
     [10  2  1]
     [10  3  0]
     [10  4  1]
     [10  5  1]
     [10  6  2]
     [10  7  1]
     [10  8  0]
     [10  9  1]
     [10 10  1]
     [10 11  2]
     [10 12  1]
     [10 13  0]
     [11  0  1]
     [11  1  1]
     [11  2  1]
     [11  3  1]
     [11  4  1]
     [11  5  1]
     [11  6  1]
     [11  7  1]
     [11  8  1]
     [11  9  1]
     [11 10  1]
     [11 11  1]
     [11 12  1]
     [11 13  1]
     [12  0  1]
     [12  1  1]
     [12  2  1]
     [12  3  1]
     [12  4  1]
     [12  5  1]
     [12  6  1]
     [12  7  1]
     [12  8  1]
     [12  9  1]
     [12 10  1]
     [12 11  1]
     [12 12  1]
     [12 13  1]
     [13  0  1]
     [13  1  1]
     [13  2  1]
     [13  3  1]
     [13  4  1]
     [13  5  1]
     [13  6  1]
     [13  7  1]
     [13  8  1]
     [13  9  1]
     [13 10  1]
     [13 11  1]
     [13 12  1]
     [13 13  1]] [[ 0  1 -1 -1 -1]
     [ 1  0 -1 -1 -1]
     [ 2  1 -1 -1 -1]]



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-8-d5c9b9d87871> in <cell line: 26>()
         26 with np.printoptions(threshold=5000):
         27   # print(compress_array(np.full((20, 20), 5), sample_ratio=0.2))
    ---> 28   print(compress_array(donut, difference=True))
         29   # print(compress_array(d3))
         30 


    <ipython-input-8-d5c9b9d87871> in compress_array(data, difference, **kwargs)
         19   flag = True
         20   for i in range(50):
    ---> 21     data, d, flag = compression_round(data, d, dims, **kwargs)
         22     if not flag:
         23       break


    NameError: name 'compression_round' is not defined



    
![png](video-compression_files/video-compression_8_2.png)
    



```python
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
# import gzip
```


```python
def make_donut(res: int = 25) -> np.ndarray:
  xx, yy = np.mgrid[:res, :res]
  r = res / 3
  r2 = 1
  k = 2
  circle = ((xx - res / 2) * 1.1) ** k + (yy - res / 2) ** k
  # donut = np.logical_and(circle < (r + r2) ** 2, circle > (r - r2) ** 2).astype(int)
  donut = (circle / 20).round() + 20 + np.random.normal(0, 0.3, (res, res)).round()
  donut = donut.astype(int)
  return donut

donut = make_donut()
plt.imshow(donut)

donut2 = np.tile(np.tile(donut, 2).T, 2).T
# plt.imshow(donut2)
```


    
![png](video-compression_files/video-compression_10_0.png)
    



```python
def generate_indices(shape: tuple) -> np.ndarray:
  if isinstance(shape, np.ndarray):
    shape = tuple(shape)
  return np.indices(shape).transpose(*range(1, (l := len(shape))+1), 0).reshape((np.array(shape).prod(), l))
# print(generate_indices((3, 3, 3))[1:] + 1)
```


```python
def segment(data: np.ndarray, w: Tuple[int, ...], ns: np.ndarray = None) -> np.ndarray:
  s = np.array(data.shape)
  w_ = np.array(w)
  if ns is None:
    ns = s / w_
  assert not (s % w_).any(), (data.shape, w)
  return sliding_window_view(data, w)[tuple(slice(None, None, j) for j in w)].reshape((int(ns.prod()), *w))
l = 4
# print(z := segment(np.arange(l ** 2).reshape((l, l)), s := (2, 1), np.array([l, l]) // np.array(s)), z.shape)
```


```python
# CompressionDict = Tuple[np.ndarray, np.ndarray]
# CompressionDict = dict[int, np.ndarray]
CompressionDict = list[np.ndarray]
Shape = Tuple[int, ...]
CompressedData = Tuple[np.ndarray, CompressionDict]
Compressed = Tuple[CompressedData, list[Tuple[Shape, Shape, int]]]

def compression_round_2(data: np.ndarray, d: CompressionDict, dims: int,
                        sample_ratio: float = 1.0,
                        window_size: int = 2,
                        window: Shape = None,
                        trial_windows: bool = False) -> Tuple[CompressedData, Shape, Shape, int]:
  # print(data, d, dims)
  # assumption: all window sizes reduce size of compressed data (but not dictionary) by the same amount
  if window is None:
    # window = tuple((np.eye(dims)[2] + 1).astype(int))
    window = (1, 4, 4, 4)
  def compress_window(w: Tuple[int, ...] = (1, 2)) -> Tuple[CompressedData, Shape, int]:
    w_ = np.array(w)
    padding = w_ - (np.array(data.shape) % w_)
    dp = np.pad(data, [(0, i) for i in padding], mode='reflect')
    new_shape = np.array(dp.shape) // w_
    windows = segment(dp, w, new_shape)
    # k, v = d
    # print(w, windows[:20])
    values, idx = np.unique(windows, return_inverse=True, axis=0)
    return ((idx2 := idx + (km := len(d))).reshape(tuple(new_shape)),
          # d + [list(x) for x in values.reshape((values.shape[0], -1))]
          d + list(values)
          ), tuple(padding), values.shape[0]
  # ws = [tuple((np.eye(dims)[i] + 1).astype(int)) for i in range(dims)]
  # TODO: restrict to size of data
  if trial_windows:
      ws = list(map(tuple, generate_indices(np.minimum(np.array(data.shape), np.full(dims, window_size)))[1:] + 1))
      ((data_, d_), p, n), w = min(zip(map(compress_window, ws), ws),
                 #  key=lambda x: len(x[0][1]))
                 key=lambda x: len(compress_bytes(x[0])))
  else:
      # default_w = (2,) * dims
      ((data_, d_), p, n), w = compress_window(window), window
  return (data_, d_), w, p, n

# Tuple[CompressedData, list[np.ndarray]]:
def compress_array_2(data: np.ndarray, difference: bool = True, **kwargs) -> Compressed:
  dims = len(data.shape)
  # TODO: pad (e.g., padding = v - (a.shape % v), (a.shape + padding) % v == <0, ...>)
  # TODO: early stopping
  # TODO: automatically test different shifts/offsets to better align patterns
    # (can also roll just rows/columns)
  # TODO: column permutations? other easily reversible transforms?
  # maybe: search for matching noise patterns that can be compressed to a single seed value
  # TODO: only store starting rule/pattern
  # TODO: use rule encodings that attempt to match actual values (so intra-round differencing is more effective) ?
  # TODO: denoising methods? (i.e., decompose into base signal and noise, compress separately, recombine)
  # TODO: pack metadata into single array for more efficient serialization
  # TODO: automatic refactoring of array code
  if difference:
    # TODO: re-include initial row so it is differenced along the other axis (so we don't need to include both in the metadata -- gains are amplified for e.g., 3D arrays with small cross-sections (I think))
    data = diff(data)
  # print(data)
  # rprint(compress_bytes(data))
  # plt.imshow(data); plt.show()
  # root =
  # dx, dy = data.shape
  # xs, ys = np.mgrid[:dx, :dy] # TODO

  u, idx = np.unique(data, return_inverse=True)
  data = idx.reshape(data.shape)
  d = list(u) # ?
  # d = dict(zip(d := np.unique(data), map(list, d.reshape((d.size, 1)))))
  q = np.inf
  ws = []
  for i in range(15):
    (data, d), w, p, n = compression_round_2(data, d, dims, **kwargs) # TODO: reindex
    # assert list(d.values())[-1].shape == w
    ws.append((w, p, (n,))) # TODO...
    # plt.imshow(data); plt.show()
    if (q2 := len(compress_bytes((data, d)))) >= q:
      break
    q = q2
  return d + [data], ws# + [(data.shape, (0, 0))]

def compress_bytes(data: Compressed) -> bytes:
  return zlib.compress(pickle.dumps(data), level=9)

def compress_2(data: np.ndarray, **kwargs) -> bytes:
  return compress_bytes(compress_array_2(data, **kwargs))

def pack_metadata(m: Compressed) -> np.ndarray:
  d, meta = m
  print(meta)
  meta_ = np.array([list(itertools.chain.from_iterable(x)) for x in meta]).flatten()
  dims = len(meta[0][0]) # TODO
  return np.concatenate([[dims, meta_.size], meta_, *(di.flatten() for di in d)], axis=0)

with np.printoptions(threshold=5000):
  print(compress_array_2(np.full((10, 10, 10), 5), sample_ratio=0.2))
  # pprint(compress_array_2(donut, difference=True))
  # rprint(base64.b64encode(compress_2(donut, difference=True)))
    
  # pprint(pack_metadata(compress_array_2(donut, difference=True)))
  # rprint(base64.b64encode(compress_bytes(pack_metadata(compress_array_2(donut, difference=True)))))
  # rprint(base64.b64encode(compress(donut, difference=False)))
  # print(compress_array(d3))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[308], line 97
         94   return np.concatenate([[dims, meta_.size], meta_, *(di.flatten() for di in d)], axis=0)
         96 with np.printoptions(threshold=5000):
    ---> 97   print(compress_array_2(np.full((10, 10, 10), 5), sample_ratio=0.2))
         98   # pprint(compress_array_2(donut, difference=True))
         99   # rprint(base64.b64encode(compress_2(donut, difference=True)))
        100     
       (...)
        103   # rprint(base64.b64encode(compress(donut, difference=False)))
        104   # print(compress_array(d3))


    Cell In[308], line 74, in compress_array_2(data, difference, **kwargs)
         72 ws = []
         73 for i in range(15):
    ---> 74   (data, d), w, p, n = compression_round_2(data, d, dims, **kwargs) # TODO: reindex
         75   # assert list(d.values())[-1].shape == w
         76   ws.append((w, p, (n,))) # TODO...


    Cell In[308], line 40, in compression_round_2(data, d, dims, sample_ratio, window_size, window, trial_windows)
         35     ((data_, d_), p, n), w = min(zip(map(compress_window, ws), ws),
         36                #  key=lambda x: len(x[0][1]))
         37                key=lambda x: len(compress_bytes(x[0])))
         38 else:
         39     # default_w = (2,) * dims
    ---> 40     ((data_, d_), p, n), w = compress_window(window), window
         41 return (data_, d_), w, p, n


    Cell In[308], line 20, in compression_round_2.<locals>.compress_window(w)
         18 def compress_window(w: Tuple[int, ...] = (1, 2)) -> Tuple[CompressedData, Shape, int]:
         19   w_ = np.array(w)
    ---> 20   padding = w_ - (np.array(data.shape) % w_)
         21   dp = np.pad(data, [(0, i) for i in padding], mode='reflect')
         22   new_shape = np.array(dp.shape) // w_


    ValueError: operands could not be broadcast together with shapes (3,) (4,) 



```python
# list(map(sum, [((1, 2), (3, 4))]))
list(range(5, -1, -1))
```




    [5, 4, 3, 2, 1, 0]




```python
def decompress_bytes(data: bytes) -> Compressed:
  return pickle.loads(zlib.decompress(data))

def decompress(data: bytes, difference=True) -> np.ndarray:
  d, meta = decompress_bytes(data)
  a = d[-1]
  # for w, p, n in reversed(meta + [((1, 1), (0, 0), (None,))]):
  for w, p, n in reversed(meta):
    b = np.empty(np.prod(a.shape), dtype=object)
    b[:] = [d[x] for x in a.flatten()] # awful hack
    a = np.block(b.reshape(a.shape).tolist())
    idx = tuple(slice(0, (-i if i > 0 else None)) for i in p)
    print(p, idx)
    a = a[idx]
  a = np.vectorize(lambda x: d[x])(a)
  dims = len(a.shape)
  print(dims)
  if difference:
    a = dediff(a)
  return a

# rprint(decompress_bytes(compress(donut)))
# plt.imshow(decompress(compress(donut)))
z = decompress(compress(donut, difference=True), difference=True)
plt.imshow(z)
# print(z)
```

    (2, 1) (slice(0, -2, None), slice(0, -1, None))
    (1, 1) (slice(0, -1, None), slice(0, -1, None))
    2





    <matplotlib.image.AxesImage at 0x7b93d0b4be80>




    
![png](video-compression_files/video-compression_15_2.png)
    



```python
len(compress_2(get_data()[0][:]))
```




    592202




```python
def diff(data: np.ndarray, axes: list[int] = None) -> np.ndarray:
  if axes is None:
    axes = list(range(dims := len(data.shape)))
  for i in axes:
    data = np.concatenate([np.expand_dims(np.take(data, indices=0, axis=i), i), np.diff(data, n=1, axis=i)], axis=i)
  return data

def dediff(data: np.ndarray) -> np.ndarray:
  for i in range(len(data.shape)-1, -1, -1):
    data = data.cumsum(axis=i)
  return data
```


```python
print(compression_rate(donut, difference=True))
print(zlib_compression_rate(donut))
```

    9.25044404973357
    12.579710144927537



```python
def compression_rate(f: Callable[np.ndarray, bytes], data: np.ndarray, **kwargs) -> float:
  return len(data.tobytes()) / len(f(data, **kwargs))

def zlib_compression_rate(data: np.ndarray) -> float:
  return len(s := data.tobytes()) / len(zlib.compress(s, level=9))
```


```python
a = np.random.randint(0, 2, (30,) * 3)
print(compression_rate(a))
print(zlib_compression_rate(a)) # performs worse than zlib if geometrically unstructured
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[198], line 2
          1 a = np.random.randint(0, 2, (30,) * 3)
    ----> 2 print(compression_rate(a))
          3 print(zlib_compression_rate(a)) # performs worse than zlib if geometrically unstructured


    TypeError: compression_rate() missing 1 required positional argument: 'data'



```python
# TODO: test out-of-the-box neural compressors
# n-dimensional Hilbert curve?
# test mixed window sizes for curve segmentation
# can also probably optimize by creating a function to map indices from 1d to 2d, though this may be more complex to implement
```


```python
def segment_ragged(a: np.ndarray, r: int = 2) -> list[np.ndarray]:
  out = [a]
  dims = len(a.shape)
  if isinstance(r, int):
    r = [r] * dims
  for i in range(dims):
    out = list(itertools.chain.from_iterable(np.array_split(x, r[i], axis=i) for x in out))
  return out

# a2 = np.arange(343).reshape((7, 7, 7))
a2 = np.arange(121).reshape((11, 11))
print(segment_ragged(a2))
```

    [array([[ 0,  1,  2,  3,  4,  5],
           [11, 12, 13, 14, 15, 16],
           [22, 23, 24, 25, 26, 27],
           [33, 34, 35, 36, 37, 38],
           [44, 45, 46, 47, 48, 49],
           [55, 56, 57, 58, 59, 60]]), array([[ 6,  7,  8,  9, 10],
           [17, 18, 19, 20, 21],
           [28, 29, 30, 31, 32],
           [39, 40, 41, 42, 43],
           [50, 51, 52, 53, 54],
           [61, 62, 63, 64, 65]]), array([[ 66,  67,  68,  69,  70,  71],
           [ 77,  78,  79,  80,  81,  82],
           [ 88,  89,  90,  91,  92,  93],
           [ 99, 100, 101, 102, 103, 104],
           [110, 111, 112, 113, 114, 115]]), array([[ 72,  73,  74,  75,  76],
           [ 83,  84,  85,  86,  87],
           [ 94,  95,  96,  97,  98],
           [105, 106, 107, 108, 109],
           [116, 117, 118, 119, 120]])]



```python
def unroll(a: np.ndarray, r: int = 2) -> np.ndarray:
  if a.size <= 1: return a.flatten()
  return np.concatenate([unroll(x, r) for x in segment_ragged(a, r)], axis=0)
print(unroll(a2))

# def reconstitute(a: np.ndarray, r: int = 2) -> np.ndarray:
```

    [  0   1  11  12   2  13  22  23  24   3   4  14  15   5  16  25  26  27
      33  34  44  45  35  46  55  56  57  36  37  47  48  38  49  58  59  60
       6   7  17  18   8  19  28  29  30   9  20  10  21  31  32  39  40  50
      51  41  52  61  62  63  42  53  43  54  64  65  66  67  77  78  68  79
      88  89  90  69  70  80  81  71  82  91  92  93  99 100 101 110 111 112
     102 103 104 113 114 115  72  73  83  84  74  85  94  95  96  75  86  76
      87  97  98 105 106 107 116 117 118 108 109 119 120]



```python
def unroll_2(a):
  return segment(a, (1, 1200, 1, 1)).flatten()
print(unroll_2(get_data()).shape)
```

    (1536000,)



```python
def compress_3(data: np.ndarray) -> bytes:
  # return zlib.compress(unroll(diff(data)).dumps(), level=9)
  # return zlib.compress(diff(unroll(data)).dumps(), level=9)
  # return zlib.compress(diff(unroll(diff(data))).dumps(), level=9)
  d = data
  # d = diff(d, axes=[1]) # [2, 3]
  # d = np.swapaxes(d, 0, 1)
  d = unroll_2(d)
  # d = diff(d, axes=[0])
  # d = np.transpose(d, axes=[1, 0, 2, 3])
  # d = d.reshape((1, -1, 1, 1))
  return zlib.compress(d.tobytes(), level=9)

# d = make_donut(100)
# d = np.random.randint(0, 5, (20,) * 3)
d = get_data(20)#[:, :300, :, :]
# print(compression_rate(compress_2, d))
print(compression_rate(compress_3, d))
print(zlib_compression_rate(d))
```

    2.090169162860715
    1.7474859979271358



```python
def get_data(n: int = 10) -> np.ndarray:
    return np.stack([np.load('./data/' + p) for p in ds['0'][:n]['path']])
```


```python
get_data().shape
```




    (40, 1200, 8, 16)




```python
n = 7
plt.imshow(np.diff(get_data()[0][42:42+n+1], axis=1).reshape((8 * n, 16)))
# plt.imshow(get_data()[0][49])
# transform to binary array first ?
```




    <matplotlib.image.AxesImage at 0x7f4363953150>




    
![png](video-compression_files/video-compression_28_1.png)
    



```python
get_data().max()
```




    1023




```python
t = 200
plt.figure(figsize=(8, 15))
plt.imshow(get_data()[3, 300:300+t].reshape((t, 8*16)))
# plt.imshow(diff(get_data()[3, 300:300+t], axes=[1, 2]).reshape((t, 8*16)))
# plt.imshow(np.diff(get_data()[4, 300:300+t].reshape((t, 8*16)), axis=0))
# plt.imshow(np.sort(get_data()[3, 300:300+t].reshape((t, 8*16)), axis=0))
plt.imshow(np.sort(get_data()[3, 300:300+t], axis=2).reshape((t, 8*16)))
# multi-cell differencing?
# consider column transformations
# use indirect sort
# TODO: neural methods
    # convnets?
```




    <matplotlib.image.AxesImage at 0x7f437af9af50>




    
![png](video-compression_files/video-compression_30_1.png)
    



```python
z = get_data()[4, :100]
plt.imshow(h := z.mean(axis=0))
```




    <matplotlib.image.AxesImage at 0x7f43b0ed9ad0>




    
![png](video-compression_files/video-compression_31_1.png)
    



```python
# ds = load_dataset('commaai/commavq', num_proc=num_proc, data_dir='~/dataset')
# ds = load_dataset('commaai/commavq', streaming=True)
ds = load_dataset('commaai/commavq', num_proc=16)
# it is nice to work on something where overfitting is encouraged, for a change.
```

    Using the latest cached version of the dataset since commaai/commavq couldn't be found on the Hugging Face Hub
    Found the latest cached dataset configuration 'default' at /home/annaa/.cache/huggingface/datasets/commaai___commavq/default/0.0.0/e9480d52a92c063a683aa72ad9aa5cb52d7dfbfc (last modified on Sat Jun  1 20:07:23 2024).



```python
plt.imshow(donut[25:, 25:])
```




    <matplotlib.image.AxesImage at 0x7b93d186ec80>




    
![png](video-compression_files/video-compression_33_1.png)
    



```python
a = np.arange(36).reshape((6, 6))
for i in range(dims := len(a.shape)):
  a = np.concatenate([np.expand_dims(np.take(a, indices=0, axis=i), i), np.diff(a, n=1, axis=i)], axis=i)
print(a)
for i in range(dims-1, -1, -1):
  a = a.cumsum(axis=i)
print(a)
```

    [[0 1 1 1 1 1]
     [6 0 0 0 0 0]
     [6 0 0 0 0 0]
     [6 0 0 0 0 0]
     [6 0 0 0 0 0]
     [6 0 0 0 0 0]]
    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]
     [18 19 20 21 22 23]
     [24 25 26 27 28 29]
     [30 31 32 33 34 35]]



```python
# np.array(np.array_split(np.array_split(a, 3, axis=0), 3, axis=2))
# np.array(np.array_split(np.array(np.array_split(a, 3, axis=0)), 3, axis=2)).reshape()
```


```python
# np.vectorize(lambda x: np.array([x]*4))(np.arange(25).reshape((5, 5)))
np.reshape(r := [np.arange(25).reshape((5, 5)) for i in range(25)], (25, 25))
print(r)
b = np.empty(25, dtype=object)
b[:] = r
# print(np.block(np.reshape(np.array(r, dtype=object), (5, 5))))
# np.block(list(map(list, r)))
print(np.block(b.reshape((5, 5)).tolist()))
```

    [array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])]
    [[ 0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3
       4]
     [ 5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8
       9]
     [10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13
      14]
     [15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18
      19]
     [20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23
      24]
     [ 0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3
       4]
     [ 5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8
       9]
     [10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13
      14]
     [15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18
      19]
     [20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23
      24]
     [ 0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3
       4]
     [ 5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8
       9]
     [10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13
      14]
     [15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18
      19]
     [20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23
      24]
     [ 0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3
       4]
     [ 5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8
       9]
     [10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13
      14]
     [15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18
      19]
     [20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23
      24]
     [ 0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3
       4]
     [ 5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8
       9]
     [10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13
      14]
     [15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18
      19]
     [20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23
      24]]



```python
np.pad(np.arange(25).reshape((5, 5)), 2, mode='reflect')

np.arange(50)[slice(1, 20, 3)]

# np.indices((5, 6)).reshape((30, 2))
np.indices((5, 6)).transpose(1, 2, 0).reshape((30, 2))
np.indices((3, 3, 3)).transpose(1, 2, 3, 0).reshape((27, 3))

# (1, 2, 3) % (2)
# type(np.array([1, 2]).shape)
# np.full((5, 5), 3).shape % 2

plt.imshow(np.gradient(donut)[1])

compress_bytes()
# automatic inverse function generation?

print(pickle.dumps([1, 2, 3]))
print(pickle.dumps(np.array([])))
print(pickle.dumps([]))
print(zlib.compress(pickle.dumps([[i] for i in range(30)])))
print(zlib.compress(pickle.dumps(list(range(30)))))
print(pickle.dumps({5: 30, 6: 40}))

# recursive rules?

np.unique(np.random.randint(0, 3, (50, 3)), axis=0, return_counts=True)

dx, dy = donut.shape
xs, ys = np.mgrid[:dx, :dy]
np.column_stack((xs.ravel(), ys.ravel(), donut.T.ravel()))

# np.unique([[1, 2, 3, 4, 2, 1, 3]] * 2, return_inverse=True)[1].reshape((2, 7))
# np.unique(donut, return_inverse=True)[1].reshape((res, res))

a = np.array([1, 4, 7, 4, 2, 3, 1])
x, y = np.unique(a, return_index=True)
print(x, y)
print(a[y])

sliding_window_view(z := np.arange(100).reshape((10, 10)), (2, 2), axis=(0, 1))[::2, ::2].reshape((25, 2, 2))
```


```python
np.load('./data/fee6c410e805c8f71b554d475b35a750_16.npy').shape
```




    (1200, 8, 16)


