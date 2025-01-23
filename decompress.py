from arithmeticcoding import ArithmeticDecoder, SimpleFrequencyTable, BitInputStream
from pathlib import Path
from typing import Tuple
from video_compression_core import to_binary, get_data, get_paths
import io
import multiprocessing as mp
import numpy as np
import os
import tqdm
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
ctx = mp.get_context("spawn")
n_processes = 30
HERE = Path(__file__).resolve().parent
(target_dir := HERE / 'compression_challenge_submission_decompressed').mkdir(exist_ok=True)

def get_context(xs: np.ndarray, t: int, i: int, j: int, t_: int, k_: int,
                m: int=10, extra_bit=True) -> np.ndarray:
    h = k_ * 2 + 1
    r = xs[t:t+t_, i:i+h, j:j+h]
    return to_binary(r, m + int(extra_bit))

def decompress(M: 'tf.keras.Model', data: bytes, s: Tuple[int, int, int], m: int = 10,
               t_: int = 5, k_: int = 2, extra_bit=True, verbose=False) -> np.ndarray:

    decoder = ArithmeticDecoder(32, BitInputStream(io.BytesIO(data)))
    frame_shape = s[1:]
    output = np.full((s[0] + t_, s[1] + 2 * k_, s[2] + 2 * k_),
                     (2**m if extra_bit else 0))
    for t in range(s[0]):
        if verbose:
            print(f'Decoding frame {t+1}')
        raw_freqs = tf.nn.softmax(M(np.stack([
                    get_context(output, t, x, y, t_, k_, m=m, extra_bit=extra_bit)
                    for x, y in np.ndindex(*frame_shape)], axis=0)),
                                  axis=1).numpy().reshape(frame_shape + (1024,))
        scaled_freqs = (raw_freqs * (2 ** 16)).round().astype(int) + 1
        for x, y in np.ndindex(*frame_shape):
            freqs = SimpleFrequencyTable(scaled_freqs[x, y])
            symbol = decoder.read(freqs)
            output[t+t_, x+k_, y+k_] = symbol
    return output[t_:, k_:-k_, k_:-k_]

def decompress_wrapped(p: Path) -> np.ndarray:
    import tensorflow as tf
    model = tf.keras.models.load_model(str(HERE / 'checkpoint.keras'))
    source = HERE / p.name
    target = target_dir / p.name
    if source.exists() and not target.exists():
        print(f"Process {os.getpid()} starting work on {p.name}")
        r = decompress(model, source.read_bytes(), (1200, 8, 16), 10, 5, 2, True)
        np.save(target, r)
        print(f"Process {os.getpid()} finished work on {p.name}")
        print(f"Wrote decompressed data to {target}")
    else:
        print(f'Warning: missing file {p.name}')
    # return r

# def test_decompression():
    # data = get_data(1)[0]
    # b = compress_wrapped(data[:])
    # decompressed = decompress_wrapped(b)
    # print(data.shape, decompressed.shape)
    # print(c := data == decompressed)
    # print(np.all(c))
    # breakpoint()

def decompress_all():
    paths = list(map(Path, get_paths()))
    print(len(paths))
    with ctx.Pool(n_processes) as pool:
        tqdm.tqdm(pool.map(decompress_wrapped, paths), total=len(paths))

if __name__ == '__main__':
    decompress_all()
