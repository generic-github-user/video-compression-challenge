from datasets import load_dataset
import itertools
import numpy as np

def load_data():
    splits = ['0', '1']
    data_files = {'0': 'data_0_to_2500.zip', '1': 'data_2500_to_5000.zip'}
    return load_dataset('commaai/commavq', num_proc=16, split=splits, data_files=data_files)

def get_paths():
    return [f['path'] for f in itertools.chain.from_iterable(load_data())]

def get_data(n: int = 10) -> np.ndarray:
    # return np.stack([np.load('./data/' + p) for p in ds['0'][:n]['path']])
    # return np.stack([np.load('/home/ubuntu/test-instance/data/' + x['path']) for x in itertools.islice(ds['0'], n)])
    return np.stack([np.load(p) for p in get_paths()[:n]])
# print(get_data().shape)

def build_model(t, j, m, m2, h1, h2, sample_shape, lstm_layers=1, binary=True, activation='relu'):
    import tensorflow as tf
    from tensorflow import keras

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
        keras.layers.Dense(h2, activation=activation),
        keras.layers.Dense(h2, activation=activation),
        keras.layers.Dense(2 ** m, activation=None)
    ])
    return model

# adapted from https://stackoverflow.com/a/22227898
def to_binary(d: np.array, m: int) -> np.array:
    return (((d.ravel()[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int).reshape((*d.shape, m))

