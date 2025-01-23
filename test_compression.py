from video_compression import *
from video_compression_core import *
from decompress import *

all_data = get_data(20)
for i in range(20):
    data = all_data[i] # sorry
    model = build_model(5, 5, 10, 11, 300, 300, (5, 25 * 11))
    model.load_weights('./training_checkpoint_old.weights.h5')
    c = compress(model, data[np.newaxis, ...])
    d = decompress(model, c, (1200, 8, 16), 10, 5, 2, True)
    # print(data)
    # print(d)
    # print(data == d)
    print(i, x := np.all(data == d))
    if not x:
        print(data, d, data == d)
