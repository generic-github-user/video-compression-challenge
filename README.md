# compression-challenge

Experiments in lossless data compression for the comma.ai compression challenge (see https://github.com/commaai/commavq). The most promising method is implemented in video_compression.py, decompress.py, and video_compression_core.py and is based on neural [arithmetic coding](https://en.wikipedia.org/wiki/Arithmetic_coding), a form of entropy encoding in which we learn a (discrete) conditional distribution over the possible values at each point in the sequence and size the intervals in the encoding scheme accordingly, such that as the accuracy of the predictor approaches 100%, the number of bits required to encode any particular sample approaches 0. There are many possible predictors (e.g., one of the simplest is merely observing the empirical distribution of values without conditioning on sequence context); neural networks allow us to approximate an optimal predictor by modeling the process that generated the data, and thus yield much stronger results. [Practical Full Resolution Learned Lossless Image Compression](https://arxiv.org/abs/1811.12817) demonstrates an example application of this technique to lossless image compression.

In this case, we are operating on condensed representations of videos with 1200 frames each, where the 8 * 16 "pixels" are each 10-bit numbers representing a learned embedding of a patch of video from the source. The context for the conditional distribution here is therefore every pixel in an `n`-by-`n` window surrounding the target, over the past `t` timesteps (some padding is of course needed for the first few frames); we want to learn a mapping from that context to a distribution over the 1024 possible patches, and use this to size the intervals in the arithmetic coding. Currently, an LSTM is used to learn this mapping, but a (3D) convolutional neural network might also be appropriate. We transform this into a sequential problem (like that of generic text/sequence compression -- note that the leading entries on http://prize.hutter1.net/ almost all use learned compression schemes similar to the one used here) by predicting (i.e., rows -> columns -> frames), such that the reconstruction process never uses patches that have not been previously reconstructed.

The compression process and decompression processes are therefore direct inverses of each other: in the former, we iterate over each pixel and narrow the encoding interval according to the learned probability function and the actual value of that patch; in the latter, we iterate through the video in the same order and deduce the interval at that step (and therefore the true value of the patch) from the inferred distribution at that point (from this, you might be able to see how the neural network acts as a lossy, highly condensed compression dictionary).

The trained model must of course be included with the compressed data for it to be able to be reconstructed, but for large amounts of data (as in this challenge), this is generally a negligible cost. In the future, I'd like to explore automatically optimizing model size vs. compression performance as a secondary hyperparameter. In my experiments, this scheme achieved a compression ratio of between 3.5 and 4.0, which meets or exceeds the state of the art on this benchmark (to the best of my memory), and far exceeds the 1.6 achieved by the generic lzma benchmark. The main drawback of this method (which is not specific to my implementation) is that it is incredibly slow, even with GPU acceleration, since inference must be run separately for every pixel in the video (fortunately, we can batch inference across entire frames, but this offers a limited speedup for this case since the number of frames per video here vastly exceeds the number of pixels per frame).
