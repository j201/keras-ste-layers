# STE Layers for Keras

Implementation of STE (stochastically trained ensemble) layers as introduced in [arXiv:1911.09669](https://arxiv.org/abs/1911.09669).

## Usage

STE layer code and full documentation is in `ste.py`. The only export is the `STE` class, which is more or less a drop-in replacement for the Keras `Dense` layer class, but with additional arguments for STE layers.

See `example.py` for a full example of using STE layers in LeNet-5.
