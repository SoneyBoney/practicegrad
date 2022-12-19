import gzip
import numpy as np
import os
import pathlib

# modified from https://github.com/geohot/tinygrad/blob/master/datasets/__init__.py
def load_mnist():
    parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
    mnist_dir = pathlib.Path(__file__).parent.parent.resolve() / "mnist"
    X_train = parse(mnist_dir / "train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
    Y_train = parse(mnist_dir / "train-labels-idx1-ubyte.gz")[8:]
    X_test = parse(mnist_dir / "t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
    Y_test = parse(mnist_dir / "t10k-labels-idx1-ubyte.gz")[8:]
    return X_train.tolist(), Y_train.tolist(), X_test.tolist(), Y_test.tolist()