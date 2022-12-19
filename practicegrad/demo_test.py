import numpy as np
import pathlib
import random

from practicegrad.models import MLP
from practicegrad.utils import load_mnist
from practicegrad.wrapper import Wrapper

import matplotlib.pyplot as plt

def train(model, learning_rate, loss_function):
    pass

def int_to_one_hot(num: int, total_nums: int):
    return [1.0 if i == num else 0.0 for i in range(total_nums)]

if __name__ == "__main__":
    # x_train, x_label_int, test_in, test_out_int = load_mnist()
    # x_label = list(map(lambda x: int_to_one_hot(x,10), x_label_int))
    # test_label = list(map(lambda x: int_to_one_hot(x,10), test_out_int))
    # # 28 * 28 inputs, 2 hidden layers with 32 nodese each, output layer of 10 for each digit
    # model = MLP([28 * 28, 16, 16, 10])
    
    # BATCH_SIZE = 32
    # LEARNING_RATE = 0.01
    # for _ in range(10):
    #     ri = np.random.permutation(x_train.shape[0])[:BATCH_SIZE]
    #     Xb, yb = x_train[ri], x_label_int[ri]
    #     x_batch_train = [[Wrapper(zz) for zz in z] for z in Xb]
    #     y_batch_train = [int_to_one_hot(z, 10) for z in yb]
    #     outs = list(map(model.forward, x_batch_train))
    #     print(len(outs), len(y_batch_train))
    #     print(len(list(zip(outs, y_batch_train))))
    #     losses = [sum([-1*y1*x1.log() for (x1,y1) in zip(o,y)]) for (o,y) in zip(outs, y_batch_train)]
    #     #losses = [(1 + -yii*scoreii).tanh() for yii,scoreii in zip(yi,scorei) for yi, scorei in zip(y_batch_train, outs)]
    #     total_loss = sum(losses) * (1/len(losses))
    #     print(total_loss)
    #     for param in model.get_parameters():
    #         param.grad = 0
    #     total_loss.backward()
    #     for param in model.get_parameters():
    #         temp = LEARNING_RATE * param.grad
    #         print(temp)
    #         param.val -= temp
    from sklearn.datasets import make_moons, make_blobs
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y*2 - 1
    model = MLP([2,16, 16, 1])
    def loss(batch_size=None):
    
        # inline DataLoader :)
        if batch_size is None:
            Xb, yb = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[ri], y[ri]
        inputs = [list(map(Wrapper, xrow)) for xrow in Xb]
        
        # forward the model to get scores
        scores = list(map(model.forward, inputs))
        
        # svm "max-margin" loss
        losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))
        # L2 regularization
        alpha = 1e-4
        reg_loss = alpha * sum((p*p for p in model.get_parameters()))
        total_loss = data_loss + reg_loss
        
        # also get accuracy
        accuracy = [(yi > 0) == (scorei.val > 0) for yi, scorei in zip(yb, scores)]
        return total_loss, sum(accuracy) / len(accuracy)

    total_loss, acc = loss()
    print(total_loss, acc)

    for k in range(60):
        
        # forward
        total_loss, acc = loss()
        
        # backward
        model.zero_grad()
        total_loss.backward()
        
        # update (sgd)
        learning_rate = 1.0 - 0.9*k/100
        for p in model.get_parameters():
            p.val -= learning_rate * p.grad
        
        if k % 1 == 0:
            print(f"step {k} loss {total_loss.val}, accuracy {acc*100}%")

    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Wrapper, xrow)) for xrow in Xmesh]
    scores = list(map(model.forward, inputs))
    Z = np.array([s.val > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
        
