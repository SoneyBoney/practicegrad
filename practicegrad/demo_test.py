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
    x_train, x_label_int, test_in, test_out_int = load_mnist()
    x_label = list(map(lambda x: int_to_one_hot(x,10), x_label_int))
    test_label = list(map(lambda x: int_to_one_hot(x,10), test_out_int))
    # 28 * 28 inputs, 2 hidden layers with 32 nodese each, output layer of 10 for each digit
    model = MLP([28 * 28, 16, 16, 10])
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    for _ in range(10):
        ri = np.random.permutation(x_train.shape[0])[:BATCH_SIZE]
        Xb, yb = x_train[ri], x_label_int[ri]
        x_batch_train = [[Wrapper(zz) for zz in z] for z in Xb]
        y_batch_train = [int_to_one_hot(z, 10) for z in yb]
        outs = list(map(model.forward, x_batch_train))
        print(len(outs), len(y_batch_train))
        print(len(list(zip(outs, y_batch_train))))
        losses = [sum([-1*y1*x1.log() for (x1,y1) in zip(o,y)]) for (o,y) in zip(outs, y_batch_train)]
        #losses = [(1 + -yii*scoreii).tanh() for yii,scoreii in zip(yi,scorei) for yi, scorei in zip(y_batch_train, outs)]
        total_loss = sum(losses) * (1/len(losses))
        print(total_loss)
        for param in model.get_parameters():
            param.grad = 0
        total_loss.backward()
        for param in model.get_parameters():
            temp = LEARNING_RATE * param.grad
            print(temp)
            param.val -= temp
        
