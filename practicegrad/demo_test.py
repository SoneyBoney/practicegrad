import pathlib
from practicegrad.models import MLP
from practicegrad.utils import load_mnist


def train(model, learning_rate, loss_function):
    pass


if __name__ == "__main__":
    x_train, x_label, test_in, test_out = load_mnist()



    model = MLP([28*28, 32, 32, 1])
    print('hi') 