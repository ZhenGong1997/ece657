from MLP import MLP
from MLP import Layer
import numpy as np
import pandas as pd
from acc_calc import accuracy 

if __name__ == '__main__':
    x_train = np.array(pd.read_csv('train_data.csv',header=None))
    y_train = np.array(pd.read_csv('train_labels.csv',header=None))

    # split training set and validation set
    len_valid = len(x_train)//4
    x_valid = x_train[:len_valid]
    x_train = x_train[len_valid:]
    y_valid = y_train[:len_valid]
    y_train = y_train[len_valid:]

    # initialize a general model, 784->64->4
    layers = []
    layer1 = Layer(x_train.shape[1], 64)
    layer2 = Layer(64, y_train.shape[1])
    layers.append(layer1)
    layers.append(layer2)
    mlp = MLP(layers)
    mlp.fit(x_train, y_train, 10, 0.01)

    # validation
    prediction = mlp.predict(x_valid)
    test_acc = accuracy(y_valid, prediction)
    print("validation acc = ", test_acc*100)
    mlp.save_model("a1")