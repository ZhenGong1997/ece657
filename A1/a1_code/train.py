from MLP import MLP
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

    mlp = MLP(input_size=x_train.shape[1], hidden_size=64, output_size = y_train.shape[1])
    mlp.fit(x_train, y_train, 10, 0.01)
    prediction = mlp.predict(x_valid)
    test_acc = accuracy(prediction, y_valid)
    print("test acc = ", test_acc)
    mlp.save_model("a1")