# import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler

from pickle import dump

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
def generate_data():
    # Get data
    data = pd.read_csv('./data/q2_dataset.csv')
    # get headers
    cols = list(data.columns.values.tolist())
    # vol open high low
    features = cols[2:]
    features_data = data[features].values
    # get a copy of data, prevent from modifying original data
    dates = data[cols[0]].copy()
    # make sure data format are consistent mm/dd/yy, where we only keep the last two digits of the year
    for i in range(len(dates)):
        if(len(dates[i])>8):
            dates[i] = dates[i][:6]+dates[i][8:]
    X = []
    y = []

    for i in range(len(features_data)-3):
        # Vol, open, high, low in the past 3 days
        x_i = features_data[i+1:i+4]
        # open in the current day
        y_i = features_data[i][1]
        X.append(x_i.reshape(12))
        y.append([y_i, dates[i]])

    X = np.array(X)
    y = np.array(y)

    # shuffle data and split train and test dataset 70/30
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    split_index = int(len(X) * 0.7)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # generate csv files
    headers = ['feature '+ str(i+1) for i in range(12)] + ['y', 'date']
    train_df = pd.DataFrame(data=np.concatenate((X_train, y_train), axis=1), columns=headers, index=None)
    train_df
    train_df.to_csv('./data/train_data_RNN.csv',index = False)
    test_df = pd.DataFrame(data=np.concatenate((X_test, y_test), axis=1), columns=headers, index=None)
    test_df.to_csv('./data/test_data_RNN.csv' ,index = False)

class RNN_LSTM(nn.Module):
    
    # a very simple LSTM model, 1 LSTM layer and 1 linear layer
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_LSTM, self).__init__()
        # nn.LSTM requires a batch size, therefore we need to reshape the data to 3d
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # lstm will return a 3d tensor, we need to convert it to 2d and fit the fc layer
        output,_ = self.lstm(x)
        n, batch_size, d = output.size()
        output = output.view(n*batch_size, d)
        output = self.linear(output)
        # the output is the y_pred, it should have same shape as train_y
        output = output.view(n,batch_size,1)
        return output

def fit(epochs, lr, model, train_X, train_y, opt_func=torch.optim.Adam, criterion = nn.MSELoss()):
    # optimizer is Adam
    opt = opt_func(model.parameters(), lr)
    losses = []
    # reshape data to fit the first lstm layer
    train_X = train_X.reshape(-1,1,12)
    train_y = train_y.reshape(-1,1,1)
    
    # Training phase
    for epoch in range(epochs):  
        output = model(train_X)
        loss = criterion(output, train_y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())    
		# print process every 10 epochs
        if (epoch+1)%10 == 0:
            print('Epoch:%i, Loss: %f' % (epoch+1, loss.item()))
    return losses

def predict(model, x):
    # reshape input data to fit first layer of lstm
    x = x.reshape(-1,1,12)
    # y_pred is transformed, require to inverse transform later on
    y_pred = model(x).detach().numpy().reshape(-1,1)
    return y_pred  

def plot_graph(train_y, y_pred, train_date):
	columns = ["gt", "pred", "date"]
	df = pd.DataFrame(data=np.concatenate((train_y, y_pred, train_date), axis=1), columns=columns)
	df['date'] = pd.to_datetime(df['date'], format="%m/%d/%y")
	df = df.sort_values(by='date')
	plt.figure(figsize=(8,6))
	plt.plot(df['date'], df['gt'])
	plt.plot(df['date'], df['pred'])
	plt.xlabel('Date')
	plt.ylabel('Opening price')
	plt.legend(['groundtruth', 'prediction'])
	plt.show()

def plot_loss(losses):
    plt.figure(figsize=(8,6))
    plt.plot([i for i in range(200)],losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

if __name__ == "__main__": 
    # 1. load your training data
    # generate_data()
    train_set = pd.read_csv('./data/train_data_RNN.csv')
    headers_x = ["feature " + str(i+1) for i in range(12)]
    train_X = train_set[headers_x].values
    train_y = train_set[['y']].values
    train_date = train_set[['date']].values

    # rescale data
    scaler = MinMaxScaler()
    X_scaler = MinMaxScaler()
    train_X_scaler = X_scaler.fit_transform(train_X)
    y_scaler = MinMaxScaler()
    train_y_scaler = y_scaler.fit_transform(train_y)
    train_X_scaler = torch.from_numpy(train_X_scaler).float()
    train_y_scaler = torch.from_numpy(train_y_scaler).float()

    # 2. Train your network
    # 		Make sure to print your training loss within training to show progress
    # 		Make sure you print the final training loss
    model = RNN_LSTM(input_size=12, hidden_size=24, output_size=1)
    training_loss = fit(200, 0.1, model, train_X_scaler, train_y_scaler)
    y_pred = predict(model, train_X_scaler)
    y_pred = y_scaler.inverse_transform(y_pred)
    # plot graph
    plot_graph(train_y, y_pred, train_date)
    plot_loss(training_loss)
    # Save your model
    save_model = {'model': model, 'x_scaler': X_scaler, 'y_scaler': y_scaler}
    dump(save_model, open('./models/group1_RNN_model.pkl', 'wb'))