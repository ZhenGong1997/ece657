# import required packages
from train_RNN import *
from pickle import load
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
	# 1. Load your saved model
	saved_model = load(open('./models/group1_RNN_model.pkl', 'rb'))
	model = saved_model['model']
	X_scaler = saved_model['x_scaler']
	y_scaler = saved_model['y_scaler']

	# 2. Load your testing data
	test_set = pd.read_csv('./data/test_data_RNN.csv')
	# get the name of 12 features
	headers_x = ["feature " + str(i+1) for i in range(12)]
	# extract (n,12) feature vectors and store in test_X
	test_X = test_set[headers_x].values
	# extract y vectors
	test_y = test_set[['y']].values
	# extract dates
	test_date = test_set[['date']].values
	# rescale input to fit the model
	test_X_scaler = X_scaler.fit_transform(test_X)
	test_y_scaler = y_scaler.fit_transform(test_y)
	# change to torch tensor (lstm layer) require this type
	test_X_scaler_torch = torch.from_numpy(test_X_scaler).float()
	test_y_scaler_torch = torch.from_numpy(test_y_scaler).float()
	# 3. Run prediction on the test data and output required plot and loss
	y_pred = predict(model, test_X_scaler_torch)
	loss_fn = nn.MSELoss()
	# note nn.MSELoss can only take tensor, so I change numpy to tensor
	loss = loss_fn(torch.from_numpy(y_pred).float(), test_y_scaler_torch)
	print("Test loss = ", loss.item())
	y_pred = y_scaler.inverse_transform(y_pred)
	plot_graph(test_y, y_pred, test_date)

