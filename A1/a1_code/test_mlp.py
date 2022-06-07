import numpy as np
import pandas as pd
from MLP import MLP

STUDENT_NAME = 'Zhen Gong, Eugene Wang'
STUDENT_ID = '20673670, 20657282'

def test_mlp(data_file):
	# Load the test set
	# START
	x_test = np.array(pd.read_csv(data_file,header=None))
    # END


	# Load your network
	# START
	mlp = MLP()
	mlp.read_model('./a1pretrained_model.pkl')
	# END


	# Predict test set - one-hot encoded
	# y_pred = ...
	y_pred = mlp.predict(x_test)
	return y_pred
	# return y_pred


'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''