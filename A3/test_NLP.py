# import required packages

import pickle
from process_data import *
import torch
import numpy as np
from train_NLP import data_loader
from train_NLP import NLP

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
def predict(model, batch_X, batch_y):
	batch_size = 1000
	batch_num = 25
	correct = 0
	model.eval()
	print("Predicting is in progress, Please be patient...")
	for i in range(batch_num):
		X = batch_X[i].reshape(batch_size,-1)
		y = batch_y[i].reshape(batch_size)
		output = model(X)   
		# softmax output, get the output with max possibility
		pred = output.max(1, keepdim=True)[1]       
		correct += pred.eq(y.view_as(pred)).sum().item()
	accuracy = correct / (batch_num*batch_size)	
	print("Test Accuracy = ", accuracy) 

if __name__ == "__main__": 

	# 1. Load your saved model
	saved_model = pickle.load(open('./models/group1_NLP_model.pkl','rb'))
	model = saved_model['model']
	tfidf_vectorizer = saved_model['tf_vector']
	# 2. Load your testing data

	test_neg_data, test_neg_labels = get_dataset('./data/aclImdb/test/neg/', "negative")
	test_pos_data, test_pos_labels = get_dataset('./data/aclImdb/test/pos/', "positive")
	test_data, test_labels = combine_dataset(test_neg_data, test_pos_data, test_neg_labels, test_pos_labels)
	# remove punctuation and stopwords
	test_data_filtered = data_filter(test_data)
	test_features = tfidf_vectorizer.transform(test_data_filtered)
	test_X = np.array(test_features.toarray())
	test_labels = np.array(test_labels)

	# 3. Run prediction on the test data and print the test accuracy
	batch_X, batch_y = data_loader(test_X, test_labels, batch_size=1000, batch_num=25)
	predict(model,batch_X, batch_y)
