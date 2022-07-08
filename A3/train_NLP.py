# import required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from pickle import dump
import random
from process_data import *

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
class NLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(in_size, hidden_size)
        # hidden layer 2
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # output layer
        self.linear3 = nn.Linear(hidden_size, out_size)
        self.input_size = in_size
    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out   

def train(model, batch_X, batch_y, optim, criterion=nn.CrossEntropyLoss(), epochs=7):
    optim = optim
    train_losses = []
    train_accs = []
    batch_size = 1000
    batch_num = 25
    print("Training begin, it will run 7 epochs, each epoch has 25 batchs, batch_size = 1000")
    print("Training Start, Please be patient...")
    for epoch in range(1, epochs+1):
        loss_sum = 0.0
        correct = 0
        model.train()
        for i in range(batch_num):
            X = batch_X[i].reshape(batch_size,-1)
            y = batch_y[i].reshape(batch_size)
            output = model(X)   
            pred = output.max(1, keepdim=True)[1]       
            correct += pred.eq(y.view_as(pred)).sum().item()
            loss = criterion(output, y)
            loss.backward()
            optim.step()
            loss_sum += loss.item()
        
        # eval the acc and loss after an epoch
        model.eval()
        epoch_loss = loss_sum
        train_losses.append(epoch_loss)
        accuracy = correct / (batch_num*batch_size)
        train_accs.append(accuracy)
        print('Train Epoch: %i, Loss: %f, Accuracy: %f' % (epoch, epoch_loss, accuracy))   
    return train_losses, train_accs

def data_loader(train_X, train_labels, batch_size, batch_num):
    batch_X = []
    batch_y = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(batch_num):
        start = i*batch_size
        end = (i+1)*batch_size
        batch_X.append(torch.from_numpy(train_X[start:end]).float().to(device))
        batch_y.append(torch.from_numpy(train_labels[start:end]).to(device))
    return batch_X, batch_y 

def plot(train_losses, train_accs):
    plt.figure()
    plt.title('Train loss vs epochs')
    plt.plot([i + 1 for i in range(7)], train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.figure()
    plt.title('Train Accuracy')
    plt.plot([i + 1 for i in range(7)], train_accs)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__": 
	# 1. load your training data
    # download pos& neg reviews for training
    train_neg_data, train_neg_labels = get_dataset('./data/aclImdb/train/neg/', "negative")
    train_pos_data, train_pos_labels = get_dataset('./data/aclImdb/train/pos/', "positive")
    # combine them
    train_data, train_labels = combine_dataset(train_neg_data, train_pos_data, train_neg_labels, train_pos_labels)
    # remove punctuation and stopwords
    train_data_filtered = data_filter(train_data)
    # shuffle dataset
    temp = list(zip(train_data_filtered, train_labels))
    random.shuffle(temp)
    train_data_filtered, train_labels = zip(*temp)
    vectorizer, train_features = vectorization(train_data_filtered)
    train_X = np.array(train_features.toarray())
    train_labels = np.array(train_labels)
    # split data into batchs for training
    batch_X, batch_y = data_loader(train_X, train_labels, batch_size=1000, batch_num=25)
    

	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=NLP(in_size = batch_X[0].shape[1], hidden_size=512, out_size=2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_losses, train_accs = train(model, batch_X, batch_y, optimizer)
    plot(train_losses, train_accs)

	# 3. Save your model
    save_model = {'tf_vector': vectorizer, 'model': model}
    dump(save_model, open('./models/group1_NLP_model.pkl', 'wb'))
