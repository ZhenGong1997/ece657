import numpy as np
import pickle

class MLP:
    def __init__(self, input_size=784, hidden_size=64, output_size=4):
        self.w1 = np.random.randn(input_size, hidden_size) # 784x64
        self.b1 = np.random.randn(1,hidden_size) # 1x64
        self.w2 = np.random.randn(hidden_size, output_size) #64x4
        self.b2 = np.random.randn(1, output_size) # 1x4
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivative(self, y): 
        return y * (1.0 - y)
    
    # MSE
    def loss_fn(self,y_pred, y):
        loss = np.sum((y - y_pred)**2)
        return loss
    
    def predict(self, X):
        z = X @ self.w1 + self.b1
        # pass in the activation function at hidden layer
        h_z = self.sigmoid(z)
        # calculate the output
        Y_hat = h_z @ self.w2 + self.b2
        # pass in the activation function
        Y_pred = self.sigmoid(Y_hat)      
        Y_pred = (Y_pred == Y_pred.max(axis=1)[:,None]).astype(int)
        return Y_pred
    
    # training process
    def fit(self, X, Y , epochs, lr):
        for epoch in range(epochs):
            losses = []
            acc_history = []
            for x, y in zip(X,Y):
                # Forward process
                # calculate xw+b at hidden layer
                z = x @ self.w1 + self.b1
                # pass in the activation function at hidden layer
                h_z = self.sigmoid(z)
                # calculate the output
                y_hat = h_z @ self.w2 + self.b2
                # pass in the activation function
                y_pred = self.sigmoid(y_hat)
                
                # compute loss for one point
                loss = self.loss_fn(y_pred, y)
                losses.append(loss)
                
                single_acc = self.point_acc(y_pred, y)
                acc_history.append(single_acc)
                
                # Backward propagation process
                # delta2 size (1,4), grad_b2 size (1,4), grad_w2 size (64,4), h_z size (1,64)
                delta2 = ((y_pred - y) * self.sigmoid_derivative(y_pred)).reshape(1,-1)
                grad_b2 = lr * delta2
                grad_w2 = h_z.T @ grad_b2
                
                # w2 size (64,4), grad_b1 size (1,64), grad_w1 size (784,64), x size (1,784)
                grad_b1 = (lr * (delta2 @ self.w2.T) * self.sigmoid_derivative(h_z)).reshape(1,-1)               
                grad_w1 = x.T.reshape(-1,1) @ grad_b1
                
                # update weights
                self.w1 -= grad_w1
                self.b1 -= grad_b1
                self.w2 -= grad_w2
                self.b2 -= grad_b2
            
            # for self-validation
            epoch_loss = np.array(losses).mean()
            epoch_acc = np.array(acc_history).mean()
            print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}".format(epoch, epoch_loss, epoch_acc))
    
    def point_acc(self, y_pred, y):
        if np.argmax(y_pred) == np.argmax(y): 
            return 1
        return 0

    def save_model(self, path):
        model_path = path + 'pretrained_model.pkl'
        model = {'w1': self.w1, 'b1': self.b1,'w2': self.w2, 'b2': self.b2}
        with open(model_path, "wb") as fp:
            pickle.dump(model, fp)

    def read_model(self, model_path):
        with open(model_path, "rb") as fp:
            model = pickle.load(fp)
        self.w1 = model['w1']
        self.b1 = model['b1']
        self.w2 = model['w2']
        self.b2 = model['b2']    