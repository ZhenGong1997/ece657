from zipfile import ZipInfo
import numpy as np
import pickle
class Layer:
    def __init__(self, input_size, output_size):
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(1, output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        z = x @ self.w + self.b
        y_pred = self.sigmoid(z)
        return y_pred
    
    
class MLP:
    def __init__(self, layers=[]):
        self.layers = layers
        
    def sigmoid_derivative(self, y): 
        return y * (1.0 - y)    
    
    def forward(self, x):
        # store intermediant output
        h_z = [x.reshape(1,-1)]
        out = x.reshape(1,-1)
    
        for layer in self.layers:
            out = layer.forward(out)
            h_z.append(out)
        # note h_z[0] = x, h_z[-1] = y_pred
        return out, h_z
    
    def backpropagate(self, h_z, y, lr): 
        grad_w_list = []
        grad_b_list = []   
        # last layer
        n = len(h_z)-1

        # (h_z[n] - y) size (1,4); h_z[n] size (1,4); grad_b (1,4) ; delta (1,4)
        # follow backpropagation equation, last layer is a special case
        delta = ((h_z[n] - y) * self.sigmoid_derivative(h_z[n]))
        grad_b = delta
        grad_w = h_z[n-1].T @ delta
        grad_w_list.insert(0, grad_w)
        grad_b_list.insert(0, grad_b)
        
        # intermediate and first layer
        # follow backpropagation equation, these layers are normal cases
        for i in range(len(self.layers)-2,-1,-1):
            delta = delta @ self.layers[i+1].w.T * self.sigmoid_derivative(h_z[i+1])
            grad_b = delta
            grad_w = h_z[i].T @ delta
            grad_w_list.insert(0, grad_w)
            grad_b_list.insert(0, grad_b)
        
        # update weights and bias
        for j in range(len(grad_w_list)):
            self.layers[j].w -= lr*grad_w_list[j]
            self.layers[j].b -= lr*grad_b_list[j]
                 
    def loss_fn(self, y_pred, y):
        loss = np.sum((y - y_pred)**2)/len(y)
        return loss
    
    def predict(self, X):
        Y_pred = []
        for x in X:
            x = x.reshape(1,-1)
            y_pred, _ = self.forward(x)
            Y_pred.append(y_pred)
       
        Y_pred = np.array(Y_pred).reshape(-1,4)
        Y_pred = (Y_pred == Y_pred.max(axis=1)[:,None]).astype(int)
        return Y_pred

    def get_accuracy(self, y_true, y_pred):
        if not (len(y_true) == len(y_pred)):
            print('Size of predicted and true labels not equal.')
            return 0.0

        corr = 0
        for i in range(0,len(y_true)):
            corr += 1 if (y_true[i] == y_pred[i]).all() else 0

        return corr/len(y_true)

    # training process
    def fit(self, X, Y , epochs, lr):
        for epoch in range(epochs):    
            losses = []  
            acc_history = []
            for x, y in zip(X, Y):
                x = x.reshape(1,-1) 
                y = y.reshape(1,-1)        
                # feed forward
                y_pred, h_z = self.forward(x)

                # calculate training loss
                loss = self.loss_fn(y_pred, y)
                losses.append(loss)
                # hot encode y_pred, calculate training accuracy
                he_y_pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
                single_acc = self.get_accuracy(y, he_y_pred)
                acc_history.append(single_acc)

                # backpropagation
                self.backpropagate(h_z, y, lr)

            epoch_loss = np.array(losses).mean()
            epoch_acc = np.array(acc_history).mean()   
            print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}".format(epoch, epoch_loss, epoch_acc*100))

    def save_model(self, path):
        model_path = path + 'pretrained_model.pkl'
        model = {'layers': self.layers}
        with open(model_path, "wb") as fp:
            pickle.dump(model, fp)

    def read_model(self, model_path):
        with open(model_path, "rb") as fp:
            model = pickle.load(fp)
        self.layers = model['layers']