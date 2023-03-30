"""
You will need to validate your NN implementation using PyTorch. You can use any PyTorch functional or modules in this file.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
import pdb


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SingleLayerMLP(nn.Module):
    """constructing a single layer neural network with Pytorch"""
    def __init__(self, indim, outdim, hidden_layer=100):
        super(SingleLayerMLP, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(indim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, outdim),
            torch.nn.Softmax() )

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)    


    def forward(self, x):
        """
        x shape (batch_size, input_dim)
        """      
        y_pred = self.model(x)
        self.y_pred = y_pred
        return y_pred


class DS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.length = len(X)
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return (x, y)

    def __len__(self):
        return self.length

def labels2onehot(labels: np.ndarray):
    return np.array([[i==lab for i in range(2)] for lab in labels]).astype(int)

def validate(loader):
    """takes in a dataloader, then returns the model loss and accuracy on this loader"""
    test_features, test_labels = next(iter(test_loader))
    y_pred = model(test_features.float())
    loss = loss_fn(y_pred, test_labels)
    loss = loss.item()
    y_labels = torch.from_numpy(labels2onehot(test_labels))
    acc_batch = sum(torch.argmax(y_pred,1) ==y_labels[:,1])/y_labels.shape[0]
    out={}
    out['loss'] = loss
    out['accu'] = acc_batch.item()
    return out
    #raise NotImplementedError()



if __name__ == "__main__":
    """The dataset loaders were provided for you.
    You need to implement your own training process.
    You need plot the loss and accuracies during the training process and test process. 
    """


    indim = 10
    outdim = 2
    hidden_dim = 100
    lr = 0.01
    batch_size = 64
    epochs = 200

    #dataset
    datadir = '/Users/misha/Documents/Classes/2023 Spring/Deep_Learning/HW1/data/'
    Xtrain = np.loadtxt(datadir+'XTrain.txt', delimiter="\t")
    Ytrain = np.loadtxt(datadir+'YTrain.txt', delimiter="\t").astype(int)
    Xtest = np.loadtxt(datadir+"XTest.txt", delimiter="\t")
    Ytest = np.loadtxt(datadir+"yTest.txt", delimiter="\t").astype(int)
    # Xtest = np.loadtxt("/Users/liu5/Documents/ta/2023 spring/HW1/data/XTest.txt", delimiter="\t")
    # Ytest = np.loadtxt("/Users/liu5/Documents/ta/2023 spring/HW1/data/yTest.txt", delimiter="\t").astype(int)    
    # Xtrain = np.loadtxt("/Users/liu5/Documents/ta/2023 spring/HW1/data/XTrain.txt", delimiter="\t")
    # Ytrain = np.loadtxt("/Users/liu5/Documents/ta/2023 spring/HW1/data/yTrain.txt", delimiter="\t").astype(int)

    m1, n1 = Xtrain.shape
    print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

 
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    #construct the model
    model = SingleLayerMLP(indim, outdim, hidden_dim)
    loss_fn = torch.nn.CrossEntropyLoss() #define loss

    #construct the training process
    Loss_per_testE, Prct_acc_testE = [],[] # remove later
    Loss_per_epoch, Prct_acc_epoch =[], []
    for epc in np.arange(epochs):
        loss_batch, acc_batch = [], []
        for batch in train_loader: #iterate thru dataloader
            train_features, train_labels = next(iter(train_loader))
            y_pred = model(train_features.float())
            #calculate loss
            loss = loss_fn(y_pred, train_labels)
            loss_batch.append(loss.item())
            # zero the gradient before backprop
            model.optimizer.zero_grad()
            #backpropogate the loss
            loss.backward()
            model.optimizer.step()
            y_labels = torch.from_numpy(labels2onehot(train_labels))
            acc_batch.append(sum(torch.argmax(y_pred,1) ==y_labels[:,1])/y_labels.shape[0])
        print('Epoch # ' + str(epc+1) + ': loss = '+ str(np.round(loss.item(),3)))        
        out = validate(test_loader)
        Loss_per_testE.append(out['loss'])
        Prct_acc_testE.append(out['accu'])        
        # train data out        
        Prct_acc_epoch.append(np.mean(acc_batch))
        Loss_per_epoch.append(np.mean(loss_batch))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(np.array(Loss_per_epoch))
    ax.plot(np.array(Loss_per_testE))
    #ax.plot(np.array(Test_Loss))    
    ax.set_ylabel('Loss')
    ax.set_xlabel('num. training epochs')
    ax.legend(['Train', 'Test'])
    # Accuracy plot
    fig, ax = plt.subplots()
    ax.plot(np.array(Prct_acc_epoch)*100)
    ax.plot(np.array(Prct_acc_testE)*100)
    ax.set_ylabel('% Accurate')
    ax.set_xlabel('num. training epochs')
    ax.legend(['Train', 'Test'])    
    plt.show(block=False)   
    pdb.set_trace() 
