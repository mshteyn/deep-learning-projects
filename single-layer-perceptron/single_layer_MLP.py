"""
Implementation of a simple multilayer perceptron from scratch
Forward and backward propogation transformations are coded by hand
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
import pdb
import os


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Transform(object):
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        """
        pass


class ReLU(Transform):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        """ 
        x shape (indim, batch_size)
        return shape (indim, batch_size)
        """
        x[x<0] = 0 # rectify by assigning all values < 0 to 0
        self.x = x
        return x
        #raise NotImplementedError()

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        relu_gradient = self.x>0
        grad_wrt_out = grad_wrt_out * relu_gradient.T

        return grad_wrt_out.T
        #raise NotImplementedError()


class LinearMap(Transform):
    def __init__(self, indim, outdim, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        lr: learning rate
        """
        super(LinearMap, self).__init__()
        self.weights = 0.01 *torch.rand((outdim, indim), dtype=torch.float64, requires_grad=True, device=device)
        self.bias = 0.01 * torch.rand((outdim, 1), dtype=torch.float64, requires_grad=True, device=device)
        self.lr = lr


    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        # Linear transformation
        self.x = x
        q  = x.T @ self.weights.T + self.bias.T # matrix multiply    
        q = q.T # to shape into (outdim, batch_size)  
        self.q = q
        return  q
        #raise NotImplementedError()


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        """
        #compute grad_wrt_weights

        grad_wrt_input = grad_wrt_out.T @ self.weights # dot product
        grad_wrt_weights = grad_wrt_out @ self.x.T
        grad_wrt_bias = torch.sum(grad_wrt_out, dim=1, keepdim=True) # 
        # Final shape should be 10 x 100? (not 10 x64...)

        self.grad_wrt_weights = grad_wrt_weights
        self.grad_wrt_bias = grad_wrt_bias


        return grad_wrt_input


    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        """
        self.weights += self.grad_wrt_weights*self.lr
        self.bias += self.grad_wrt_bias*self.lr
        # self.weights -= self.grad_wrt_weights*self.lr
        # self.bias -= self.grad_wrt_bias*self.lr        
        #raise NotImplementedError()


class SoftmaxCrossEntropyLoss(object):
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size)
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """

        # # compute softmax
        probs = torch.exp(logits.T)/(torch.sum(torch.exp(logits.T), dim=0))

        #Compute cross entropy loss 
        labels = torch.from_numpy(labels)
        #L = sum(labels[0]*np.log(p[0])) * -1 #But this is only for one column
        L = - (torch.sum(labels[0]*torch.log(probs[0])) + torch.sum(labels[1]*torch.log(probs[1])))#For both columns..does it matter? This is linear scaling of loss
        L = L / labels.shape[1] # Divide by length of all labels -- both columns
        self.probs = probs
        self.labels = labels
        self.L = L.item()
        #self.error = ((probs[0]>probs[1]) == (labels[0]>labels[1])).astype('int') # compute error
        return L.item()
       #raise NotImplementedError()


    def backward(self):
        """
        return grad_wrt_logits shape (num_classes, batch_size)
        """

        back_logits = (self.labels - self.probs)/self.probs.shape[1]

        grad_wrt_logits = back_logits  # propogate error backwards -- the derivative accounts for the error

        self.grad_wrt_logits = grad_wrt_logits
        return grad_wrt_logits


        #raise NotImplementedError()
    
    def getAccu(self):
        """
        return accuracy here
        """
        pct_acc = sum(torch.argmax(self.probs,0) == torch.argmax(self.labels,0))/self.labels.shape[1]
        pct_acc = pct_acc.item()
        return pct_acc


class SingleLayerMLP(Transform):
    """constructing a single layer neural network with  previous functions"""
    def __init__(self, indim, outdim, hidden_layer=100, lr=0.01):
        super(SingleLayerMLP, self).__init__()
        self.LT1    = LinearMap(indim, hidden_dim, lr)  #initialize LT1
        self.ReLU   = ReLU()
        self.LT2    = LinearMap(hidden_dim, outdim, lr) #initialize LT2
        self.SM_CEL = SoftmaxCrossEntropyLoss()

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return the presoftmax logits shape(outdim, batch_size)
        """
        LinearMap.forward(self.LT1, train_features.T)
        ReLU.forward(self.ReLU, self.LT1.q) # 0 is a dummy variable
        LinearMap.forward(self.LT2, self.ReLU.x)
        Y_labels_train = labels2onehot(train_labels).T      
        SoftmaxCrossEntropyLoss.forward(self.SM_CEL, self.LT2.q.T, Y_labels_train)
        grad_wrt_out = self.LT2.q

        return grad_wrt_out
        #raise NotImplementedError()


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        calculate the gradients wrt the parameters
        """
        o_back = SoftmaxCrossEntropyLoss.backward(self.SM_CEL)
        h_back = LinearMap.backward(self.LT2, o_back)  
        q_back = ReLU.backward(self.ReLU, h_back)
        out = LinearMap.backward(self.LT1, q_back)
        #raise NotImplementedError()

    
    def step(self):
        """update model parameters"""
        LinearMap.step(self.LT1)   
        LinearMap.step(self.LT2)            

        #raise NotImplementedError()


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

if __name__ == "__main__":
    """ plot the loss and accuracies during the training process and test process. 
    """

    indim = 10
    outdim = 2
    hidden_dim = 100
    lr = 0.01
    batch_size = 64
    epochs = 200

    #dataset

    datadir = os.get_pwd()
    Xtrain = np.loadtxt(datadir+'/XTrain.txt', delimiter="\t")
    Ytrain = np.loadtxt(datadir+'/YTrain.txt', delimiter="\t").astype(int)

    m1, n1 = Xtrain.shape
    print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    pdb.set_trace()
    
    Xtest = np.loadtxt(datadir+"/XTest.txt", delimiter="\t")
    Ytest = np.loadtxt(datadir+"/yTest.txt", delimiter="\t").astype(int)
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)


    train_model = SingleLayerMLP(indim, outdim, hidden_dim, lr)
    test_model = SingleLayerMLP(indim, outdim, hidden_dim, lr)

    Train_Loss, Train_Accu  = [], []
    Test_Loss, Test_Accu = [], []


    for ep in np.arange(epochs):
        Train_Loss_epoch, Train_Accu_epoch  = [], [] # per epoch loss 
        for batch in train_loader: #iterate thru dataloader
            train_features, train_labels = next(iter(train_loader))
            grad_wrt_out = SingleLayerMLP.forward(train_model, train_features.T)
            SingleLayerMLP.backward(train_model,  grad_wrt_out)
            SingleLayerMLP.step(train_model)
            Train_Accu_epoch.append(train_model.SM_CEL.getAccu())
            Train_Loss_epoch.append(train_model.SM_CEL.L)
        print('Epoch # ' + str(ep+1) + ': loss = '+ str(np.round(train_model.SM_CEL.L,3)))
        #now test the model
        test_features, test_labels = torch.from_numpy(Xtest), torch.from_numpy(Ytest)
        test_model.LT1.weights = train_model.LT1.weights #set test weights to train weights
        test_model.LT2.weights = train_model.LT2.weights
        SingleLayerMLP.forward(test_model, test_features.T)
        Test_Accu.append(test_model.SM_CEL.getAccu())
        Test_Loss.append(test_model.SM_CEL.L)
        Train_Accu.append(np.mean(Train_Accu_epoch))
        Train_Loss.append(np.mean(Train_Loss_epoch))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(np.array(Train_Loss))
    ax.plot(np.array(Test_Loss))    
    ax.set_ylabel('Loss')
    ax.set_xlabel('num. training epochs')
    ax.legend(['Train', 'Test'])
   
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(np.array(Train_Accu))
    ax.plot(np.array(Test_Accu))    
    ax.set_ylabel('% Accuracy')
    ax.set_xlabel('num. training epochs')
    ax.legend(['Train', 'Test'])
    plt.show(block=False)    
    pdb.set_trace()
