
import os
import numpy as np

print("network definition")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
import torch.autograd.variable

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import collections 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bnm1 = nn.BatchNorm1d(2, momentum=0.1)
        self.fc0 = nn.Conv1d(2, 32, kernel_size=11, padding=5)
        self.fc1 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.fc2 = nn.Conv1d(64, 128, kernel_size=11, padding=5)
        self.bnm2 = nn.BatchNorm1d(128, momentum=0.1)
        self.fc3 = nn.Conv1d(128, 256, kernel_size=11, padding=5)
        self.fc4 = nn.Conv1d(256, 3, kernel_size=11, padding=5)
        self.fc5 = nn.Conv1d(35, 64, kernel_size=21, padding=10)
        self.fc6 = nn.Conv1d(64, 3, kernel_size=21, padding=10)

    def forward(self, x):
        x = self.bnm1(x)
        
        x1 = F.leaky_relu(self.fc0(x))
        x = F.max_pool1d(x1, kernel_size=4, stride=4, padding=0)
        x = F.leaky_relu(self.fc1(x))
        x = F.max_pool1d(x, kernel_size=4, stride=4, padding=0)
        x = F.leaky_relu(self.fc2(x))
        x = F.max_pool1d(x, kernel_size=4, stride=4, padding=0)
        
        x = self.bnm2(x)
        
        x = F.leaky_relu(self.fc3(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2, padding=0)
        x = self.fc4(x)
        x = torch.nn.functional.interpolate(x, mode='linear', scale_factor=128)
        
        x = torch.cat([x,x1],dim=1)
        x = F.leaky_relu(self.fc5(x))
        x = self.fc6(x)
        return x

model = Net()

print("visualization function")
import matplotlib.pyplot as plt
from PIL import Image 

def visualizecurve(x,y,z):
    grid = np.ones((200,1000,3),dtype=int)*255
    x = x[0:1000]
    x = x-min(x)
    x = x*400/max(x)
    
    x = np.minimum(x,np.ones(1000)*180)
    x = x.astype(int)
    
    if z is not None:
        for t in range(1000):
            grid[x[t]][t][:] = 0
            grid[0:20,t,y[t]] = 0            
            grid[180:200,t,z[t]] = 0
    else:
        for t in range(1000):
            grid[x[t]][t][:] = 0
            grid[0:20,t,y[t]] = 0            
        
    return np.uint8(grid)

print("load data")
allecgdata = np.load("alldata.npz")
allecgdata = allecgdata["arr_0"]

X,Y = [],[]    
for i in range(allecgdata.shape[0]):
    X.append(allecgdata[i][0:2])
    Y.append(allecgdata[i][2].astype(int))

assert(all([x.shape==(2,8192) for x in X]))
assert(all([y.shape[0]==8192 for y in Y]))

X,Y = shuffle(X,Y)

Xtest,Ytest = X[0:15],Y[0:15]
X,Y = X[15:],Y[15:]

print("start training")
def eval_model(X,Y,model):
    cm = np.zeros((3,3),dtype=int)
    Z = model(torch.Tensor(np.stack(X)))
    Z = Z.cpu().data.numpy()
    Z = np.argmax(Z,axis=1)
    for i in range(len(Y)):
        cm += confusion_matrix(Y[i], Z[i],list(range(3)))
    return (cm[0][0]+cm[1][1]+cm[2][2])/(np.sum(cm)+1),cm

def visuall(X,Y,model):
    Z = model(torch.Tensor(np.stack(X)))
    Z = Z.cpu().data.numpy()
    Z = np.argmax(Z,axis=1)
    for i in range(len(Y)):
        im = Image.fromarray(visualizecurve(X[i][0],Y[i],Z[i]))
        im.save("tmp/"+str(i)+".png")
    

def visu():
    z = model(torch.unsqueeze(torch.Tensor(Xtest[0]),dim=0))
    z = z[0].cpu().data.numpy()
    z = np.argmax(z,axis=0)
    
    #plt.imshow(visualizecurve(Xtest[0][0],Ytest[0],z))
    #plt.show()
    im = Image.fromarray(visualizecurve(Xtest[0][0],Ytest[0],z))
    im.save("tmp/test.png")
    
    z = model(torch.unsqueeze(torch.Tensor(X[0]),dim=0))
    z = z[0].cpu().data.numpy()
    z = np.argmax(z,axis=0)
    
    #plt.imshow(visualizecurve(X[0][0],Y[0],z))
    #plt.show()
    im = Image.fromarray(visualizecurve(X[0][0],Y[0],z))
    im.save("tmp/train.png")

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
losslayer = nn.CrossEntropyLoss()
batchsize = 50
memoryofloss = collections.deque(maxlen=200)

for iteration in range(4000):
    model.train()
    model
    
    X,Y = shuffle(X,Y)
    Ybatch = np.stack(Y[0:batchsize])
    Z = model(torch.Tensor(np.stack(X[0:batchsize])))
    assert(Z.shape==(50,3,8192))
    
    # move from BATCH x NB CLASSES x 8192 to 409600 x NB CLASSES
    Z = torch.transpose(Z,1,2)
    Z = Z.contiguous().view(409600,3)
    Ybatch = Ybatch.flatten()
    Ybatch = torch.from_numpy(Ybatch).long()
    
    loss = losslayer(Z,Ybatch)
    
    memoryofloss.append(loss.cpu().data.numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if iteration%100==99:
        print(sum(memoryofloss)/len(memoryofloss))
    
    if iteration%1000 == 999:
        with torch.no_grad():
            model.eval()
            model
            print(eval_model(Xtest,Ytest,model))
            print(eval_model(X,Y,model))
            visuall(Xtest,Ytest,model)
    
    
