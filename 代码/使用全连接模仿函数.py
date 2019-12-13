import numpy as np
import torch.nn as nn
import torch
import torch.utils.data as data

x=torch.linspace(-1,1,100000).view(100000,-1)
y=x**2
class data_(data.Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __getitem__(self,ind):
        return self.x[ind],self.y[ind]
    def __len__(self):
        return len(self.x)
data_1=data_(x,y)
datatrain=data.DataLoader(data_1,1000,True)

class x2nn(nn.Module):
    def __init__(self,inputsize,outputsize):
        super(x2nn,self).__init__()
        self.L1=nn.Linear(inputsize,50)
        self.L2=nn.Linear(50,50)
        self.L3=nn.Linear(50,100)
        self.L5=nn.Linear(100,outputsize)
        self.sig=nn.Sigmoid()
        self.sig2=nn.Tanh()
    def forward(self,x):
        out=self.L1(x)
        out=self.sig(out)
        out=self.L2(out)
        out=self.L3(out)
        out=self.sig2(out)
        out=self.L5(out)
        out=self.sig2(out)
        return out

mode=x2nn(1,1)
ops=torch.optim.SGD(mode.parameters(),lr=0.001,momentum=0.9,weight_decay=0.01)
loss_f=nn.MSELoss()
for i in range(10):
    print(i)
    for k in datatrain:
        #print(k[0])
        ops.zero_grad()
        out=mode(k[0])
        loss=loss_f(out,k[1])
        print(loss)
        loss.backward()
        ops.step()
  
