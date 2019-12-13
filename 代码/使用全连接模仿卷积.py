import numpy as np
import torch.nn as nn
import torch
import torch.utils.data as data

Conv=nn.Conv2d(1,1,3,1)
weight=torch.ones_like(Conv.weight.data)
bias=torch.ones_like(Conv.bias.data)
Conv.weight.data=weight
Conv.bias.data=bias


class trainset(data.Dataset):
    def __init__(self):
        self.a=torch.rand(1,1,6,6)
        y=Conv(self.a)
        self.y=y.view(-1,1)
    def __getitem__(self,index):
        self.a=torch.rand(1,1,6,6)
        y=Conv(self.a)
        self.y=y.data.view(1,-1)
        return (self.a.view(1,-1),self.y)
    def __len__(self):
        return 100000

class x2nn(nn.Module):
    def __init__(self,inputsize,outputsize):
        super(x2nn,self).__init__()
        self.L1=nn.Linear(inputsize,50)
        self.L2=nn.Linear(50,50)
        self.L3=nn.Linear(50,100)
#         self.L4=nn.Linear(50,50)
        self.L5=nn.Linear(100,outputsize)
        self.sig=nn.LeakyReLU(0.2)
        self.drop=nn.Dropout(1)
        self.batch=nn.BatchNorm1d(50)   #batchnorm 参数和上一层输出一样 一维卷积用1d 二维卷积用2d 与输出通道数一致
        
    def forward(self,x):
       # print("标准化"+str(x.shape))
        out=self.L1(x)
        out=self.sig(out)
#         out=self.drop(out)
        
        out=self.L2(out)
        out=self.sig(out)
#        out=self.drop(out)
        #out=self.batch(out)
        out=self.L3(out)
        out=self.sig(out)
#         out=self.drop(out)
        
#         out=self.L4(out)
#         out=self.sig(out)
#         out=self.drop(out)
        
        out=self.L5(out)
#        out=self.sig(out)
        return out
mode=x2nn(36,16)
ops=torch.optim.SGD(mode.parameters(),lr=0.001,momentum=0.9,weight_decay=0.1)
loss_f=nn.MSELoss()

train=trainset()
traindata=data.DataLoader(train,batch_size=500,shuffle=True)
cuda=3
mode=mode.cuda(cuda)
for j in range(10):
    print(j)
    #print(mode(test.view(1,-1).cuda(cuda)))
    for i ,k in enumerate(traindata):
        ops.zero_grad()
        
        out=mode(k[0].cuda(cuda))
        loss=loss_f(out,k[1].cuda(cuda))
        loss.backward()
        ops.step()
        if i%20==0:
            print(loss)

x=torch.ones(1,1,36)
print(net(x.cuda(cuda)))
y=x.view(1,1,6,6)
print(Conv(y))
