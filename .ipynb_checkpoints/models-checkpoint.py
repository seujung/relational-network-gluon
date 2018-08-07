import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn,utils 
import mxnet.ndarray as F
import numpy as np

class ConvInputModel(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(ConvInputModel,self).__init__(**kwargs)
                
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=24,kernel_size=3,strides=2,padding=1,activation='relu')
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels=24,kernel_size=3,strides=2,padding=1,activation='relu')
            self.bn2 = nn.BatchNorm()
            self.conv3 = nn.Conv2D(channels=24,kernel_size=3,strides=2,padding=1,activation='relu')
            self.bn3 = nn.BatchNorm()
            self.conv4 = nn.Conv2D(channels=24,kernel_size=3,strides=2,padding=1,activation='relu')
            self.bn4 = nn.BatchNorm()
            
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        
        return x
    
class FCOutputModel(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(FCOutputModel,self).__init__(**kwargs)
        
        with self.name_scope():
            self.fc2 = nn.Dense(256)
            self.fc3 = nn.Dense(10)
    
    def forward(self,x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.Dropout(x)
        x = self.fc3(x)
        
        return x

class RN_Model(nn.HybridBlock):
    def __init__(self,args,coord_tensor=None,**kwargs):
        super(RN_Model,self).__init__(**kwargs)
    
        with self.name_scope():
            self.conv = ConvInputModel()
            
            self.g_fc1 = nn.Dense(256,activation='relu')
            self.g_fc2 = nn.Dense(256,activation='relu')
            self.g_fc3 = nn.Dense(256,activation='relu')
            self.g_fc4 = nn.Dense(256,activation='relu')
            
            self.f_fc1 = nn.Dense(256,activation='relu')
            self.fcout = FCOutputModel()
            
            

    def forward(self,x,qst):
        with x.context:
            self.coord_tensor = F.zeros((x.shape[0], 25, 2))

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        for i in range(25):
             self.coord_tensor[:,i,:] = F.array( cvt_coord(i) )

        
        #input size = (64 * 3 * 75 * 75)
        x = self.conv(x) ## x = (64 * 24 * 5 * 5)
                
        ##g part
        mb = x.shape[0]
        n_channels = x.shape[1]
        d = x.shape[2]
        
        x_flat = x.reshape(shape=(mb,n_channels,d*d))
        x_flat = F.swapaxes(x_flat,1,2) ## (64 * 25 * 24)
        
        ##add coordinates
        x_flat = F.concat(x_flat,self.coord_tensor,dim=2)
        
        ##add question
        qst = qst.expand_dims(1)
        qst = F.repeat(qst,repeats=25,axis=1)
        qst = qst.expand_dims(2)
        
        # cast all pairs against each other
        x_i = x_flat.expand_dims(1)
        x_i = F.repeat(x_i,repeats=25,axis=1)
        
        x_j = x_flat.expand_dims(2)
        x_j = F.concat(x_j,qst,dim=3)
        x_j = F.repeat(x_j,repeats=25,axis=2)
        
        #concatenate all
        x_full = F.concat(x_i,x_j,dim=3)
        
        #reshape and apply dnn network
        x_ = x_full.reshape((-1,63))
        x_ = self.g_fc1(x_)
        x_ = self.g_fc2(x_)
        x_ = self.g_fc3(x_)
        x_ = self.g_fc4(x_)
        
        x_g = x_.reshape((mb,-1,256))
        x_g = x_g.sum(1)
        
        ##### f part #######
        x_f = self.f_fc1(x_g)
        
        return self.fcout(x_f)
    

