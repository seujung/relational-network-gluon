import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn,utils 
import mxnet.ndarray as F
import numpy as np
import os, sys
from tqdm import trange
import pickle
import random

from models import *
from utils import *
from data_loader import load_data

# set gpu count
def setting_ctx(GPU_COUNT):
    if GPU_COUNT > 0 :
        ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
    else :
        ctx = [mx.cpu()]
    return ctx

#Define Evaluation metric
def evaluate_accuracy(data,  model, batch_size, ctx):
    data_conv = cvt_data_axis(data)
    acc = mx.metric.Accuracy()
    accuracy_mat = []
    for batch_idx in range(len(data) // (batch_size)):
        input_img, input_qst, label = ndarray_conv(data_conv,batch_idx,batch_size)
        input_img = input_img.as_in_context(ctx)
        input_qst = input_qst.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = model(input_img,input_qst)
        predictions = nd.argmax(output,axis=1)
        acc.update(preds=predictions, labels=label)
        accuracy_mat.append(acc.get()[1])
    accuracy = sum(accuracy_mat) / len(accuracy_mat)
    return accuracy

class Train(object):
    def __init__(self, config):
        ##setting hyper-parameters
        self.args = dict()
        self.args['batch_size'] = config.batch_size
        self.args['epoches'] =  config.epoches
        self.GPU_COUNT = config.GPU_COUNT
        self.ctx = setting_ctx(self.GPU_COUNT)
        self.show_status = config.show_status
        self.build_model()
        
    
    def build_model(self):
        self.model = RN_Model(self.args)        
        #parameter initialozation
        self.model.collect_params().initialize(ctx=self.ctx)
        #set optimizer
        self.trainer = gluon.Trainer(self.model.collect_params(),optimizer='adam',optimizer_params={'learning_rate':0.0001})
        #define loss function
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()
    
    def train(self):
        ##load input data
        rel_train, rel_test, norel_train, norel_test = load_data()
        
        rel_loss = list()
        norel_loss =  list()
        rel_acc = list()
        noel_acc = list()
        
        for epoch in trange(self.args['epoches']):
            cumulative_rel_loss = 0.0
            cumulative_norel_loss = 0.0
    
            input_rel_train = rel_train.copy()
            input_norel_train = norel_train.copy()
    
            #shuffle data
            random.shuffle(input_rel_train)
            random.shuffle(input_norel_train)
    
            rel = cvt_data_axis(input_rel_train)
            norel = cvt_data_axis(input_norel_train)
            
            #for batch_idx in tqdm(range(len(rel[0]) // (args['batch_size'] * 4))):
            for batch_idx in range(len(rel[0]) // (self.args['batch_size']*self.GPU_COUNT)):
                input_rel_img, input_rel_qst, rel_label = ndarray_conv(rel,batch_idx,self.args['batch_size']*self.GPU_COUNT)
        
                #data split
                input_rel_img = gluon.utils.split_and_load(input_rel_img,self.ctx)
                input_rel_qst = gluon.utils.split_and_load(input_rel_qst,self.ctx)
                rel_label = gluon.utils.split_and_load(rel_label,self.ctx)
                coord_tensor = F.zeros((self.args['batch_size'] * self.GPU_COUNT, 25, 2))
                coord_tensor = gluon.utils.split_and_load(coord_tensor,self.ctx)
                with autograd.record():
                    rel_losses = [self.loss(self.model(X,Y),Z) for X, Y, Z in zip(input_rel_img,input_rel_qst,rel_label)]
                for l in rel_losses:
                    l.backward()
                self.trainer.step(self.args['batch_size']*self.GPU_COUNT)
                for l in rel_losses:
                    cumulative_rel_loss += nd.sum(l).asscalar()

        
                input_norel_img, input_norel_qst, norel_label = ndarray_conv(norel,batch_idx,self.args['batch_size']*self.GPU_COUNT)  
        
                #data split
                input_norel_img = gluon.utils.split_and_load(input_norel_img,self.ctx)
                input_norel_qst = gluon.utils.split_and_load(input_norel_qst,self.ctx)
                norel_label = gluon.utils.split_and_load(norel_label,self.ctx)
                with autograd.record():
                    norel_losses = [self.loss(self.model(X,Y),Z) for X, Y, Z in zip(input_norel_img,input_norel_qst,norel_label)]
                for l in norel_losses:
                    l.backward()
                self.trainer.step(self.args['batch_size']*self.GPU_COUNT)
                for l in norel_losses:
                    cumulative_norel_loss += nd.sum(l).asscalar()
            rel_accuracy = evaluate_accuracy(rel_test, self.model, self.args['batch_size'], mx.gpu(0))
            norel_accuracy = evaluate_accuracy(norel_test, self.model, self.args['batch_size'], mx.gpu(0))
            if(self.show_status):
                if(epoch % 10 == 0):
                    print("Epoch {e}. rel_Loss: {rl} norel_Loss: {nrl} rel_ACC: {rl_acc} norel_ACC: {nrl_acc}".format(e=epoch+1, rl=cumulative_rel_loss/(len(rel[0]) // self.args['batch_size']), nrl=cumulative_norel_loss/ (len(rel[0]) // self.args['batch_size']), rl_acc=rel_accuracy,nrl_acc=norel_accuracy))
        
        
        
        
        
        
    
    

