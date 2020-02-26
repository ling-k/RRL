import numpy as np
import random
import collections
from time import sleep
from datetime import datetime
import sys
import tensorflow as tf
from itertools import product
from itertools import combinations_with_replacement
from .PredicateLibV5 import PredFunc

from .mylibw import *

class MLP(PredFunc):
    def __init__(self,name='',trainable=True, dims=[100,1], acts=[tf.nn.sigmoid,tf.nn.sigmoid],fast=False,dims2=[200,1],acts2=[tf.sigmoid,tf.identity]):
        
        super().__init__(name,trainable)
        self.dims = dims
        self.acts = acts
        
        
        self.dims2 = dims2
        self.acts2 = acts2
        
        self.fast=fast
        self.neg=False

    def pred_func(self,xi,xcs=None,t=0):
        
        return FC( xi, self.dims,self.acts,self.name+'fc')
           
    def pred_func2(self,xi):
        
        return FC( xi, self.dims2,self.acts2,self.name+'fc2')[:,:,0]
    
    
    def conv_weight_np(self,w):
        return None
    def conv_weight(self,w):
        return None
    def get_func(self,session,names,threshold=.2,print_th=True):
        return '' 