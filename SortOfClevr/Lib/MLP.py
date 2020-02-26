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
    def __init__(self,name='',trainable=True, dims=[100,1], acts=[tf.nn.sigmoid,tf.nn.sigmoid],fast=False):
        
        super().__init__(name,trainable)
        self.dims = dims
        self.acts = acts
        self.fast=fast
        self.neg=False
    def pred_func(self,xi,xcs=None,t=0):
        
        return FC( xi, self.dims,self.acts,self.name+'fc',initializer=tf.random_uniform_initializer(00,1.))
           
    def conv_weight_np(self,w):
        return None
    def conv_weight(self,w):
        return None
    def get_func(self,session,names,threshold=.2,print_th=True):
        return '' 