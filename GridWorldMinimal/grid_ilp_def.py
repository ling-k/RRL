#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:35:35 2020

@author: ali
"""


from  Lib.ILPRLEngine import *
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.CONJ import CONJ
from Lib.MLP import MLP


class ILP_MODEL(object):
    def __init__(self, num_box,is_train=True):
        self.num_box = num_box
        self.args = self.load_ilp_config()
        self.define_preds()
        self.Xo=None
        self.X0=None
        self.has_key=None
    def reset(self):        
        self.has_key=None

    def load_ilp_config(self):
        
        param = dotdict({})
        param.BS = 1
        param.T =  1
        param.W_DISP_TH = .1
        param.GPU = 1
         
        return param
        

    def define_preds(self):
        
        nCOLOR = 10
        Colors=[i for i in range(nCOLOR)]
        Pos = [i for i in range(12)]
        # Ds = ['%d'%i for i in range(self.num_box)]
        self.Constants = dict( {'C':Colors,'P':Pos,'Q':Pos}) #, 'N':['%d'%i for i in range(6)] })
        
        
        self.predColl = PredCollection (self.Constants)
        
        self.predColl.add_pred(dname='color'  ,arguments=['P','Q','C']) 
        

        for i in range(nCOLOR):
            self.predColl.add_pred(dname='is%d'%i  ,arguments=['C'])
        self.predColl.add_pred(dname='has_key'  ,arguments=['C'] )
        
        self.predColl.add_pred(dname='neq'  ,arguments=['C','C'])
        self.predColl.add_pred(dname='incq'  ,arguments=['Q','Q'])
        
        self.predColl.add_pred(dname='isBK'  ,arguments=['P','Q'], variables=['C'] ,pFunc = 
                DNF('isBK',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['color(A,B,C), is0(C)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 
        self.predColl.add_pred(dname='isAgent'  ,arguments=['P','Q'], variables=['C'] ,pFunc = 
                DNF('isAgent',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['color(A,B,C), is1(C)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 

        
        self.predColl.add_pred(dname='isGem'  ,arguments=['P','Q'], variables=['C'] ,pFunc = 
                DNF('isGem',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['color(A,B,C), is2(C)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 

        self.predColl.add_pred(dname='isItem'  ,arguments=['P','Q'], variables=[] ,pFunc = 
                DNF('isItem',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['not isBK(A,B), not isAgent(A,B)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='eq') 

          

        pt=[]
        self.predColl.add_pred(dname='move',oname='move',arguments=['P','Q'] , variables=['C' ] ,pFunc = 
              DNF('move',terms=6,init=[-1,.1,-1,.1],init_terms=[],sig=1,predColl=self.predColl,post_terms=pt) , use_neg=True, Fam='eq', exc_preds=['move','Q']+[ 'is%d'%i for i in range(nCOLOR)],exc_conds=[('*','rep1') ])        
 
        
        self.predColl.initialize_predicates()    

        self.bg = Background( self.predColl )
        #define backgrounds
        self.bg.add_backgroud('Pos0',(0,) )
        self.bg.add_backgroud('Pos11',(11,) )
        for i in range(nCOLOR):
            self.bg.add_backgroud('is%d'%i  , (i,))
            for j in range(nCOLOR):
                if i!=j:
                    self.bg.add_backgroud('neq',(i,i) )

        
        for i in range(12):
            self.bg.add_backgroud('eq',(i,i) )
            for j in range(12):
                
                if i<j:
                    self.bg.add_backgroud('lt',(i,j) )
                if j==i+1:
                    # self.bg.add_backgroud('incp',(i,j) )
                    self.bg.add_backgroud('incq',(i,j) )
            
         
            
        
        print('displaying config setting...')
        self.mdl = ILPRLEngine( args=self.args ,predColl=self.predColl ,bgs=None )


    def run( self, state):
        has_key,color = state
        
        bs= tf.shape(has_key)[0]
        self.X0=OrderedDict()
        for p in self.predColl.outpreds:
            tmp = tf.expand_dims( tf.constant( self.bg.get_X0(p.oname) ,tf.float32) , 0)
            self.X0[p.oname] = tf.tile( tmp , [bs,1]  )

        self.X0['color'] = color
        self.X0['has_key'] = has_key
        # if self.has_key is not None:
        #     self.X0['has_key'] = has_key

        self.Xo,L3 = self.mdl.getTSteps(self.X0)
        # self.has_key = self.X0['has_key'] 
        return  self.Xo["move"]


    def runtest( self, state):
        has_key,color = state
        bs= tf.shape(has_key)[0]
        
        self.X0=OrderedDict()
        for p in self.predColl.outpreds:
            tmp = tf.expand_dims( tf.constant( self.bg.get_X0(p.oname) ,tf.float32) , 0)
            self.X0[p.oname] = tf.tile( tmp , [bs,1]  )
        # if self.has_key is not None:
        #     self.X0['has_key'] = has_key

        self.X0['color'] = color
        self.X0['has_key'] = has_key
        
        self.Xo,L3 = self.mdl.getTSteps(self.X0)
        # self.has_key = self.X0['has_key'] 

        L1=0
        L2=0
        
        def get_pen(x):
            return tf.nn.relu(2*x-2)-2*tf.nn.relu(2*x-1)+tf.nn.relu(2*x)
        # for p in  self.predColl.preds:
        #     vs = tf.get_collection( p.dname)
        #     # for wi in vs:
            #     if '_AND' in wi.name:
            #         wi = p.pFunc.conv_weight(wi)
                    
            #         # L2 += tf.reduce_mean( wi*(1.0-wi))
            #         L2 += tf.reduce_mean( get_pen(wi))

            #         s = tf.reduce_sum( wi,-1)
            #         # L1 += tf.reduce_mean(  tf.nn.relu( s-7)  )
            #         s = tf.reduce_max( wi,-1)
            #         L1 += tf.reduce_mean(  tf.nn.relu( 1-s)  )


        return  self.Xo,L2*0