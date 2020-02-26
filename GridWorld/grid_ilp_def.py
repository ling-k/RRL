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
        # self.predColl.add_pred(dname='Pos0'  ,arguments=['P'])
        # self.predColl.add_pred(dname='Pos11'  ,arguments=['P'])
        
        self.predColl.add_pred(dname='has_key'  ,arguments=['C'] )
        
        self.predColl.add_pred(dname='neq'  ,arguments=['C','C'])
        # self.predColl.add_pred(dname='lt'  ,arguments=['P','P'])
        # self.predColl.add_pred(dname='incp'  ,arguments=['P','P'])
        self.predColl.add_pred(dname='incq'  ,arguments=['Q','Q'])
        
        self.predColl.add_pred(dname='isBK'  ,arguments=['P','Q'], variables=['C'] ,pFunc = 
                DNF('isBK',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['color(A,B,C), is0(C)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 
        self.predColl.add_pred(dname='isAgent'  ,arguments=['P','Q'], variables=['C'] ,pFunc = 
                DNF('isAgent',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['color(A,B,C), is1(C)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 

        
        self.predColl.add_pred(dname='isGem'  ,arguments=['P','Q'], variables=['C'] ,pFunc = 
                DNF('isGem',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['color(A,B,C), is2(C)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 

        self.predColl.add_pred(dname='isItem'  ,arguments=['P','Q'], variables=[] ,pFunc = 
                DNF('isItem',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['not isBK(A,B), not isAgent(A,B)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='eq') 

        # self.predColl.add_pred(dname='sameColor'  ,arguments=['P','P','P','P'], variables=['C'] ,pFunc = 
        #         DNF('sameColor',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['color(A,B,E), color(C,D,E)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='or') 
        

        self.predColl.add_pred(dname='locked'  ,arguments=['P','Q'], variables=['Q'] ,pFunc = 
                DNF('locked',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['isItem(A,B), isItem(A,C), incq(B,C)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 
        self.predColl.add_pred(dname='isLock'  ,arguments=['P','Q'], variables=['Q'] ,pFunc = 
                DNF('isLock',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['isItem(A,B), isItem(A,C), incq(C,B)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 
        # self.predColl.add_pred(dname='locked1'  ,arguments=['P','P'], variables=['P'] ,pFunc = 
        #         DNF('locked1',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['isItem(A,B), isItem(C,B), inc(C,B)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='or') 
        
        #self.predColl.add_pred(dname='LockColor'  ,arguments=['C','C'], variables=['P','Q','Q'] ,pFunc = 
        #        DNF('LockColor',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['isItem(C,D), isItem(C,E), color(C,D,A), color(C,E,B), incq(D,E)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 
        
        #self.predColl.add_pred(dname='loosekey'  ,arguments=['P','Q'], variables=[] ,pFunc = 
        #        DNF('loosekey',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['isItem(A,B), not isLock(A,B), not locked(A,B)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='eq') 

        #self.predColl.add_pred(dname='key_color'  ,arguments=['P','Q','C'], variables=['Q'] ,pFunc = 
        ##        DNF('key_color',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['isItem(A,B), isLock(A,B), color(A,B,C)',
        #        'isItem(A,B), locked(A,D), incq(D,B), color(A,D,C)' ],predColl=self.predColl,fast=True)  , use_neg=True, Fam='eq') 

        #self.predColl.add_pred(dname='inGoal'  ,arguments=['C'], variables=['C'] ,pFunc = 
        #        DNF('inGoal',terms=2,init=[1,.1,-1,.1],sig=2,init_terms=['is2(A)' , 'inGoal(B), LockColor(B,A)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='or') 


        # self.predColl.add_pred(dname='valid',oname='valid',arguments=['P','P'] , variables=['C'] ,pFunc = 
        #     DNF('valid',terms=2,init_terms=['loosekey(A,B)', 'isItem(A,B), isLock(A,B), color(A,B,C), has_key(C)'],sig=2,predColl=self.predColl,fast=True )  , use_neg=True, Fam='eq') 
        
        #,post_terms=[('and','isItem(A,B)')] ) 
        
        # self.predColl.add_pred(dname='move',oname='move',arguments=['P','P'] , variables=['C'] ,pFunc = 
        #   DNF('move',terms=4,init=[-1,.1,-1,.1],sig=2,predColl=self.predColl,fast=False,post_terms=[('and','not isBK(A,B)')]) , use_neg=True, Fam='eq') 

        # self.predColl.add_pred(dname='move',oname='move',arguments=['P','P'] , variables=['C','C'] ,pFunc = 
        #     DNF('move',terms=8,init=[1,-1,-2,.3],sig=1,predColl=self.predColl,fast=False,post_terms=[('and','isItem(A,B)')]) , use_neg=True, Fam='eq', exc_preds=['move']+[ 'is%d'%i for i in range(nCOLOR)],exc_conds=[('*','rep1') ]) 
        #   DNF('move',terms=5,init=[1,-3,-1,.1],sig=2,predColl=self.predColl,fast=False,post_terms=[('and','isItem(A,B)')]) , use_neg=False, Fam='eq', exc_preds=['move']+[ 'is%d'%i for i in range(nCOLOR)],exc_conds=[('*','rep1') ]) 
            # DNF('move',terms=10,init=[1,-1,-1,.1],sig=2,predColl=self.predColl,fast=False,post_terms=[('and','isItem(A,B)'),('or','loosekey(A,B)')]) , use_neg=True, Fam='eq', exc_preds=['move', 'color']) 
        # DNF('move',terms=5,init_terms=['color(A,B,C), inGoal(C), loosekey(A,B)','key_color(A,B,D), color(A,B,C), inGoal(D), isLock(A,B), inGoal(D), LockColor(D,C), has_key(C)'],sig=2,predColl=self.predColl,fast=True )  , use_neg=True, Fam='eq') 
            # DNF('move',terms=2,init_terms=['loosekey(A,B)', 'isLock(A,B), color(A,B,C), has_key(C)'],sig=2,predColl=self.predColl,fast=True )  , use_neg=True, Fam='eq') 
        
        # for i in range(10):
            
        #     self.predColl.add_pred(dname='aux%d'%i,oname='aux%d'%i,arguments=['P','Q'] , variables=['C','C'] ,pFunc = 
        #         CONJ('aux%d'%i,init=[-1,1],init_terms=[],sig=1,predColl=self.predColl,post_terms=[]) , use_neg=False, Fam='eq', exc_preds=['move']+[ '%d'%i for i in range(nCOLOR)]+['aux%d'%i for i in range(10)],exc_conds=[('*','rep1') ])

        # it=[]

        #self.predColl.add_pred(dname='Q',oname='Q',arguments=['P','Q'] , variables=['P',  'C' ] ,pFunc = 
        #      DNF('Q',terms=6,init=[-1,.1,-1,1],init_terms=[],sig=1,predColl=self.predColl ) , use_neg=True, Fam='eq', exc_preds=['move','Q']+[ 'is%d'%i for i in range(nCOLOR)],exc_conds=[('*','rep1') ])        
        
        #self.predColl.add_pred(dname='move',oname='move',arguments=['P','Q'] , variables=['C' ] ,pFunc = 
        #      DNF('move',terms=6,init=[1,-1,-2,1],init_terms=[],sig=2,predColl=self.predColl,post_terms=[('and','isItem(A,B)'),('and','inGoal(C)') ]) , use_neg=True, Fam='eq', exc_preds=['move','Q']+[ 'is%d'%i for i in range(nCOLOR)],exc_conds=[('*','rep1') ])        

        pt=[('and','isItem(A,B)')]
        pt=[]
        self.predColl.add_pred(dname='move',oname='move',arguments=['P','Q'] , variables=['C' ] ,pFunc = 
              DNF('move',terms=6,init=[-1,.1,-1,.1],init_terms=[],sig=1,predColl=self.predColl,post_terms=pt) , use_neg=True, Fam='eq', exc_preds=['move','Q']+[ 'is%d'%i for i in range(nCOLOR)],exc_conds=[('*','rep1') ])        


        # for i in range(5):            
        #     if i==0:
        #         tp="eq"
        #         # it=['color(A,B,C), inGoal(C), loosekey(A,B)']
        #     else:
        #         tp="or"
        #         it=[]
        #     vvv=1
        #     vv=-1
        #     Vars = ['C','C' ]
        #     if i==5:
        #         Vars=['C','C']
        #         vvv=2
        #         vv=-1
            
        #     # if i==1:
        #     #     it=['key_color(A,B,D), color(A,B,C), isLock(A,B), inGoal(D), LockColor(D,C), has_key(C)']
                
        #     self.predColl.add_pred(dname='move%d'%i,oname='move',arguments=['P','Q'] , variables=Vars ,pFunc = 
        #         CONJ('move%d'%i,init=[vv,vvv],init_terms=it,sig=1,predColl=self.predColl,post_terms=[('and','isItem(A,B)')]) , use_neg=False, Fam=tp, exc_preds=['move']+[ '%d'%i for i in range(nCOLOR)],exc_conds=[('*','rep1') ])
        # self.predColl.add_pred(dname='move2',oname='move',arguments=['P','P'] , variables=['C'] ,pFunc = 
        #   CONJ('move2',init=[-1,.1],sig=2,predColl=self.predColl,fast=False) , use_neg=True, Fam='eq', exc_preds=['move'],exc_conds=[('*','rep1') ]) 
        
        # MLP('move',dims=[64,64,1], acts=[relu1,relu1,tf.nn.sigmoid] )  , use_neg=False, Fam='eq') 
        #    DNF('move',terms=10,init=[1,-1,-1,.1],sig=2,predColl=self.predColl,fast=False,post_terms=[('and','isItem(A,B)'),('or','loosekey(A,B)')]) , use_neg=True, Fam='eq', exc_preds=['move', 'color']) 

        # self.predColl.add_pred(dname='move',oname='move',arguments=['P','P'] , variables=['C'] ,pFunc = 
        #   DNF('move',terms=2,init_terms=['loosekey(A,B)', 'isLock(A,B), color(A,B,C), has_key(C)'],sig=2,predColl=self.predColl,fast=True )  , use_neg=True, Fam='eq') 

        # self.predColl.add_pred(dname='has_key'  ,arguments=['C'], variables=['P','P'] ,pFunc = 
        #         DNF('has_key',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['loosekey(B,C), move(B,C), color(B,C,A)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='or') 

        # self.predColl.add_pred(dname='enableLeft'  ,arguments=['P','P'], variables=['P'] ,pFunc = 
        #         DNF('enableLeft',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['agent(A,B), inc(C,A), not pos0(C)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='or') 

        # self.predColl.add_pred(dname='enableRight'  ,arguments=['P','P'], variables=['P'] ,pFunc = 
        #         DNF('enableRight',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['agent(A,B), not locked(A,B), not Pos0(A)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='or') 

        # self.predColl.add_pred(dname='enableLeft'  ,arguments=['P','P'], variables=['P'] ,pFunc = 
        #         DNF('enableLeft',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['agent(A,B), not locked(A,B), not Pos0(A)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='or') 

        # self.predColl.add_pred(dname='enableLeft'  ,arguments=['P','P'], variables=['P'] ,pFunc = 
        #         DNF('enableLeft',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['agent(A,B), not locked(A,B), not Pos0(A)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='or') 



       
        # self.predColl.add_pred(dname='moveLeft',oname='moveLeft',arguments=[] , variables=['P','P','C'] ,pFunc = 
        #         DNF('moveLeft',terms=6,init=[1,.1,-1,.1],sig=2,predColl=self.predColl,fast=False)  , use_neg=True, Fam='eq') 
        # self.predColl.add_pred(dname='moveRight',oname='moveRight',arguments=[] , variables=['P','P','C'] ,pFunc = 
        #         DNF('moveRight',terms=6,init=[1,.1,-1,.1],sig=2,predColl=self.predColl,fast=False)  , use_neg=True, Fam='eq') 
        # self.predColl.add_pred(dname='moveLeft',oname='moveLeft',arguments=[] , variables=['P','P','C'] ,pFunc = 
        #         DNF('moveLeft',terms=6,init=[1,.1,-1,.1],sig=2,predColl=self.predColl,fast=False)  , use_neg=True, Fam='eq') 
        # self.predColl.add_pred(dname='moveLeft',oname='moveLeft',arguments=[] , variables=['P','P','C'] ,pFunc = 
        #         DNF('moveLeft',terms=6,init=[1,.1,-1,.1],sig=2,predColl=self.predColl,fast=False)  , use_neg=True, Fam='eq') 
        
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