from  Lib.ILPRLEngine import *
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.CONJ import CONJ
from Lib.MLP import MLP
N_ACTION = 5

class ILP_MODEL(object):
    def __init__(self, DIM1,DIM2,F_COUNT=3):
        self.DIM1=DIM1
        self.DIM2=DIM2
        self.F_COUNT=F_COUNT

        
        self.args = self.load_ilp_config()
        self.define_preds()
        self.Xo=None
        self.X0=None

    def load_ilp_config(self):
        
        param = dotdict({})
        param.BS = 1
        param.T = 1
        param.W_DISP_TH = .2
        param.GPU = 1
         
        return param
        

    def define_preds(self):
        
        
        
        X = ['%d'%i for i in range(self.DIM1)]
        Y = ['%d'%i for i in range(self.DIM2)]
        self.Constants = dict( {'N':X,'Y':Y})
        
        
        self.predColl = PredCollection (self.Constants)
        
        # self.predColl.add_pred(dname='agentX'  ,arguments=['N'])
        # self.predColl.add_pred(dname='agentY'  ,arguments=['Y'])


        for i in range(self.DIM1):
            if i==0 or i==self.DIM1-1:
                self.predColl.add_pred(dname='X_%d'%i,arguments=['N'])

        for i in range(self.DIM2):
            if i==0 or i==self.DIM2-1:
                self.predColl.add_pred(dname='Y_%d'%i,arguments=['Y'])

        # self.predColl.add_pred(dname='f_1',arguments=['N','Y'], variables=[ ],pFunc = 
        #         DNF('f_1',init_terms=['agentX(A), agentY(B)' ],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 
        self.predColl.add_pred(dname='f_1',arguments=['N','Y'])
        self.predColl.add_pred(dname='f_2',arguments=['N','Y'])
        self.predColl.add_pred(dname='f_3',arguments=['N','Y'])
        self.predColl.add_pred(dname='f_4',arguments=['N','Y'])
 
        self.predColl.add_pred(dname='ltY',arguments=['Y','Y'])
        self.predColl.add_pred(dname='close',arguments=['Y','Y'])
        count_type = 'max'
        
        excs = ['action_noop', 'action_up', 'action_down','action_left','action_right', 'Q']
        w = [-1,.1,-2,.1]
        
        
        
        Alicount_type = None
        self.predColl.add_pred(dname='en_up'  ,arguments=[], variables=[ 'N','Y', 'Y'],pFunc = 
                DNF('en_up',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['f_1(A,B), X_0(A)','f_1(A,B), f_3(M_A,C), close(B,C), ltY(B,C)', 'f_1(A,B), f_2(M_A,C), close(B,C), ltY(C,B)'    ],
                    predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq',count_type=count_type, count_th=100,arg_funcs=['M']) 
        
        self.predColl.add_pred(dname='en_down'  ,arguments=[ ], variables=['N','Y',  'Y'],pFunc = 
                DNF('en_down',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['f_1(A,B), X_%d(A)'%(self.DIM1-1) , 'f_1(A,B), f_3(P_A,C), close(B,C), ltY(B,C)', 'f_1(A,B), f_2(P_A,C), close(B,C), ltY(C,B)'   ]
                    ,predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq',count_type=count_type, count_th=100,arg_funcs=['P'])  


        self.predColl.add_pred(dname='en_right'  ,arguments=[ ], variables=['N','Y','Y'],pFunc = 
                DNF('en_right',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['f_1(A,B), Y_%d(B)'%(self.DIM2-1) ,'f_1(A,B), f_3(A,C), close(B,C), ltY(B,C)' ,'f_1(A,B), f_2(A,C), close(B,C), ltY(B,C)'  ],
                    predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq',count_type=count_type, count_th=100,arg_funcs=[ ]) 
        
        self.predColl.add_pred(dname='en_left'  ,arguments=[ ], variables=['N','Y','Y'],pFunc = 
                DNF('en_left',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['f_1(A,B), Y_%d(B)'%0, 'f_1(A,B), f_2(A,C), close(B,C), ltY(C,B)','f_1(A,B), f_3(A,C), close(B,C), ltY(C,B)'   ],
                    predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq',count_type=count_type, count_th=100,arg_funcs=[ ])  

        self.predColl.add_pred(dname='en_noop'  ,arguments=[ ], variables=[],pFunc = 
                DNF('en_noop',terms=1,init=[-1,.1,-1,.1],sig=2,
                    #init_terms=['f_1(A,B), f_3(A,C), ltY(C,B)','f_1(A,B), f_2(A,C), ltY(B,C)'  ],
                    init_terms=['en_right()','en_left()'  ],
                    predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq',count_type=count_type, count_th=100) 




        
        pt = [('and', 'not en_noop()') ]
        self.predColl.add_pred(dname='action_noop',arguments=[] , variables=['N','Y','Y'  ] ,pFunc = 
            DNF('action_noop',terms=8,init=w,sig=2,predColl=self.predColl,fast=False,post_terms=pt)   , use_neg=True, Fam='eq'  , 
            exc_preds=excs,count_type=count_type,arg_funcs=['M']) 

        
        pt = [('and', 'not en_up()') ]
        self.predColl.add_pred(dname='action_up',arguments=[] , variables=['N','Y','Y'   ] ,pFunc = 
            DNF('action_up',terms=8,init=w,sig=2,predColl=self.predColl,fast=False,post_terms=pt)   , use_neg=True, Fam='eq'  , 
            exc_preds=excs,count_type=count_type, count_th=100,arg_funcs=[ 'M']) 

        
        pt = [('and', 'not en_right()') ]
        self.predColl.add_pred(dname='action_right',arguments=[] , variables=['N','Y','Y' ] ,pFunc = 
            DNF('action_right',terms=8,init=w,sig=2,predColl=self.predColl,fast=False,post_terms=pt)   , use_neg=True, Fam='eq'  , 
            exc_preds=excs,count_type=count_type,arg_funcs=['M']) 

      
        pt = [('and', 'not en_left()') ]
        self.predColl.add_pred(dname='action_left',arguments=[] , variables=['N','Y', 'Y'  ] ,pFunc = 
            DNF('action_left',terms=8,init=w,sig=2,predColl=self.predColl,fast=False,post_terms=pt)   , use_neg=True, Fam='eq'  , 
            exc_preds=excs,count_type=count_type, arg_funcs=['M'])  

        
        pt = [('and', 'not en_down()') ]
        self.predColl.add_pred(dname='action_down',arguments=[] , variables=['N','Y','Y'  ] ,pFunc = 
            DNF('action_down',terms=8,init=w,sig=2,predColl=self.predColl,fast=False,post_terms=pt)   , use_neg=True, Fam='eq'  , 
            exc_preds=excs,count_type=count_type, arg_funcs=['M']) 


         
        self.predColl.initialize_predicates()    

        self.bg = Background( self.predColl )
        #define backgrounds
          
        
       
        self.bg.add_backgroud('X_%d'%0 ,('%d'%0,) ) 
        self.bg.add_backgroud('X_%d'%(self.DIM1-1) ,('%d'%(self.DIM1-1),) ) 
        
        self.bg.add_backgroud('Y_%d'%0 ,('%d'%0,) ) 
        self.bg.add_backgroud('Y_%d'%(self.DIM2-1) ,('%d'%(self.DIM2-1),) ) 
        self.bg.add_backgroud('Y_%d'%0 ,('%d'%1,) ) 
        self.bg.add_backgroud('Y_%d'%(self.DIM2-1) ,('%d'%(self.DIM2-2),) ) 
        
           
           
                    
        for i in range(self.DIM2):
            for j in range(self.DIM2):
                if i<=j+1:
                    self.bg.add_backgroud('ltY' ,('%d'%i,'%d'%j,) )     
                if abs(i-j)<3:
                    self.bg.add_backgroud('close' ,('%d'%i,'%d'%j,) )     
                
       
        print('displaying config setting...')
        self.mdl = ILPRLEngine( args=self.args ,predColl=self.predColl ,bgs=None )


    def run( self, states):
        bs= tf.shape(states[0])[0]
        self.X0=OrderedDict()
        
        for p in self.predColl.outpreds:
            tmp = tf.expand_dims( tf.constant( self.bg.get_X0(p.oname) ,tf.float32) , 0)
            self.X0[p.oname] = tf.tile( tmp , [bs,1]  )

        
        self.X0['f_1'] = states[0]
        self.X0['f_2'] = states[1]
        self.X0['f_3'] = states[2]
        self.X0['f_4'] = states[3]

        self.Xo,L3 = self.mdl.getTSteps(self.X0)

        move = [ self.Xo[i] for i in ['action_noop','action_up','action_right','action_left', 'action_down']]
        # move[0] = tf.zeros_like( move[0] )-5
        return  tf.concat(move,-1),self.X0,self.Xo
    