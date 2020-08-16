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
        
        
        self.predColl.add_pred(dname='sameX'  ,arguments=['N','N'])
        self.predColl.add_pred(dname='sameY'  ,arguments=['Y','Y'])
        self.predColl.add_pred(dname='X_U',arguments=['N'])
        self.predColl.add_pred(dname='X_D',arguments=['N'])
        self.predColl.add_pred(dname='Y_L',arguments=['Y'])
        self.predColl.add_pred(dname='Y_R',arguments=['Y'])
        self.predColl.add_pred(dname='ltY',arguments=['Y','Y'])
        self.predColl.add_pred(dname='close',arguments=['Y','Y'])
        
                

        
        self.predColl.add_pred(dname='agent',arguments=['N','Y'])
        self.predColl.add_pred(dname='predLR',arguments=['N','Y'])
        self.predColl.add_pred(dname='predRL',arguments=['N','Y'])
        self.predColl.add_pred(dname='food',arguments=['N','Y'])
 
        count_type = 'max'
        self.predColl.add_pred(dname='pred'  ,arguments=['N','Y'], variables=[], pFunc = 
                DNF('pred', init_terms=['predLR(A,B)', 'predRL(A,B)' ], predColl=self.predColl,fast=True)
                , use_neg=False, Fam='eq',count_type='or',arg_funcs=[]) 
        
         
        
        self.predColl.add_pred(dname='agentX'  ,arguments=['N'], variables=['Y'], pFunc = 
                DNF('agentX', init_terms=['agent(A,B)' ], predColl=self.predColl,fast=True)
                , use_neg=False, Fam='eq',count_type='or',arg_funcs=[]) 
        
        
        self.predColl.add_pred(dname='agentY'  ,arguments=['Y'], variables=['N'], pFunc = 
                DNF('agentY', init_terms=['agent(B,A)' ], predColl=self.predColl,fast=True)
                , use_neg=False, Fam='eq',count_type='or',arg_funcs=[]) 
        
          
        
        self.predColl.add_pred(dname='C1'  ,arguments=[], variables=[ 'N','N'],pFunc = 
                DNF('C1',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['agentX(A), agentX(B), not sameX(A,B)'  ],
                    predColl=self.predColl,fast=True)  , use_neg=True, Fam='eq',count_type='max', arg_funcs=[ ]) 
        
        self.predColl.add_pred(dname='C2'  ,arguments=[], variables=[ 'Y','Y'],pFunc = 
                DNF('C2',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['agentY(A), agentY(B), not close(A,B)'  ],
                    predColl=self.predColl,fast=True)  , use_neg=True, Fam='eq',count_type='max', arg_funcs=[ ]) 
        
        
        self.predColl.add_pred(dname='C3'  ,arguments=[], variables=['N' ,'Y'],pFunc = 
                DNF('C3',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['predLR(A,B), predRL(A,B)',
                                'agent(A,B), predRL(A,B)',
                                'agent(A,B), predLR(A,B)',
                                ],
                    predColl=self.predColl,fast=True)  , use_neg=True, Fam='eq',count_type='max', arg_funcs=[ ]) 
        
        
        self.predColl.add_pred(dname='C4'  ,arguments=[], variables=[ 'N','Y'],pFunc = 
                DNF('C4',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['agent(A,B)'  ],
                    predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq',count_type='max', arg_funcs=[ ]) 
         
        
        excs = ['action_noop', 'action_up', 'action_down','action_left','action_right', 'Q']
        w = [-1,.1,-3,.1]
        
         
        self.predColl.add_pred(dname='en_up'  ,arguments=[], variables=[ 'N','Y', 'Y'],pFunc = 
                DNF('en_up',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['agent(A,B), X_U(A)','agent(A,B), pred(M_A,C), close(B,C)' ],
                    predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq',count_type=count_type, count_th=100,arg_funcs=['M']) 
        
        self.predColl.add_pred(dname='en_down'  ,arguments=[ ], variables=['N','Y',  'Y'],pFunc = 
                DNF('en_down',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['agent(A,B), X_D(A)'  , 'agent(A,B), pred(P_A,C), close(B,C)' ]
                    ,predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq',count_type=count_type, count_th=100,arg_funcs=['P'])  


        self.predColl.add_pred(dname='en_right'  ,arguments=[ ], variables=['N','Y','Y'],pFunc = 
                DNF('en_right',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['agent(A,B), Y_R(B)'  ,'agent(A,B), pred(A,C), close(B,C), ltY(B,C)' ],
                    predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq',count_type=count_type, count_th=100,arg_funcs=[ ]) 
        
        self.predColl.add_pred(dname='en_left'  ,arguments=[ ], variables=['N','Y','Y'],pFunc = 
                DNF('en_left',terms=1,init=[-1,.1,-1,.1],sig=2,
                    init_terms=['agent(A,B), Y_L(B)', 'agent(A,B), pred(A,C), close(B,C), ltY(C,B)'],
                    predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq',count_type=count_type, count_th=100,arg_funcs=[ ])  

        self.predColl.add_pred(dname='en_noop'  ,arguments=[ ], variables=[],pFunc = 
                DNF('en_noop',terms=1,init=[-1,.1,-1,.1],sig=2,
                    #init_terms=['agent(A,B), predRL(A,C), ltY(C,B)','agent(A,B), predLR(A,C), ltY(B,C)'  ],
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
         
          
        
       
        self.bg.add_backgroud('X_U' ,('%d'%0,) ) 
        self.bg.add_backgroud('X_D' ,('%d'%(self.DIM1-1),) ) 
        self.bg.add_backgroud('Y_L' ,('%d'%0,) ) 
        self.bg.add_backgroud('Y_R' ,('%d'%(self.DIM2-1),) ) 
        
        
        
           
           
        for i in range(self.DIM1):
            self.bg.add_backgroud('sameX' ,('%d'%i,'%d'%i,) )     
        
        for i in range(self.DIM2):
            self.bg.add_backgroud('sameY' ,('%d'%i,'%d'%i,) )     
        
        for i in range(self.DIM2):
            for j in range(self.DIM2):
                if i<j:
                    self.bg.add_backgroud('ltY' ,('%d'%i,'%d'%j,) )     
                if abs(i-j)<2:
                    self.bg.add_backgroud('close' ,('%d'%i,'%d'%j,) )     
                
       
        print('displaying config setting...')
        self.mdl = ILPRLEngine( args=self.args ,predColl=self.predColl ,bgs=None )


    def run( self, states):
        bs= tf.shape(states[0])[0]
        self.X0=OrderedDict()
        
        for p in self.predColl.outpreds:
            tmp = tf.expand_dims( tf.constant( self.bg.get_X0(p.oname) ,tf.float32) , 0)
            self.X0[p.oname] = tf.tile( tmp , [bs,1]  )

        
        self.X0['agent'] = states[0]
        self.X0['predLR'] = states[1]
        self.X0['predRL'] = states[2]
        self.X0['food'] = states[3]

        self.Xo,L3 = self.mdl.getTSteps(self.X0)
        
        xo_r = {}
        for p in self.Xo:
            xo_r[p]= tf.reshape( self.Xo[p], [-1] + [len(self.predColl.constants[i]) for i in self.predColl[p].arguments] )
        move = [ self.Xo[i] for i in ['action_noop','action_up','action_right','action_left', 'action_down']]
        # move = [ 1.0-self.Xo[i] for i in ['en_noop','en_up','en_right','en_left', 'en_down']]
        # move[0] = tf.zeros_like( move[0] )-5
        return  tf.concat(move,-1),self.X0,self.Xo,xo_r
    