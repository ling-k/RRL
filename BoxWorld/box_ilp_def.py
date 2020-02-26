from  Lib.ILPRLEngine import *
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.CONJ import CONJ


class ILP_MODEL(object):
    def __init__(self, num_box,is_train=True):
        self.num_box = num_box
        self.args = self.load_ilp_config()
        self.define_preds()
        self.Xo=None
        self.X0=None

    def load_ilp_config(self):
        
        param = dotdict({})
        param.BS = 1
        param.T = 1
        param.W_DISP_TH = .1
        param.GPU = 1
         
        return param 
        

    def define_preds(self):
        
        
        
        Box = ['%d'%i for i in range(self.num_box+1)]
        Ds = ['%d'%i for i in range(self.num_box+1)]
        self.Constants = dict( {'C':Box,'D':Ds})  
        
        
        self.predColl = PredCollection (self.Constants)
        
        self.predColl.add_pred(dname='posH',arguments=['C','D'])
        self.predColl.add_pred(dname='posV',arguments=['C','D'])
        
        self.predColl.add_pred(dname='is_one'  ,arguments=['D'])
        self.predColl.add_pred(dname='lt'  ,arguments=['D','D'])
        self.predColl.add_pred(dname='inc'  ,arguments=['D','D'])
        self.predColl.add_pred(dname='same'  ,arguments=['C','C'])
        self.predColl.add_pred(dname='is_floor'  ,arguments=['C'])
        
        self.predColl.add_pred(dname='is_blue'  ,arguments=['C'])
        
        
        
        

        self.predColl.add_pred(dname='same_col'  ,arguments=['C','C' ], variables=['D'],pFunc = 
                DNF('same_col',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['posH(A,C), posH(B,C)' ],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 

        self.predColl.add_pred(dname='above'  ,arguments=['C','C' ], variables=['D','D'],pFunc = 
                DNF('above',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['same_col(A,B), posV(A,C), posV(B,D), lt(D,C)' ],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 

        self.predColl.add_pred(dname='below'  ,arguments=['C','C' ], variables=['D','D'],pFunc = 
                DNF('below',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['same_col(A,B), posV(A,C), posV(B,D), lt(C,D)' ],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 

        self.predColl.add_pred(dname='on'  ,arguments=['C','C' ], variables=['D','D'],pFunc = 
                DNF('on',terms=2,init=[1,.1,-1,.1],sig=2,init_terms=['is_floor(B), posV(A,C), is_one(C)', 
                'same_col(A,B), posV(A,C), posV(B,D), inc(D,C)' ],predColl=self.predColl,fast=True)  , use_neg=False, Fam='eq') 

        self.predColl.add_pred(dname='isCovered',arguments=['C'], variables=['C'] ,pFunc = 
                DNF('isCovered',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['on(B,A), not is_floor(A)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='eq') 
        
        
        self.predColl.add_pred(dname='lower',oname='lower',arguments=['C','C'] , variables=['D','D'] ,pFunc = 
                DNF('lower',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['posV(A,C), posV(B,D), lt(C,D)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='eq') 

        self.predColl.add_pred(dname='moveable',oname='moveable',arguments=['C','C'] , variables=[] ,pFunc = 
                DNF('moveable',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['not isCovered(A), not isCovered(B), not same(A,B), not is_floor(A), not on(A,B), not is_blue(A), not is_floor(B), not lower(B,A)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='eq') 

        
        pt=[ ('and','moveable(A,B)'),]
        self.predColl.add_pred(dname='move',oname='move',arguments=['C','C'] , variables=[] ,pFunc = 
                DNF('move',terms=4,init=[1,-1,-1,.1],sig=2,predColl=self.predColl,fast=False,post_terms=pt)  , use_neg=True, Fam='eq',exc_preds=[], exc_conds=[('*','rep1')]) 
        
        
        self.predColl.initialize_predicates()    
        
        
        self.bg = Background( self.predColl )
       
        #define backgrounds
        self.bg.add_backgroud('is_floor',('0',) )
        self.bg.add_backgroud('is_one',('1',) )
        self.bg.add_backgroud('is_blue' ,('1',) )         
        

        for i in range(self.num_box+1):
            self.bg.add_backgroud('same',('%d'%i,'%d'%i) )
            
            if '%d'%(i+1)  in Ds:
                self.bg.add_backgroud('inc',('%d'%i,'%d'%(i+1) ) )


        for i in range(self.num_box+1):
            for j in range(self.num_box+1):
                if i<j:
                    self.bg.add_backgroud('lt',('%d'%i,'%d'%(j) ) )

        
            
        
        print('displaying config setting...')
        self.mdl = ILPRLEngine( args=self.args ,predColl=self.predColl ,bgs=None )


    def run( self, state_in_x,state_in_y):
        bs= tf.shape(state_in_x)[0]
        self.X0=OrderedDict()
        
        for p in self.predColl.outpreds:
            tmp = tf.expand_dims( tf.constant( self.bg.get_X0(p.oname) ,tf.float32) , 0)
            self.X0[p.oname] = tf.tile( tmp , [bs,1]  )
        
        flx=np.zeros( (self.num_box+1,self.num_box+1), dtype=np.float32)
        flx[0,:] = 1
        xx = tf.pad(state_in_x, [[0,0],[1,0],[1,0]])+ flx[np.newaxis,:,:]
        self.X0['posH'] = tf.reshape(xx,[-1,(self.num_box+1)**2])
        
        fly=np.zeros( (self.num_box+1,self.num_box+1), dtype=np.float32)
        fly[0,0] = 1
        yy = tf.pad(state_in_y, [[0,0],[1,0],[1,0]]) + fly[np.newaxis,:,:]
        self.X0['posV'] = tf.reshape(yy,[-1,(self.num_box+1)**2])
        self.Xo,L3 = self.mdl.getTSteps(self.X0)

        return  self.Xo

     