from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
try:
    import tfplot
except:
    pass

from  Lib.ILPRLEngine import *
import argparse
from Lib.DNF import DNF
from Lib.DNF_Ex import DNF_Ex

from Lib.CNF import CNF
from Lib.CONJ import CONJ
from Lib.MLP import MLP


import tensorflow as tf

def attention_fun(Q, K, scaled_=True, masked_=False):
    attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

    if scaled_:
        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

    if masked_:
        raise NotImplementedError

    attention = tf.nn.softmax(attention, dim=   1)  # [batch_size, sequence_length, sequence_length]
    return attention
def model_fun(data, **config):
    # data=tf.expand_dims(data,1)
    Q = tf.layers.dense(data, config['key_size'] *config['num_head'])  # [batch_size, sequence_length, hidden_dim]
    K = tf.layers.dense(data, config['key_size']*config['num_head'])  # [batch_size, sequence_length, hidden_dim]
    V = tf.layers.dense(data, config['n_classes']*config['num_head'])  # [batch_size, sequence_length, n_classes]

    Q = tf.split(Q,config['num_head'],axis=-1)
    K = tf.split(K,config['num_head'],axis=-1)
    V = tf.split(V,config['num_head'],axis=-1)
    
    Outputs=[]
    for i in range(config['num_head']):
        with tf.variable_scope('head_%d'%i):
            
            attention = attention_fun(Q[i], K[i])  # [batch_size, sequence_length, sequence_length]
            output = tf.matmul(attention, V[i])
            if config['num_head']==1:
                return output
            Outputs.append(output)

    Output = tf.layers.dense(tf.concat(Outputs,-1), 5*config['n_classes'], use_bias=False,activation=None)
    return Output




from ops import conv2d, fc
from util import log

from vqa_util import question2str, answer2str


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information
        self.XO={}
        self.pen=None
        self.config = config
        self.batch_size = self.config.batch_size
        self.img_size = self.config.data_info[0]
        self.c_dim = self.config.data_info[2]
        self.q_dim = self.config.data_info[3]
        self.a_dim = self.config.data_info[4]
        self.conv_info = self.config.conv_info
        self.acc=0
        self.feat_count = 64
        self.ilp_params = None

        # create placeholders for the input
        self.img = tf.placeholder(
            name='img', dtype=tf.float32,
            shape=[self.batch_size, self.img_size, self.img_size, self.c_dim],
        )
        self.q = tf.placeholder(
            name='q', dtype=tf.float32, shape=[self.batch_size, self.q_dim],
        )
        self.a = tf.placeholder(
            name='a', dtype=tf.float32, shape=[self.batch_size, self.a_dim],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def load_ilp_config(self):

        parser = argparse.ArgumentParser()
        batch_size = self.batch_size
        
        parser.add_argument('--BS',default=16*2 ,help='Batch Size',type=int)
        parser.add_argument('--T',default=1 ,help='Number of forward chain',type=int)
        
        parser.add_argument('--MAXTERMS',default=6 ,help='Maximum number of terms in each clause',type=int)
        parser.add_argument('--L1',default=0 ,help='Penalty for maxterm',type=float)
        parser.add_argument('--L2',default=0 ,help='Penalty for distance from binary for weights',type=float)
        parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary for each term',type=float)
        parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
        parser.add_argument('--SYNC',default=0,help='Synchronized Update',type=int)
        parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
        parser.add_argument('--TB', default=0, help='Use Tensorboard',type=int)
        parser.add_argument('--SEED',default=0,help='Random seed',type=int)
        
        self.args_ilp = parser.parse_args()

    def define_preds(self):
        nL=6
        nD=16

        D = ['%d'%i for i in range(nD)]
        self.Constants = dict( {'D':D}) 
        
        
        self.predColl = PredCollection (self.Constants)
        
        self.predColl.add_pred(dname='eq'  ,arguments=['D','D' ])
        self.predColl.add_pred(dname='ltD'  ,arguments=['D','D','D'])
        self.predColl.add_pred(dname='gtD'  ,arguments=['D','D','D'])
        self.predColl.add_pred(dname='left'  ,arguments=['D'])
        self.predColl.add_pred(dname='button'  ,arguments=['D'])
        # self.predColl.add_pred(dname='top'  ,arguments=['D'])
        # self.predColl.add_pred(dname='right'  ,arguments=['D'])
 
        self.predColl.add_pred(dname='obj'  ,arguments=['D'])
        self.predColl.add_pred(dname='rectangle'  ,arguments=['D'])
        
        #instead of color(D,C) we define a set of is_
        for i in range(nL):
            self.predColl.add_pred(dname='is_color_%d'%i  ,arguments=['D'])

        # for i in range(nD):
        #     self.predColl.add_pred(dname='is_d_%d'%i  ,arguments=['D'])

        for i in range(self.q_dim):
            self.predColl.add_pred(dname='is_q_%d'%i  ,arguments=[])

        self.predColl.add_pred(dname='closer',arguments=['D','D','D'], variables=[] ,pFunc = 
                DNF('closer',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['obj(A), obj(B), obj(C), ltD(A,B,C)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='or') 
        self.predColl.add_pred(dname='farther',arguments=['D','D','D'], variables=[] ,pFunc = 
                DNF('farther',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['obj(A), obj(B), obj(C), gtD(A,B,C)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='or') 
        
        
        self.predColl.add_pred(dname='notClosest',arguments=['D','D'], variables=['D'] ,pFunc = 
               DNF('notClosest',terms=3,init=[1,.1,-1,.1],sig=2,init_terms=['closer(A,C,B)','not obj(A)','not obj(B)','eq(A,B)'],predColl=self.predColl,fast=True,neg=True)  , use_neg=True, Fam='eq') 
        
        self.predColl.add_pred(dname='notFarthest',arguments=['D','D'], variables=['D'] ,pFunc = 
                DNF('notFarthest',terms=3,init=[1,.1,-1,.1],sig=2,init_terms=['farther(A,C,B)','not obj(A)','not obj(B)','eq(A,B)'],predColl=self.predColl,fast=True,neg=True)  , use_neg=True, Fam='eq') 
 
        exc = ['CL_%d'%i for i in range(self.a_dim)]
        
        self.predColl.add_pred(dname='qa',oname='qa',arguments=['D'] , variables=[] ,
                               pFunc =  DNF('qa',predColl=self.predColl,
                                        init_terms=['is_q_0(), is_color_0(A)',  
                                                    'is_q_1(), is_color_1(A)',  
                                                    'is_q_2(), is_color_2(A)',  
                                                    'is_q_3(), is_color_3(A)',  
                                                    'is_q_4(), is_color_4(A)',  
                                                    'is_q_5(), is_color_5(A)'] ,fast=True) ,use_neg=True, Fam='eq',  exc_conds=[('*','rep1') ] , exc_preds=exc )

         
        for k in range(0,10):
            
            post_term = [ ('and', 'qa(A)')]
            if k==6:
                post_term.append( ('and', 'not rectangle(A)'))
            if k==7:
                post_term.append( ('and', 'rectangle(A)'))

            post_terms=[]
            self.predColl.add_pred(dname='CL_%d'%k,oname='CL_%d'%k,arguments=[] , variables=['D','D'] , 
                                   pFunc =  DNF('CL_%d'%k,terms=12,init=[-1,-1,-1,-1],sig=1,predColl=self.predColl,post_terms=post_term ) 
                                   ,use_neg=True, Fam='eq' , exc_preds=exc+[ 'eq','ltD','gtD'])
        
        self.predColl.initialize_predicates()    

        self.bg = Background( self.predColl )
        for i in range(nD):
            
            self.bg.add_backgroud('eq', ('%d'%i,'%d'%i))
            
            ri,ci=int(i//4),int(i%4)
            Y1=(ri+0.5)*28+ri**2
            X1=(ci+0.5)*28+ci**2
            if Y1>64:
                self.bg.add_backgroud('button', ('%d'%i,))
            if X1<64:
                self.bg.add_backgroud('left', ('%d'%i,))
            
            for j in range(nD):
                rj,cj=int(j//4),int(j%4)
                for k in range(nD):

                    rk,ck=int(k//4),int(k%4)
                    Y2=(rj+0.5)*28+rj**2
                    X2=(cj+0.5)*28+cj**2
                    Y3=(rk+0.5)*28+rk**2
                    X3=(ck+0.5)*28+ck**2


                    d1= 1.1*(X1-X2)**2+(Y1-Y2)**2
                    d2= 1.1*(X1-X3)**2+(Y1-Y3)**2

                    if(d1<d2 and i!=j and i!=k and j!=k):
                        self.bg.add_backgroud('ltD', ('%d'%i,'%d'%j,'%d'%k))
                    if(d1>d2 and i!=j and i!=k and j!=k):
                        self.bg.add_backgroud('gtD', ('%d'%i,'%d'%j,'%d'%k))



        bg_set=[]
        self.X0=OrderedDict()
        for p in self.predColl.outpreds:
            if p.oname not in bg_set:
                tmp = tf.expand_dims( tf.constant( self.bg.get_X0(p.oname) ,tf.float32) , 0)
                self.X0[p.oname] = tf.tile( tmp , [self.batch_size,1]  )
                

        print('displaying config setting...')
       
        self.mdl = ILPRLEngine( args=self.args_ilp ,predColl=self.predColl ,bgs=None )





    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.img: batch_chunk['img'],  # [B, h, w, c]
            self.q: batch_chunk['q'],  # [B, n]
            self.a: batch_chunk['a'],  # [B, m]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def build(self, is_train=True):

        n = self.a_dim
        conv_info = self.conv_info

        # build loss and accuracy {{{
        def build_loss(logits, labels):
            
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) + .02*self.pen
            
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            self.acc = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy
        
        
        def concat_coor(o, i, d):
            coor = tf.tile(tf.expand_dims(
                [float(int(i / d)) / d, (i % d) / d], axis=0), [self.batch_size, 1])
            o = tf.concat([o, tf.to_float(coor)], axis=1)
            return o

        def CONV(img, q, scope='CONV'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)

                conv_1 = conv2d(img, conv_info[0], is_train, s_h=3, s_w=3, name='conv_1',batch_norm=True,activation_fn=tf.nn.relu)
                conv_2 = conv2d(conv_1, conv_info[1], is_train, s_h=3, s_w=3, name='conv_2',batch_norm=True,activation_fn=tf.nn.relu)
                conv_3 = conv2d(conv_2, conv_info[2], is_train, name='conv_3',batch_norm=True,activation_fn=tf.nn.relu)
                conv_4 = conv2d(conv_3, conv_info[3]*2, is_train, name='conv_4' ,batch_norm=True,activation_fn=tf.nn.relu)

        
                d = conv_4.get_shape().as_list()[1]
                all_g = []
                for i in range(d*d):
                    o_i = conv_4[:, int(i / d), int(i % d), :]
                    all_g.append(o_i)

                feat = tf.stack(all_g, axis=1)
        
         
                nD = 16
                nL=6
            
                for i in range(self.q_dim):
                    self.X0['is_q_%d'%i] = q[:,i:(i+1)]
                
        
                def makebin(x,t,fn):
                    #return fn(x)
                    fw = tf.cast( tf.greater_equal(x ,t) , tf.float32) 
                    return custom_grad( fw,fn( x ) )

                
                f = tf.layers.dense( feat, 1)
                self.X0['rectangle'] =  makebin( f[:,:,0],0.0, tf.sigmoid )
                
                #self.pen = tf.reduce_mean( self.X0['rectangle']*(1.0-self.X0['rectangle']))

                f = tf.layers.dense( feat, 7,tf.nn.softmax)
                #self.pen += tf.reduce_mean( f*(1.0-f))

                for i in range(6):
                    self.X0['is_color_%d'%i] = makebin( f[:,:,i+1],.5, tf.identity ) 
                
                self.X0['obj']   = ( 1.0-makebin( f[:,:,0],.5, tf.identity )  )
                #self.pen = tf.reduce_mean ( tf.square( tf.reduce_sum(self.X0['obj'],-1)-4.0  ))*.01
                 
                
                
                with tf.variable_scope('myscope'):
                    self.XO,L3 = self.mdl.getTSteps(self.X0)

                loss_w = 0 
                loss_sum = 0 
                vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CONV/myscope')
                
                for v in vs:
                    if v is None:
                        continue
                    
                    print(v)
                    w=tf.sigmoid(2*v)
                    loss_w += tf.reduce_mean( w*(1-.0-w))
                    loss_sum += tf.reduce_mean( tf.nn.relu(tf.reduce_sum(w,-1)-6)  ) 
            
                self.pen = loss_w+loss_sum
                
                
                os = tf.concat( [   self.XO['CL_%d'%i]   for i in range(self.a_dim)],-1)
                #os = tf.concat( [   custom_grad( sharp_sigmoid(self.XO['CL_%d'%i]-.5,7),self.XO['CL_%d'%i] )   for i in range(self.a_dim)],-1)
                return 10*os
                

        
        self.load_ilp_config()
        self.define_preds()
        
        logits = CONV(self.img, self.q, scope='CONV')
        # logits = f_phi(g, scope='f_phi')
        self.all_preds = tf.nn.softmax(logits)
        self.loss, self.accuracy = build_loss(logits, self.a)

        # Add summaries
        def draw_iqa(img, q, target_a, pred_a):
            fig, ax = tfplot.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title(question2str(q))
            ax.set_xlabel(answer2str(target_a)+answer2str(pred_a, 'Predicted'))
            return fig

        try:
            tfplot.summary.plot_many('IQA/',
                                     draw_iqa, [self.img, self.q, self.a, self.all_preds],
                                     max_outputs=4,
                                     collections=["plot_summaries"])
        except:
            pass

        tf.summary.scalar("loss/accuracy", self.accuracy)
        tf.summary.scalar("loss/cross_entropy", self.loss)
        log.warn('Successfully loaded the model.')
