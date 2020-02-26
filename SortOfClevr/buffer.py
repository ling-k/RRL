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
from Lib.CNF import CNF
from Lib.CONJ import CONJ
from Lib.MLP import MLP


def feat2scores(feat):

    pos = tf.constant(  np.eye(5) , tf.float32 )
    w = weight_variable( [ feat.shape[-1].value,5]  )
    b = bias_variable([5,])

    m = tf.matmul( feat,w+b)
    mm = tf.matmul( m , pos)
    mmm = tf.nn.softmax( mm,1)

    return tf.transpose( mmm, [0,2,1] )



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
        # extra_feat_count=16
 
        # parser.add_argument('--CHECK_CONVERGENCE',default=1,help='Check for convergence',type=int)
        # parser.add_argument('--SHOW_PRED_DETAILS',default=0,help='Print predicates definition details',type=int)

        parser.add_argument('--BS',default=16,help='Batch Size',type=int)
        parser.add_argument('--T',default=1 ,help='Number of forward chain',type=int)
        # parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.01} , help='Learning rate schedule',type=dict)

        # parser.add_argument('--BINARAIZE', default=1 , help='Enable binrizing at fast convergence',type=int)
        # parser.add_argument('--MAX_DISP_ITEMS', default=10 , help='Max number  of facts to display',type=int)
        # parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)
        # parser.add_argument('--W_DISP_TH', default=.2 , help='Display Threshold for weights',type=int)
        # parser.add_argument('--ITER', default=400000, help='Maximum number of iteration',type=int)
        # parser.add_argument('--ITER2', default=200, help='Epoch',type=int)
        # parser.add_argument('--PRINTPRED',default=1,help='Print predicates',type=int)
        # parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
        parser.add_argument('--MAXTERMS',default=6 ,help='Maximum number of terms in each clause',type=int)
        parser.add_argument('--L1',default=0 ,help='Penalty for maxterm',type=float)
        parser.add_argument('--L2',default=0 ,help='Penalty for distance from binary for weights',type=float)
        parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary for each term',type=float)
        parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
        parser.add_argument('--SYNC',default=0,help='Synchronized Update',type=int)
        # parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
        # parser.add_argument('--FILT_TH_MEAN', default=.5 , help='Fast convergence total loss threshold MEAN',type=float)
        # parser.add_argument('--FILT_TH_MAX', default=.5 , help='Fast convergence total loss threshold MAX',type=float)
        # parser.add_argument('--OPT_TH', default=.05 , help='Per value accuracy threshold',type=float)
        # parser.add_argument('--PLOGENT', default=.50 , help='Crossentropy coefficient',type=float)
        # parser.add_argument('--BETA1', default=.90 , help='ADAM Beta1',type=float)
        # parser.add_argument('--BETA2', default=.999 , help='ADAM Beta2',type=float)
        # parser.add_argument('--EPS', default=1e-6, help='ADAM Epsillon',type=float)
        parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
        # parser.add_argument('--LOGDIR', default='./logs/Logic', help='Log Dir',type=str)
        parser.add_argument('--TB', default=0, help='Use Tensorboard',type=int)
        parser.add_argument('--SEED',default=0,help='Random seed',type=int)
        # parser.add_argument('--ADDGRAPH', default=1, help='Add graph to Tensorboard',type=int)
        # parser.add_argument('--CLIP_NORM', default=0, help='Clip gradient',type=float)
        self.args_ilp = parser.parse_args()

    def define_preds(self):
        nC=6
        nD=16

        C = ['%d'%i for i in range(nC)]
        D = ['%d'%i for i in range(nD)]

        self.Constants = dict( { 'C':C, 'D':D}) #, 'N':['%d'%i for i in range(6)] })
        
        
        self.predColl = PredCollection (self.Constants)
        
        self.predColl.add_pred(dname='pos'  ,arguments=['C','D' ])
        self.predColl.add_pred(dname='question'  ,arguments=['C'])
        

        self.predColl.add_pred(dname='eq'  ,arguments=['D','D' ])
        self.predColl.add_pred(dname='ltD'  ,arguments=['D','D','D'])
        self.predColl.add_pred(dname='gtD'  ,arguments=['D','D','D'])
        
        self.predColl.add_pred(dname='left'  ,arguments=['D'])
        self.predColl.add_pred(dname='button'  ,arguments=['D'])

        for i in range(self.q_dim):
            self.predColl.add_pred(dname='is_q_%d'%i  ,arguments=[])

        for i in range(nC):
            self.predColl.add_pred(dname='is_color_%d'%i  ,arguments=['C'])
   
        
        # ,pFunc = 
        #         DNF('obj',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['is_l_0(A)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='or') 
        self.predColl.add_pred(dname='rectangle'  ,arguments=['C'])
        self.predColl.add_pred(dname='exist'  ,arguments=['C'])
        
        self.predColl.add_pred(dname='eqC'  ,arguments=['C','C'],variables=['D','D'] ,pFunc =
            DNF('eqC',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['pos(A,C), pos(B,D), eq(C,D)'],predColl=self.predColl,fast=True)  , use_neg=True, Fam='or') 
        

        

        
        self.predColl.add_pred(dname='closer',arguments=['C','C','C'], variables=['D','D','D'] ,pFunc = 
                DNF('closer',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['pos(A,D), pos(B,E), pos(C,F), exist(A), exist(B), exist(C), ltD(D,E,F)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='or') 
        
        self.predColl.add_pred(dname='farther',arguments=['C','C','C'], variables=['D','D','D'] ,pFunc = 
                DNF('farther',terms=1,init=[1,.1,-1,.1],sig=2,init_terms=['pos(A,D), pos(B,E), pos(C,F), exist(A), exist(B), exist(C), gtD(D,E,F)'],predColl=self.predColl,fast=True)  , use_neg=False, Fam='or') 
        
        
        self.predColl.add_pred(dname='closest',arguments=['C','C'], variables=['C'] ,pFunc = 
                DNF('closest',terms=4,init=[1,.1,-1,.1],sig=2,init_terms=['closer(A,C,B)','not exist(A)','not exist(B)','eqC(A,B)'],predColl=self.predColl,fast=True,neg=True)  , use_neg=True, Fam='eq') 
        
        self.predColl.add_pred(dname='farthest',arguments=['C','C'], variables=['C'] ,pFunc = 
                DNF('farthest',terms=4,init=[1,.1,-1,.1],sig=2,init_terms=['farther(A,C,B)','not exist(A)','not exist(B)','eqC(A,B)'],predColl=self.predColl,fast=True,neg=True)  , use_neg=True, Fam='eq') 

        
    
        exc = ['CL_%d'%i for i in range(self.a_dim)]
        exc=[]
         

        for k in range(0,self.a_dim):
            
        # # #     # exc = ['CL_%d'%i for i in range(k,self.a_dim)]
        # # #     # self.predColl.add_pred(dname='CL0_%d'%k,oname='CL_%d'%k,arguments=[] , variables=[] , pFunc =  DNF('CL0_%d'%k,predColl=self.predColl,terms=10 ,init=[-1,.1,-1,.1],sig=2) ,use_neg=True, Fam='eq', exc_preds=exc )
        #     # self.predColl.add_pred(dname='CL1_%d'%k,oname='CL_%d'%k,arguments=[] , variables=['D'] , pFunc =  DNF('CL1_%d'%k,predColl=self.predColl,terms=6 ,init=[-1,.1,-1,.1],sig=2) ,use_neg=True, Fam='eq',exc_conds=[('*','rep1') ] )
            if k==0:
                post_terms=[]
            else:
                post_terms=[ ('and', 'not CL_%d()'%j ) for j in range(k)]
            post_terms=[]
            
            self.predColl.add_pred(dname='CL_%d'%k,oname='CL_%d'%k,arguments=[] , variables=['C','C','D'] , pFunc =  DNF('CL_%d'%k,predColl=self.predColl,terms=14,init=[-1,-1,-1,.1],sig=2, post_terms=post_terms) ,use_neg=True, Fam='eq',exc_conds=[('*','rep1') ] ,exc_preds=exc ) #
        # # #     # self.predColl.add_pred(dname='CL_%d'%k,oname='CL_%d'%k,arguments=[] , variables=['D','D'] , pFunc =  MLP('CL_%d'%k,dims=[200,1], acts=[tf.nn.relu,tf.sigmoid] ) ,use_neg=False, Fam='eq', exc_preds=exc )
    

        self.predColl.initialize_predicates()    

        self.bg = Background( self.predColl )

 
        # self.bg.add_backgroud('notExist', ('%d'%(nD-1),))

        for i in range(nC):
            self.bg.add_backgroud('is_color_%d'%i, ('%d'%i,))


        for i in range(nD):
            ri,ci=int(i//4),int(i%4)
            
            if ri>=2:
                self.bg.add_backgroud('button', ('%d'%i,))
            if ci<2:
                self.bg.add_backgroud('left', ('%d'%i,))


            # self.bg.add_backgroud('-%d'%i  , ('%d'%i,) )
            
            self.bg.add_backgroud('eq', ('%d'%i,'%d'%i))
            for j in range(nD):
                rj,cj=int(j//4),int(j%4)
                for k in range(nD):


                    rk,ck=int(k//4),int(k%4)

                    d1=(ri-rj)**2+(ci-cj)**2
                    d2=(ri-rk)**2+(ci-ck)**2
                    if(d1<d2 and i!=j and i!=k and j!=k):
                        self.bg.add_backgroud('ltD', ('%d'%i,'%d'%j,'%d'%k))
                    if(d1>d2 and i!=j and i!=k and j!=k):
                        self.bg.add_backgroud('gtD', ('%d'%i,'%d'%j,'%d'%k))




            a = '%d'%i

            # self.bg.add_backgroud('is_c_%d'%i  , (a,) )
            # self.bg.add_backgroud('is_r_%d'%i  , (a,) )
            
            # self.bg.add_backgroud('eqC'  , (a,a) )
            # self.bg.add_backgroud('eqR'  , (a,a) )
            
            # for j in range(nC):
            #     if i<j:
            #         self.bg.add_backgroud('ltC', ('%d'%i, '%d'%j))
            #         self.bg.add_backgroud('ltR', ('%d'%i, '%d'%j))   



        bg_set=[]
        self.X0=OrderedDict()
        for p in self.predColl.outpreds:
            if p.oname not in bg_set:
                tmp = tf.expand_dims( tf.constant( self.bg.get_X0(p.oname) ,tf.float32) , 0)
                self.X0[p.oname] = tf.tile( tmp , [self.batch_size,1]  )
                

        print('displaying config setting...')
        # for arg in vars(args):
        #         print( '{}-{}'.format ( arg, getattr(args, arg) ) )
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
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits*10, labels=labels)
            # loss = tf.reduce_sum( neg_ent_loss (labels,logits) , -1 )

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            self.acc = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy
        # }}}

        # def concat_coor(o, i, d):
        #     coor = tf.tile(tf.expand_dims(
        #         [float(int(i / d)) / d, (i % d) / d], axis=0), [self.batch_size, 1])
        #     o = tf.concat([o, tf.to_float(coor)], axis=1)
        #     return o

        # def g_theta(o_i, o_j, q, scope='g_theta', reuse=True):
        #     with tf.variable_scope(scope, reuse=reuse) as scope:
        #         if not reuse: log.warn(scope.name)
        #         g_1 = fc(tf.concat([o_i, o_j, q], axis=1), 256, name='g_1')
        #         g_2 = fc(g_1, 256, name='g_2')
        #         g_3 = fc(g_2, 256, name='g_3')
        #         g_4 = fc(g_3, 256, name='g_4')
        #         return g_4

        # Classifier: takes images as input and outputs class label [B, m]
        def CONV(img, q, scope='CONV'):
            nD = 16 
            nC=6
             

            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                
                
                self.X0['question'] = q[:,:nC]
                
                x1 = tf.layers.conv2d( img, 32*2, 3, strides=(2,2) , activation='tanh', padding='valid')
                x2 = tf.layers.conv2d( x1, 32*2, 3, strides=(2,2), activation='tanh' , padding='valid' )
                x2 = tf.layers.conv2d( x2, 32*2, 3, strides=(2,2), activation='tanh' , padding='valid' )
                sz=x2.shape[1].value
                feat = tf.reshape(x2, [-1,sz*sz,32*2])

                 
                def feat2scores2(feat,nX,nF):
                    # pos = tf.constant(  np.eye(nF) , tf.float32 )
                    w = weight_variable( [ feat.shape[-1].value,nF]  )
                    b = bias_variable([nF,])
                    m = tf.matmul( feat,w+b)
                    # mm = tf.matmul( m , pos)
                    mm = tf.nn.softmax( m,1)
                    mmm = tf.transpose( mm, [0,2,1] )
                    return tf.layers.dense( mmm, nX, tf.nn.softmax )

                    

                
                
                with tf.variable_scope('conv_pos') as scope:
                    # Fs = feat2scores2(feat,nC)    
                    # rect = tf.layers.dense( Fs, 3, tf.nn.softmax )
                    rect = feat2scores2(feat,nC,3)    
                    rect = tf.transpose( rect, [0,2,1] )

                # with tf.variable_scope('obj'):
                
                #     # obj = tf.layers.dense( obj, 2, tf.nn.softmax )
                with tf.variable_scope('cl'):
                    pos = feat2scores2(feat,nC,nD)
                    # Fs = feat2scores2(feat,nC)
                    # pos = tf.layers.dense( feat, nD, tf.nn.softmax )
                    pos = tf.transpose( pos, [0,2,1] )


                # cl1,cl2 = get_conv(1,[nL1,nL2],24)
                
                # self.X0['obj'] = obj[:,0,:]
                
                self.X0['rectangle'] = rect[:,:,1]
                self.X0['exist'] = 1.0-rect[:,:,0]
                self.X0['pos'] = tf.reshape( pos, [-1, nC*nD])

                for i in range(self.q_dim):
                    self.X0['is_q_%d'%i] = q[:,i:(i+1)]

                  
                
                with tf.variable_scope('myscope'):
                    self.XO,L3 = self.mdl.getTSteps(self.X0)

                

                os = tf.concat( [self.XO['CL_%d'%i] for i in range(self.a_dim)],-1)
                return os
                
                
                # return fc(os, self.a_dim, activation_fn=None, name='fc_3')
                # # os=os+ fc(os, 1, activation_fn=tf.sigmoid, name='fc_2')*0
                # # return os
                # all_g = tf.concat( [self.XO[i.oname] for i in self.predColl.outpreds],-1)
                
                # all_g = fc( 2*all_g-1, 256, activation_fn=tf.nn.relu, name='fc_1')
                # # all_g = slim.dropout(all_g, keep_prob=0.5, is_training=is_train, scope='fc_3/')
                # all_g = fc(all_g, 256, activation_fn=tf.nn.relu, name='fc_2')
                # # all_g = slim.dropout(all_g, keep_prob=0.8, is_training=is_train, scope='fc_2')
                # return fc(all_g, self.a_dim, activation_fn=None, name='fc_3')
                # # all_g = tf.concat( [XO[i.oname] for i in self.predColl.outpreds if 'aux' in i.dname],-1)
                # os = tf.concat( [self.XO['CL_%d'%i] for i in range(self.a_dim)],-1)
                # # os=os+ fc(os, 1, activation_fn=tf.sigmoid, name='fc_2')*0
                # return os*5 
                                # return all_g
                # return all_g

        def f_phi(g, scope='f_phi'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                fc_1 = fc(g, 256, name='fc_1')
                fc_2 = fc(fc_1, 256, name='fc_2')
                # fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
                fc_3 = fc(fc_2, n, activation_fn=None, name='fc_3')
                return fc_3

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
