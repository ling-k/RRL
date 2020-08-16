import gym
import numpy as np
import tensorflow as tf
import logging
import numpy as np
import time
from ilp_def import *
from mylib import *
import argparse
from utils import *

logging.basicConfig(filename='myapp.log', level=logging.INFO)
calibs=pickle.load( open('calib20','rb'))

calib_state = np.stack( [get_img_all(c) for c in calibs],0).astype(np.float32)
calib_obs = np.stack(calibs).astype(np.float32)

calib_state_tf=tf.constant(calib_state,dtype=tf.float32)
calib_obs_tf=tf.constant(calib_obs,dtype=tf.float32)


calib_state_tf=tf.convert_to_tensor(calib_state,dtype=tf.float32)
calib_obs_tf=tf.convert_to_tensor(calib_obs,dtype=tf.float32)


params = dotdict({})
ACTIONS = ['action_noop','action_up','action_right','action_left', 'action_down']
DIM1=8
DIM2=12
IMG_X=128
IMG_Y=144

params.LIFE_PENALTY= 0
params.IDLE_REWARD = 0
params.EAT_REWARD = 1.

params.MAX_EPISODES=200000
params.MAX_LEN = 50
 

params.LR_ACTOR=.0001
params.DISCOUNT_GAMMA=.2
params.NORMALIZE_Q = False
params.LOSS_METHOD=2
params.SOFTMAX_COEFF= lambda x: (x-.03)*12

env = gym.make("AsterixNoFrameskip-v4")
params.NEG_EXAMPLES_EXPERIENCE=0
ilp_mdl = ILP_MODEL(DIM1,DIM2,3)
ilp_mdl.mdl.print_vars()
memory = Memory(params.DISCOUNT_GAMMA)

 


    
def img_to_act (inp ):
     
    
    is_train=True
    act=tf.nn.relu
    
    x = conv2d(inp, 33, is_train, s_h=16, s_w=1, k_h=12, k_w=8, name='conv_1',bne=False,activation_fn=act)
    mask = custom_grad(tf.cast( tf.greater_equal(x[:,:,:,:1],0),tf.float32), tf.sigmoid(x[:,:,:,:1]))
    x = conv2d(x[:,:,:,1:], 256, is_train, s_h=1, s_w=1, k_h=1, k_w=8, name='conv_2',bne=False,activation_fn=act)
    x=tf.keras.layers.MaxPooling2D( (1,12))(x*mask)
    
    x = tf.layers.dense( x, 128,act, name='fc0')
    feat = tf.layers.dense( x, 5, name='fc1')
    feat = tf.reshape(feat, [-1,feat.shape[1]*feat.shape[2],feat.shape[3]])

    return  tf.unstack( tf.nn.softmax(feat),axis=-1),feat
 



class Actor:
    def __init__(self):
        
        self.ACT = tf.placeholder(tf.int32, [None], name="action")
        self.Q_VAL = tf.placeholder(tf.float32, [None, 1], name="q_value")
        self.OBS = tf.placeholder(tf.float32, [None, IMG_X,IMG_Y, 3], name="st1")
        
        with tf.variable_scope('IMG', reuse=False):
            self.state,_= img_to_act( self.OBS/255.)
        
        with tf.variable_scope('IMG', reuse=True):
            _,self.state_calib = img_to_act( calib_obs_tf/255.)
        
         
        self.calib_loss = tf.nn.softmax_cross_entropy_with_logits_v2 ( labels=calib_state_tf, logits=self.state_calib) 
        self.calib_loss = tf.reduce_mean( self.calib_loss )
        
        p,self.x0,self.xo,self.xor = ilp_mdl.run( self.state[1:])
        self.logits =  params.SOFTMAX_COEFF(p) 
        self.act = tf.nn.softmax(self.logits, name='act_prob') 
        
        
        if params.LOSS_METHOD==1:
            cross_entropy  = -tf.log(1.e-5+tf.gather(self.act,self.ACT))
        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.ACT)
        
        
        #state constraints
        self.losses=[]
        self.losses.append( tf.reduce_mean( neg_ent_loss( self.x0['C1'], self.xo['C1']) ) )
        self.losses.append( tf.reduce_mean( neg_ent_loss( self.x0['C2'], self.xo['C2']) ) )
        self.losses.append( tf.reduce_mean( neg_ent_loss( self.x0['C3'], self.xo['C3']) ) )
        self.losses.append( tf.reduce_mean( neg_ent_loss( tf.ones_like(self.x0['C4']), self.xo['C4']) ) )
        
        #actor loss
        self.actor_loss = tf.reduce_mean(cross_entropy * self.Q_VAL)  +  tf.add_n(self.losses) + 10.1* self.calib_loss
        opt=tf.train.AdamOptimizer(params.LR_ACTOR)
        self.actor_train_op = opt.minimize(self.actor_loss )
        

        self.sess = tf.Session(config=tf.ConfigProto(use_per_session_threads=True, ))
        self.sess.run(tf.global_variables_initializer())

    
    def getxo(self,obs, decimals=3):
        xo = self.sess.run( self.xor, feed_dict={self.OBS: obs[np.newaxis]})
        for p in xo:
            xo[p] = np.round(xo[p][0], decimals=decimals)
        return xo
    
    def step(self, obs,episode):
        
        prob_weights  = self.sess.run( self.act , feed_dict={self.OBS: obs})
        s = prob_weights.ravel()
        
        action = np.random.choice(range(prob_weights.shape[1]), p=s)
    
        if s[action]<.02:
            # print('rare events:  ',s,action)
            return self.step(obs,episode)
        return action,s


    def learn(self  ):
        
        obs, act, rwd, q_value = memory.get_buffer(normalizeQ = params.NORMALIZE_Q)
        if len(act)>1: 
            logging.info('before')
            _,loss ,ls,lc = self.sess.run( [self.actor_train_op,self.actor_loss, self.losses,self.calib_loss], {self.OBS: obs, self.ACT: act, self.Q_VAL: q_value })
            logging.info('after')
            # print('losses : ',loss,ls,'calib', lc)



agent = Actor()
for v in tf.trainable_variables():
    print(v)


#%%
 
def testrun(use_max=True):
    obs0 = env.reset(  ) 
    img_bk=obs0.copy()
    obs0,_,_,_=env.step( 0 )
    ep=0
    cnt=0
    while(True):
        img =  obs0[24:152,8:152,:] - img_bk[24:152,8:152,:]
        act,prob = agent.step(img[np.newaxis] ,i_episode)    
        # print(ACTIONS[act],prob)
        if use_max:
            act=np.argmax(prob)
            
            
        if cnt%4==0:
            env.render()
        if act in [1,4]:
            while cnt%4!=3:
                obs1, rwd, done, info = env.step( act )
                ep+=rwd
                cnt+=1
                if cnt%4==0:
                    env.render()    
        
        obs0, rwd, done, info = env.step( act )
        cnt+=1
        ep+= rwd
        if done:
            print( 'test run finished, count ', cnt, '  reward ', ep)
            break
        
        

obs0 = env.reset( ) 
img_bk=obs0.copy()
obs0,_,_,_=env.step( 0 )

max_cnt = 0
max_reward = 0   
 
def get_step(action):
    obs0,rwd, done, info = env.step(action)
    img =  obs0[24:152,8:152,:] - img_bk[24:152,8:152,:]
    return img,rwd,done,info
    
    
          
for i_episode in range(params.MAX_EPISODES):
    
    
    
    if i_episode%10==0:
        print('***********************************************************')
        print('\n\n running test mode: deterministic mode')
        testrun(True)
        print('***********************************************************')
        print('\n\n running test mode: random mode')
        testrun(False)
        print('***********************************************************')
        
        
    obs0 = env.reset( ) 
    img,rwd, done, info  = get_step(0)
    
    
    ep_rwd = 0
    cnt_total=0
    lives=3
    
    
    while True:
    
        
        
        if cnt_total%10==0:
            env.render()
        act,prob = agent.step(img[np.newaxis] ,i_episode)
        
        
        rwds=0 
        cnt_4=0
        rwds=0
        
        if act in [1,4]:
            while cnt_total%4!=3:
                img,rwd, done, info  = get_step(act)
                rwds+=rwd
                cnt_total+=1
                cnt_4+=1
                
        
        img1,rwd, done, info  = get_step(act)
        cnt_total+=1
        
        R = rwd+rwds
        ep_rwd+=R
        if R==0:
            R = params.IDLE_REWARD
        elif R>0:
            R = params.EAT_REWARD
         
       
        if info['ale.lives']<lives:
            lives=info['ale.lives']
            R=params.LIFE_PENALTY
            print ('lives:',lives, 'score:', ep_rwd)
             
            
        
        memory.store_transition( img, act, R)
        img=img1.copy()
        
        
        if info['ale.lives']==0:
            agent.learn()
            break
        
        
        if memory.get_len()> params.MAX_LEN:
            agent.learn()
            
        
        
    max_cnt = max( max_cnt, cnt_total)
    max_reward = max( max_reward, ep_rwd)
        
    
    print('Ep: %i' % i_episode, "|Ep_r: %.2f, cnt = %d, , |max : %.2f,%d" %  (ep_rwd,cnt_total, max_reward,max_cnt) )
    cnt_total=0
    