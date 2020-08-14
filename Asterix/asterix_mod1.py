import gym
import numpy as np
import tensorflow as tf

import numpy as np

 
from ip_def  import *
from mylib import *

import argparse
from utils2 import *
 


params = dotdict({})

ACTIONS = ['action_noop','action_up','action_right','action_left', 'action_down']
N_ACTION=5
F_COUNT=4

DIM1=8
DIM2=12

IMG_X=128
IMG_Y=144
 


params.LIFE_PENALTY= -10
params.IDLE_REWARD = 0
params.EAT_REWARD = 10.



params.MAX_EPISODES=200000

params.MAX_MEM_SIZE=600
params.MAX_LEN = 50
params.MEM_MIN_LENGTH=400
params.LR_ACTOR=.01
params.DISCOUNT_GAMMA=.2
params.NORMALIZE_Q = False
 
env = gym.make("AsterixNoFrameskip-v4")
params.LOSS_METHOD=2
params.NEG_EXAMPLES_EXPERIENCE=10
params.SOFTMAX_COEFF= lambda x: (x-.01)*6
ilp_mdl = ILP_MODEL(DIM1,DIM2,3)
ilp_mdl.mdl.print_vars()

memory = Memory(params.DISCOUNT_GAMMA, params.LIFE_PENALTY,max_neg_experience=200)

 

class ActorCritic:
    def __init__(self):
        
        self.ACT = tf.placeholder(tf.int32, [None], name="action")
        self.Q_VAL = tf.placeholder(tf.float32, [None, 1], name="q_value")
        self.OBS = tf.placeholder(tf.float32, [None, DIM1*DIM2,5], name="st1")
        
        self.state = tf.unstack(self.OBS,axis=-1)[1:]
        p,self.x0,self.xo = ilp_mdl.run( self.state)
        
        
        self.logits =  params.SOFTMAX_COEFF(p) 
        self.act = tf.nn.softmax(self.logits, name='act_prob') 
        
        if params.LOSS_METHOD==1:
            cross_entropy  = -tf.log(1.e-5+tf.gather(self.act,self.ACT))
        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.ACT)
        
        self.actor_loss = tf.reduce_mean(cross_entropy * self.Q_VAL) 
        opt=tf.train.AdamOptimizer(params.LR_ACTOR)
        self.actor_train_op = opt.minimize(self.actor_loss )
        

        grs= opt.compute_gradients(self.actor_loss  ) 
        gg=  [ g for (g,v) in grs if g is not None]
        self.grn = tf.linalg.global_norm(gg)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def step(self, obs,episode):
        
        prob_weights  = self.sess.run( self.act , feed_dict={self.OBS: obs})
        s = prob_weights.ravel()
        
        action = np.random.choice(range(prob_weights.shape[1]), p=s)
    
        if s[action]<.01:
            print('rare events:  ',s,action)
            return self.step(obs,episode)
        return action,s


    def learn(self  ):

        obs, act, rwd, q_value = memory.get_buffer(params.MAX_LEN, normalizeQ = params.NORMALIZE_Q)
        
        if params.NEG_EXAMPLES_EXPERIENCE>0:
            obs1, act1, q_value1 = memory.get_neg_exp_sample(params.NEG_EXAMPLES_EXPERIENCE)
            if obs1 is not None:
                obs = np.concatenate( (obs,obs1), 0 )
                act = np.concatenate( (act,act1), 0 )
                q_value = np.concatenate( (q_value,q_value1), 0 )
                
                
        _,loss ,ln = self.sess.run( [self.actor_train_op,self.actor_loss, self.grn], {self.OBS: obs, self.ACT: act, self.Q_VAL: q_value })
        print('losses : ',loss,ln)


agent = ActorCritic()

avgScore = MovingFn( np.mean,100 )
avgSuccess = MovingFn( np.mean,100 )
for v in tf.trainable_variables():
    print(v)


#%%


obs0 = env.reset( ) 
img_bk=obs0.copy()
obs0,_,_,_=env.step( 0 )

max_cnt = 0
max_reward = 0   
 

    
    
        
for i_episode in range(params.MAX_EPISODES):
    
    obs0 = env.reset( ) 
    obs0,rwd, done2, info = env.step(0)
    
    
    ep_rwd = 0
    cnt_total=0
    lives=3
    
    
    while True:
    
        
        if i_episode%1==0:
            env.render()
        
        
        
        img =  obs0[24:152,8:152,:]#- img_bk[24:152,8:152,:]
        obs=get_img_all(img)
        act,prob = agent.step(obs[np.newaxis] ,i_episode)
        if cnt_total%10==0:
            print(cnt_total,act,prob)
         
        rwds=0 
        
        cnt_4=0
        rwds=0
        if act in [1,4]:
            while cnt_total%4!=3:
                obs1, rwd, done, info = env.step( act )
                rwds+=rwd
                cnt_total+=1
                cnt_4+=1
                
                
        # if cnt_total%4!=0 and :
        #     act=0
            
        obs1, rwd, done, info = env.step( act )
        R = rwd+rwds
        ep_rwd+=R
        if R==0:
            R = params.IDLE_REWARD
        elif R>0:
            R = params.EAT_REWARD
         
       
        if info['ale.lives']<lives:
            lives=info['ale.lives']
            R=params.LIFE_PENALTY
            print('lives:',lives)
             
            
        # for _ in  range(cnt_4):
        memory.store_transition( obs, act, R,cnt_4)
        obs0 = obs1
        
        
        if info['ale.lives']==0:
             
            while memory.get_len()>0:
                agent.learn()
            
            break
        
        
        if memory.get_len()> params.MEM_MIN_LENGTH:
            agent.learn()
            
      
        cnt_total+=1
        
    max_cnt = max( max_cnt, cnt_total)
    max_reward = max( max_reward, ep_rwd)
        
    
    print('Ep: %i' % i_episode, "|Ep_r: %.2f, cnt = %d, , |max : %.2f,%d" %  (ep_rwd,cnt_total, max_reward,max_cnt) )
    cnt_total=0
    