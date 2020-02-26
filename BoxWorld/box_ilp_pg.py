#%%
import gym
import numpy as np
import tensorflow as tf

from BoxWorldEnv import *
from box_ilp_def import *
import argparse
 

params = dotdict({})
params.ILP_VALUE=False
params.HARD_CHOICE=False
params.DBL_SOFTMAX=False
params.REMOVE_REP=False
params.RESET_RANDOM=True
params.MAX_EPISODES=58000
params.MAX_STEPS=20
params.EPS_TH=0
params.MAX_MEM_SIZE=20
params.NUM_BOX=4
params.LR_ACTOR=.002
params.LR_VALUE=.002
params.DISCOUNT_GAMMA=.7
params.REWARD=6
params.PENALTY=-.02
params.ATTEN=1
IMG_SIZE = 64

env = BoxWorldEnvImage(params.NUM_BOX,max_episode = params.MAX_STEPS, goal_type='stack buttom blue', reward=params.REWARD,penalty=params.PENALTY,error_penalty=params.PENALTY)


def feat2scores(feat):

     
    w = weight_variable( [ feat.shape[-1].value,params.NUM_BOX]  )
    b = bias_variable([params.NUM_BOX,])

    m = tf.matmul( feat,w+b)
    mmm = tf.nn.softmax( m,1)
    return tf.transpose( mmm, [0,2,1] )



def conv2d(input, output_shape, is_train, activation_fn=tf.nn.relu,
           k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, name="conv2d",bne=True):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, s_h, s_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_shape],
                                 initializer=tf.constant_initializer(0.0))
        activation = activation_fn(conv + biases)
        if bne:
            return tf.contrib.layers.batch_norm(activation, center=True, scale=True,
                                          decay=0.9, is_training=is_train,
                                          updates_collections=None)
        else:
            return activation
    return bn

print(params)
 
def img_to_act(x,dim_out):
    
    L=32
    x1 = tf.layers.conv2d( x, 32, 3, strides=(2,2) , activation='tanh', padding='valid')
    x2 = tf.layers.conv2d( x1, L, 3, strides=(2,2), activation='tanh' , padding='valid' )
    
    sz=x2.shape[1].value
   

    feat = tf.reshape(x2, [-1,sz*sz,L])
    feat = custom_grad(tf.cast( tf.greater_equal(feat, .3),tf.float32), feat)
    
    with tf.variable_scope('posx'):
        mmm = feat2scores(feat)
        state_in_x = tf.layers.dense( mmm, params.NUM_BOX, tf.nn.softmax )
    with tf.variable_scope('posy'):
        mmm = feat2scores(feat)
        state_in_y = tf.layers.dense( mmm, params.NUM_BOX, tf.nn.softmax )
    
    
    def binarize(x):
        return x
        #return custom_grad( tf.cast(tf.greater(x,.4), tf.float32) , x)
    
    state_in_x  = binarize(state_in_x )
    state_in_y  = binarize(state_in_y )
    return (state_in_x,state_in_y)
  
ilp_mdl = ILP_MODEL(num_box=params.NUM_BOX)
ilp_mdl.mdl.print_vars()


class Memory(object):
    def __init__(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []

    def store_transition(self, obs0, act, rwd):
        self.ep_obs.append(obs0)
        self.ep_act.append(act)
        self.ep_rwd.append(float(rwd))

    def covert_to_array(self):
        array_obs = np.stack(self.ep_obs,0)
        array_act = np.array(self.ep_act)
        array_rwd = np.array(self.ep_rwd)
        return array_obs, array_act, array_rwd

    def reset(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []

    def limit_size(self):
        if len(self.ep_act)>params.MAX_MEM_SIZE:
            self.ep_act = self.ep_act[-params.MAX_MEM_SIZE:]
            self.ep_obs = self.ep_obs[-params.MAX_MEM_SIZE:]
            self.ep_rwd = self.ep_rwd[-params.MAX_MEM_SIZE:]
            
    def cleanup(self):
        for i in range( len(self.ep_obs)-2):
            for j in range(i+1,len(self.ep_obs)-1):
                if np.all(self.ep_obs[i]==self.ep_obs[j]):
                    self.ep_obs=self.ep_obs[:i]+self.ep_obs[j:]
                    self.ep_act=self.ep_act[:i]+self.ep_act[j:]
                    self.ep_rwd=self.ep_rwd[:i]+self.ep_rwd[j:]
                    i=j
                    break



class ActorCritic:
    def __init__(self, lr_actor, lr_value, gamma):
        
        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.gamma = gamma
        self.OBS = tf.placeholder(tf.float32, [None, IMG_SIZE,IMG_SIZE,3], name="observation")
        self.ACT = tf.placeholder(tf.int32, [None], name="action")
        self.Q_VAL = tf.placeholder(tf.float32, [None, 1], name="q_value")

        self.memory = Memory()

        # generate state representation
        with tf.variable_scope( "state", reuse=False):
            self.state = img_to_act( self.OBS, (params.NUM_BOX+1)**2 )  
        
        #run ILP to generate actions
        with tf.variable_scope( "ILP", reuse=False):
            self.xo =  ilp_mdl.run( self.state[0],self.state[1]) 
        
        self.probs= self.xo['move']
        self.act =  tf.nn.softmax(self.probs*12)
        
        cross_entropy =  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.probs*12, labels=self.ACT)
        
        
        
        # create penalties used in the intyerpretability
        loss_w = 0 
        loss_sum = 0 
        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ILP')
        for v in vs:
            w=tf.sigmoid(2*v)
            loss_w += tf.reduce_mean( w*(1-.0-w))
            loss_sum += tf.reduce_mean( tf.nn.relu(tf.reduce_sum(w,-1)-6)  ) 
            
        
        self.loss_sum = loss_sum
        self.loss_w=loss_w
        self.loss_s = tf.reduce_mean( tf.reduce_mean( self.state[0]*(1.0-self.state[0]),-1) + tf.reduce_mean( self.state[1]*(1.0-self.state[1]) ,-1) )
        
        
        
        self.advantage = self.Q_VAL 
        self.actor_loss = tf.reduce_mean(cross_entropy * self.advantage )
        self.actor_train_op = tf.train.RMSPropOptimizer(self.lr_actor).minimize(self.actor_loss)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs,episode):
        if obs.ndim < 4: obs = obs[np.newaxis, :]
        prob_weights = self.sess.run(self.act, feed_dict={self.OBS: obs})
        

        p = prob_weights.ravel()
        #p=softmax(10*(p-.1))
        action = np.random.choice(range(prob_weights.shape[1]), p=p)
        value = 0 
        return action, value

    def learn(self, last_value, done,reset_mem=True):
        

        obs, act, rwd = self.memory.covert_to_array()
        q_value = self.compute_q_value(last_value, done, rwd)
        q_value[q_value>0]=1.0
        #if np.sum(q_value >0)>0:
        #    print(q_value)
        self.sess.run(self.actor_train_op, {self.OBS: obs, self.ACT: act, self.Q_VAL: q_value  })
 
        self.memory.reset()
       
    def compute_q_value(self, last_value, done, rwd):
        q_value = np.zeros_like(rwd)
        value = 0 if done else last_value
        for t in reversed(range(0, len(rwd))):
            value = value * self.gamma + rwd[t]
            q_value[t] = value
        #q_value -= np.mean(q_value)
        #q_value /= (1.e-5+np.std(q_value))
        return q_value[:, np.newaxis]

 

agent = ActorCritic( lr_actor=params.LR_ACTOR, lr_value=params.LR_VALUE, gamma=params.DISCOUNT_GAMMA)


avgScore = MovingFn( np.mean,100 )
avgSuccess = MovingFn( np.mean,100 )

#%%
for v in tf.trainable_variables():
    print(v)

 
for i_episode in range(params.MAX_EPISODES):
    
    obs0 = env.reset(random=params.RESET_RANDOM) 
    ep_rwd = 0

    cnt=0
    while True:
        act, _ = agent.step(obs0,i_episode)
        obs1, rwd, done, info = env.step(act)
        
        agent.memory.store_transition(obs0, act, rwd)
        ep_rwd += rwd

        obs0 = obs1
        
        if done or cnt>params.MAX_STEPS:
           
            
            if done:
                
                agent.learn(0, done,not done )
            
            else:
                agent.learn(0, done,not done )
            

            break
            
        cnt+=1
        
    print('Ep: %i' % i_episode, "|Episode reward : %.2f, |average N: %.2f, average success %.2f " %  (ep_rwd,avgScore.add(cnt),avgSuccess.add(done and ep_rwd>1 )) )

#%%
