import gym
import numpy as np
import tensorflow as tf

from GridWorldEnv import GridWorld

from grid_ilp_def import *
import argparse
 
np.random.seed(33)                                



COLOR_COUNT=10
IMG_SIZE = 14
IMG_DEPTH=3

params = dotdict({})
params.ILP_VALUE=False
params.HARD_CHOICE=False
params.DBL_SOFTMAX=False
params.REMOVE_REP=False
params.RESET_RANDOM=False
params.MAX_EPISODES=200000
params.MAX_STEPS=50
params.EPS_TH=0
update_freq=10
params.MAX_MEM_SIZE=50 
params.NUM_BOX=4
params.BRANCH_COUNT = 0


env = GridWorld(max_episode = params.MAX_STEPS, max_branch_num=params.BRANCH_COUNT)

env.set_rewards(0.,1,10.,-.01)
env.max_length = 3

params.LR_ACTOR= .0005
params.LR_VALUE=.01
params.DISCOUNT_GAMMA=.2

print(params)


# non cnn version
def img_to_act(x,dim_out):
    

    state_in_x = tf.layers.dense( x/255, 30*COLOR_COUNT, tf.nn.relu, 'proj1')
    state_in_x = tf.layers.dense( state_in_x, COLOR_COUNT, tf.nn.softmax, 'proj2')
    
    has_key = state_in_x[:,0,0,:]
    state_in_x = tf.reshape( state_in_x[:,1:-1,1:-1,:], [-1,12*12,COLOR_COUNT])
    state_in_x = tf.reshape( state_in_x, [-1,12*12*COLOR_COUNT])
    
    return  has_key,state_in_x
 
 


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
    def __init__(self, act_dim, obs_dim, lr_actor, lr_value, gamma):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.gamma = gamma
        self.OBS = tf.placeholder(tf.float32, [None, IMG_SIZE,IMG_SIZE,IMG_DEPTH], name="observation")
        self.ACT = tf.placeholder(tf.int32, [None], name="action")
        self.Q_VAL = tf.placeholder(tf.float32, [None, 1], name="q_value")
        
        
        self.memory = Memory()
        
        
        # generate state representation
        with tf.variable_scope("STATE", reuse=False):
            self.state = img_to_act( self.OBS, (params.NUM_BOX+1)**2 )

        #run ilp to generate actions
        with tf.variable_scope("ILP", reuse=False):
            self.xo,_  = ilp_mdl.runtest( self.state)
        
        def pen(x):
            return x*(1.0-x)
        
        
            
        self.probs = (self.xo['move']-.1)*20
        self.act = tf.nn.softmax(self.probs)
        
        self.pen_loss = tf.reduce_mean(pen(self.state[0]))
        self.pen_loss += tf.reduce_mean(pen(self.state[1]))
        self.pen_loss += tf.reduce_mean(pen(self.act ))
        
        
        self.loss_cross = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.probs, labels=self.ACT)
        self.loss=tf.reduce_mean( self.loss_cross * self.Q_VAL)
        self.loss+= self.pen_loss*.1

        self.norms={}
        optimiser = tf.train.AdamOptimizer(self.lr_actor)
        
        self.gradients = optimiser.compute_gradients(loss=self.loss)
        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        for gradient, variable in self.gradients:
            if gradient is not None:
                self.norms[variable.name] = l2_norm(gradient)


        tvars2 = tf.trainable_variables()
        self.gradient_holders = []
        
        tvars=[]
        self.gradients=[]
        self.gradients2 = tf.gradients(self.loss,tvars2)
        for v,t in zip(self.gradients2,tvars2):
            if v is not None and t is not None:
                tvars.append(t)
                self.gradients.append(v)
        
        for idx,var in enumerate(tvars):
            
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.tvars=tvars
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_actor)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

        
        config=tf.ConfigProto( device_count = {'GPU': 1} )
        config.allow_soft_placement = False
        config.log_device_placement = False
        
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        
    def step(self, obs,episode):
        if obs.ndim < 4: obs = obs[np.newaxis, :]
        prob_weights,xo = self.sess.run( [self.act, self.xo], feed_dict={self.OBS: obs})
        
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        value=None
        return action, value

    def step_test(self, obs,episode):
        if obs.ndim < 4: obs = obs[np.newaxis, :]
        prob_weights,xo = self.sess.run( [self.act, self.xo], feed_dict={self.OBS: obs})
        
        action = np.argmax( prob_weights.ravel())
        
    
    def learn(self, last_value, done,reset_mem=True):

         

        obs, act, rwd = self.memory.covert_to_array()
        q_value = self.compute_q_value(last_value, done, rwd)*50
        grads,norms,loss = self.sess.run( [self.gradients,self.norms,self.loss] , {self.OBS: obs, self.ACT: act, self.Q_VAL: q_value})
        
        for idx,grad in enumerate(grads):
                gradBuffer[idx] += grad
                
        self.memory.reset()

    def compute_q_value(self, last_value, done, rwd):
        q_value = np.zeros_like(rwd)
        value = 0 if done else last_value
        for t in reversed(range(0, len(rwd))):
            value = value * self.gamma + rwd[t]
            q_value[t] = value
        return q_value[:, np.newaxis]


agent = ActorCritic(act_dim=144, obs_dim=[IMG_SIZE,IMG_SIZE,IMG_DEPTH],
                    lr_actor=params.LR_ACTOR, lr_value=params.LR_VALUE, gamma=params.DISCOUNT_GAMMA)


avgScore = MovingFn( np.mean,100 )
avgSuccess = MovingFn( np.mean,100 )

#%%
for v in tf.trainable_variables():
    print(v)

gradBuffer = agent.sess.run(agent.tvars)

for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
max_rate=0
for i_episode in range(params.MAX_EPISODES):
    
    
    obs0 = env.reset()#[1:-1,1:-1,:]
    obs0=env.toImage(1)
    ilp_mdl.reset()
    
    r1=np.copy(env.toImage(1)[:,:,0])
    ep_rwd = 0

    cnt=0
    while True:
        
        act, _ = agent.step(obs0,i_episode)
        obs1, rwd, done, info = env.step(act)
        obs1=env.toImage(1)
        
        agent.memory.store_transition(obs0, act, rwd)
        ep_rwd += rwd

        obs0 = obs1
        
        if done or cnt>params.MAX_STEPS:
            
            agent.learn(0, done,not done )
            
            # apply gradient updates 
            if i_episode%update_freq==0:
                feed_dict= dictionary = dict(zip(agent.gradient_holders, gradBuffer))
                _ = agent.sess.run(agent.update_batch, feed_dict=feed_dict)
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                    
            break
        
        cnt+=1
        
    
    #print('Ep: %i' % i_episode, "|Ep_r: %.2f, |average Ep_r: %.2f,%.2f ,difficulty:%d, max_rate:%.2f " %  (ep_rwd,avgScore.add(cnt),avgSuccess.add(done and ep_rwd>=.1),env.max_length , max_rate) )
    print( 'episode : ',i_episode,'    average episode steps : ', np.round(avgScore.add(cnt) ), '   average success rate : ', avgSuccess.add(done and ep_rwd>=.1)) 
    if avgSuccess.get()>max_rate:
        max_rate = avgSuccess.get()
    
