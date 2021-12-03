import argparse, configparser
import os, sys, argparse, collections, pickle, json
import numpy as np
import gym
from util import *
from weightmax import Network

L_SOFTPLUS = 0
L_RELU = 1
L_LINEAR = 2
L_SIGMOID = 3
L_DISCRETE = 4
L_BINARY_Z = 5
L_BINARY_N = 6
LS_REAL = [L_SOFTPLUS, L_RELU, L_LINEAR, L_SIGMOID]
LS_DISCRETE = [L_DISCRETE, L_BINARY_Z, L_BINARY_N,]

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=False, default="config_cp.ini",
   help="location of config file")
args = ap.parse_args()
f_name = os.path.join("config", "%s" % args.config) 
print("Loading config from %s" % f_name)

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read(f_name)

name = config.get("USER", "name") # Name of the run
max_eps = config.getint("USER", "max_eps") # Number of episode per run
n_run = config.getint("USER", "n_run") # Number of runs

batch_size = config.getint("USER", "batch_size") # Batch size
env_name = config.get("USER", "env_name") # Environment name
gamma = config.getfloat("USER", "gamma") # Discount rate

critic_hidden = json.loads(config.get("USER","critic_hidden")) # Number of hidden layers unit in critic network 
actor_hidden = json.loads(config.get("USER","actor_hidden"))  # Number of hidden layers unit in actor network 
temp = json.loads(config.get("USER","temp")) # Temperature on each layer

critic_lambda = config.getfloat("USER", "critic_lambda") # Lambda for critic
actor_lambda = config.getfloat("USER", "actor_lambda") # Lambda for actor
reward_lim = config.getfloat("USER", "reward_lim") # Whether to cap the maximum absolute value of reward; negative for no capping

w_reg = json.loads(config.get("USER","w_reg")) # Strength for weight regularization for each layer
w_reg_p = config.getfloat("USER", "w_reg_p") # Norm used in weight regularization
weight_max = config.getfloat("USER", "weight_max") # Norm used in Weight Max (default l2 norm)

critic_lr_st = json.loads(config.get("USER", "critic_lr_st")) # Learning rate for critic at the start (for each layer)
critic_lr_end = json.loads(config.get("USER", "critic_lr_end")) # Learning rate for critic at the end (for each layer)
actor_lr_st = json.loads(config.get("USER", "actor_lr_st")) # Learning rate for actor at the start (for each layer)
actor_lr_end = json.loads(config.get("USER", "actor_lr_end"))# Learning rate for actor at the end (for each layer)
end_t = config.getint("USER", "end_t") # Number of steps to reach the final learning rate

critic_lambda_ = critic_lambda * gamma
actor_lambda_ = actor_lambda * gamma

critic_l_type = L_SOFTPLUS
actor_l_type = L_BINARY_Z
actor_output_l_type = L_DISCRETE
reward_lim = None if reward_lim <=0 else reward_lim

h_reg, entro_reg, critic_var, actor_var = 0, 0, 0, 0
actor_bp, baseline, unbiased, det, zero_learn = False, False, False, False, False

# optimizer
critic_opt = "adam" # optimizer adam or simple
actor_opt = "adam"
beta_1, beta_2 = 0.9, 0.999

print_every = 1000     
eps_ret_hist_full = []
print("Starting experiments..")

if env_name != "multiplexer":

    env = batch_envs(name=env_name, batch_size=batch_size, rest_n=0, warm_n=0)   
    dis_act = type(env.action_space) != gym.spaces.box.Box
    action_n = env.action_space.n if dis_act else env.action_space.shape[0]    
    if actor_output_l_type == L_BINARY_Z: action_n = 1 

    for j in range(n_run):          
        critic_net = Network(state_n=env.state.shape[1], action_n=1, hidden=critic_hidden, var=critic_var, 
                                                 temp=None, hidden_l_type=critic_l_type, output_l_type=L_LINEAR, opt=critic_opt, beta_1=beta_1, beta_2=beta_2,)     

        actor_net = Network(state_n=env.state.shape[1], action_n=action_n, hidden=actor_hidden, var=actor_var, 
                                             temp=temp, hidden_l_type=actor_l_type, output_l_type=actor_output_l_type, opt=actor_opt, beta_1=beta_1, beta_2=beta_2,
                                             unbiased=unbiased)


        eps_ret_hist = []
        c_eps_ret = np.zeros(batch_size)
        
        print_count = print_every         
        value_old = None
        isEnd = env.isEnd
        prev_isEnd = env.isEnd
        truncated, solved, f_perfect = False, False, False
        
        state = env.reset()
        for i in range(int(1e9)):     
            action = actor_net.forward(state, det=det)    
            value_new = critic_net.forward(state)[:,0]
                
            if value_old is not None:      
                if reward_lim is not None: reward = np.clip(reward, -reward_lim, +reward_lim)
                targ_value = reward + gamma * value_new * (~isEnd).astype(float)
                critic_reward = targ_value - value_old
                critic_reward[prev_isEnd | info["truncatedEnd"]] = 0        
                actor_reward = targ_value - value_old
                actor_reward[prev_isEnd | info["truncatedEnd"]] = 0            
                cur_critic_lr = linear_interpolat(start=critic_lr_st, end=critic_lr_end, end_t=end_t, cur_t=i)
                cur_actor_lr = linear_interpolat(start=actor_lr_st, end=actor_lr_end, end_t=end_t, cur_t=i)  
                critic_net.learn(critic_reward, lr=cur_critic_lr)
                actor_net.learn(actor_reward, lr=cur_actor_lr, w_reg=w_reg, w_reg_p=w_reg_p, entro_reg=entro_reg, h_reg=h_reg, 
                                                weight_max=weight_max, baseline=baseline, det=det)  
         
            critic_net.clear_trace(~prev_isEnd)
            critic_net.backprop(targ=None, grad_mean=True, lambda_=critic_lambda_)

            actor_net.clear_trace(~prev_isEnd)  
            if actor_bp:            
                actor_net.backprop(targ=None, grad_mean=False, lambda_=actor_lambda_)
            else:
                for n, a in enumerate(actor_net.layers): a.record_trace(lambda_=actor_lambda_, det=getl(det, n), zero_learn=getl(zero_learn, n))          

            value_old = np.copy(value_new)
            prev_isEnd = np.copy(isEnd)    
            if actor_output_l_type == L_BINARY_Z: 
                action = action[:,0] 
            elif dis_act:
                action = from_one_hot(action)
                
            state, reward, isEnd, info = env.step(action)        
            if not truncated and np.any(info["truncatedEnd"]): 
                print ("Warning: Limit reached (Eps %d)" % len(eps_ret_hist))              
                truncated = True
                
            c_eps_ret += reward
            
            if np.any(isEnd):
                eps_ret_hist.extend(c_eps_ret[isEnd].tolist())
                c_eps_ret[isEnd] = 0.

            if len(eps_ret_hist) >= max_eps: break 
            if i*batch_size > print_count and len(eps_ret_hist) > 0:      
                f_str = "%d: Step %d Eps %d Moving Average Return %.2f Maximum Return %.2f"
                f_arg = [j, i, len(eps_ret_hist), np.average(eps_ret_hist[-100:]), np.amax(eps_ret_hist)]
                print(f_str % tuple(f_arg))         
                print_count += print_every    
        eps_ret_hist_full.append(eps_ret_hist)
else:
    env = complex_multiplexer_MDP(addr_size=4, action_size=1, zero=False, reward_zero=False)  
    print_every = 32*500  
    for j in range(n_run): 
        net = Network(state_n=env.x_size, action_n=1,  hidden=actor_hidden, var=actor_var, temp=temp, 
                                 hidden_l_type=actor_l_type, output_l_type=L_BINARY_Z, opt=actor_opt, beta_1=beta_1, beta_2=beta_2,
                                 unbiased=unbiased
                                 )  
        eps_ret_hist = []
        print_count = print_every         
        for i in range(max_eps//batch_size):  
            state = env.reset(batch_size)        
            action = net.forward(state, det)[..., np.newaxis][:,:,0]
            action = zero_to_neg(action)
            reward = env.act(action)[:,0]
            eps_ret_hist.append(np.average(reward))   
            if actor_bp:
                net.backprop(targ=None, grad_mean=False, w_reg=w_reg, w_reg_p=w_reg_p, entro_reg=entro_reg)
            else:
                for n, a in enumerate(net.layers): a.record_trace(lambda_=0, det=getl(det, n), zero_learn=getl(zero_learn, n))                
            net.learn(reward, lr=actor_lr_st, w_reg=w_reg, w_reg_p=w_reg_p, h_reg=h_reg, 
                                entro_reg=entro_reg,weight_max=weight_max,baseline=baseline, 
                                det=det)

            if i*batch_size > print_count:
                f_str = "%d: Step %d Eps %d Moving Average Return %.2f Maximum Return %.2f"
                f_arg = [j, i, len(eps_ret_hist), np.average(eps_ret_hist[-100:]), np.amax(eps_ret_hist)]
                print(f_str % tuple(f_arg))         
                print_count += print_every     
        eps_ret_hist_full.append(eps_ret_hist)

eps_ret_hist_full = np.array(eps_ret_hist_full)
print("Finished Training.")
curves = {}  
curves[name] = eps_ret_hist_full
names = {k:k for k in curves.keys()}
f_name = os.path.join("result", "%s.npy" % name) 
print("Results (saved to %s):" % f_name)
np.save(f_name, curves)
print_stat(curves, names)
plot(curves, names, mv_n=10 if env_name != "multiplexer" else 10000, end_n=max_eps, legend=True)
plt.show()

