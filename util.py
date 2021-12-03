from abc import ABC, abstractmethod
import numpy as np 
import gym
import matplotlib
import matplotlib.pyplot as plt

def relu(x):
  y = np.copy(x)
  y[y<0] = 0
  return y

def relu_d(x):  
  y = np.ones_like(x)
  y[x<0] = 0
  return y

def sigmoid(x):
  lim = 20
  l = np.zeros_like(x)
  l[np.abs(x) < lim] = 1/(1+np.exp(-x[np.abs(x) < lim]))
  l[x <= -lim] = 0
  l[x >= lim] = 1
  return l

def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1-s)

def sigmoid_i(x):
    return x/sigmoid(X)

def softplus(x):
  r = np.zeros_like(x)
  r[x>30] = x[x>30]
  r[np.abs(x)<=30] = np.log1p(np.exp(x[np.abs(x)<=30]))
  return r
  #return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)
  #return np.log(1+np.exp(x))

def softplus_i(x):
  r = np.zeros_like(x)
  r[x>30] = x[x>30]
  r[x<0.01] = np.log(-1+np.exp(0.01))
  b_flag = (x>=0.01) & (x<=30)
  r[b_flag] = np.log(-1+np.exp(x[b_flag]))
  return r

def softmax(X, theta=1.0, axis=None):
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y[y<-30] = -30
    y = np.exp(y)    
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1: p = p.flatten()
    return p

def softmax_i(x):
    return x/softmax(x)

def recip_z(x):
    y = np.zeros_like(x)
    y[x > 1e-8] = 1/x[x > 1e-8]
    y[x < 1e-8] = 1/x[x < 1e-8]
    y = np.clip(y, -10, 10)
    return y     

def multinomial_rvs(n, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out

def from_one_hot(y):      
  return np.argmax(y, axis=-1)

def to_one_hot(a, size):
  oh = np.zeros((a.shape[0], size), np.int)
  oh[np.arange(a.shape[0]), a.astype(int)] = 1
  return oh

def getl(x, n):
  return x[n] if type(x) == list else x        
        
def equal_zero(x):
  return np.logical_and(x > -1e-8, x < 1e-8).astype(np.float32)

def mask_neg(x):
  return (x < 0).astype(np.float32) 
               
def sign(x):
  return (x > 1e-8).astype(np.float32) - (x < -1e-8).astype(np.float32)

def zero_to_neg(x):
  return (x > 1e-8).astype(np.float32) - (x <= 1e-8).astype(np.float32)

def neg_to_zero(x):
  return (x > 1e-8).astype(np.float32)  

def linear_interpolat(start, end, end_t, cur_t):  
  if type(start) == list:
    if type(end_t) == list:
      return [(e - s) * min(cur_t, d) / d + s for (s, e, d) in zip(start, end, end_t)]
    else:    
      return [(e - s) * min(cur_t, end_t)  / end_t + s for (s, e) in zip(start, end)]
  else:
    if type(end_t) == list:
      return [(end - start) * min(cur_t, d) / d + s for d in end_t]
    else:          
      return (end - start) * min(cur_t, end_t) / end_t + start     
  
class MDP(ABC): 
  def __init__(self):
    super().__init__()        

  @abstractmethod
  def reset(self, batch_size):
    # return states with type np.array of size (batch_size, x_size)
    pass

  @abstractmethod
  def act(self, actions):
    # return rewards with type np.array of size (batch_size)
    # actions: action with type np.array of size (batch_size, action_size)
    pass

class complex_multiplexer_MDP(MDP):
  def __init__(self, addr_size=2, action_size=2, zero=True, reward_zero=True):
    self.addr_size=addr_size
    self.action_size=action_size      
    self.x_size= addr_size*action_size + 2 ** addr_size
    self.zero = zero
    self.reward_zero = reward_zero
    super().__init__()

  def reset(self, batch_size):
    addr_size = self.addr_size
    action_size = self.action_size
    x_size = self.x_size        
    self.x = np.random.binomial(1, 0.5, size=(batch_size, x_size))
    addr = np.sum(self.x[:, :addr_size*action_size].reshape([-1, action_size, addr_size]) * 2**(addr_size-1-np.arange(addr_size)), axis=2)        
    self.y = self.x[np.arange(self.x.shape[0])[:, np.newaxis], addr_size*action_size + addr]
    if not self.zero: self.y = zero_to_neg(self.y)
    return self.x if self.zero else zero_to_neg(self.x)

  def act(self, actions):
    corr = (self.y == actions.astype(np.int32)).astype(np.float32)        
    return corr if self.reward_zero else zero_to_neg(corr)

  def expected_reward(self, p):
    # for action_size=2 only
    reward_f = np.full((self.x.shape[0], 2), 0 if self.reward_zero else -1)
    y_zero = self.y if self.zero else neg_to_zero(self.y)
    reward_f[np.arange(self.x.shape[0]), y_zero[:,0].astype(np.int)] = 1
    return np.sum(reward_f * p, axis=-1)
      
class simple_grad_optimizer():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def delta(self, name, grads, learning_rate=None):                  
        learning_rate = self.learning_rate if learning_rate is None else learning_rate                
        return [learning_rate*i for i in grads]
    
class adam_optimizer():
    def __init__(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-09):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self._cache = {}        
    
    def delta(self, grads, name="w", learning_rate=None, gate=None):
        if name not in self._cache:
            self._cache[name] = [[np.zeros_like(i) for i in grads],
                                 [np.zeros_like(i) for i in grads],
                                 0]
        self._cache[name][2] += 1 
        t = self._cache[name][2]
        deltas = []
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        learning_rate = self.learning_rate if learning_rate is None else learning_rate        
        for n, g in enumerate(grads):                
            m = self._cache[name][0][n]
            v = self._cache[name][1][n]
            m = beta_1 * m + (1 - beta_1) * g
            v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
            self._cache[name][0][n] = m
            self._cache[name][1][n] = v            

            m_hat = m / (1 - (np.power(beta_1, t) if t < 1000 else 0))
            v_hat = v / (1 - (np.power(beta_2, t) if t < 1000 else 0))
            deltas.append(learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))
        
        return deltas 

# Following are util funcitons for learning value function

class batch_envs():
  
  def __init__(self, name, batch_size=1, rest_n=100, warm_n=100):
    self._batch_size = int(batch_size)
    self._action = None
    self._reward = np.zeros(batch_size)
    self._isEnd = np.ones(batch_size, bool)
    self._truncatedEnd = np.zeros(batch_size, bool)
    self._rest = np.zeros(batch_size)
    self._warm = np.zeros(batch_size)
    self._state = np.zeros((batch_size,)+gym.make(name).reset().shape)
    self._stateCode = np.zeros(batch_size, np.int)
    self._rest_n = rest_n
    self._warm_n = warm_n     
    self._env = [gym.make(name) for _ in range(batch_size)]    

  @property
  def name(self):
    return self._name

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def reward(self):
    return self._reward

  @property
  def action(self):
    return self._action

  @property
  def action_space(self):
    return self._env[0].action_space

  @property
  def isEnd(self):
    return self._isEnd

  @property
  def stateCode(self):
    return self._stateCode

  @property
  def state(self):
    return self._state    

  @property
  def info(self):
    return {"stateCode": self._stateCode, "truncatedEnd": self._truncatedEnd}
  
  def step(self, action):
    self._rest[self._isEnd] += 1
    self._warm += 1
    self.reset(self._rest>self._rest_n)      
    isLive = np.logical_and(self._warm>self._warm_n, ~self._isEnd)
    self._reward[~isLive] = 0
    for i in isLive.nonzero()[0]:   
      self._state[i], self._reward[i], self._isEnd[i], info = self._env[i].step(action[i])  
      self._truncatedEnd[i] = info['TimeLimit.truncated'] if 'TimeLimit.truncated' in info else False
      if self._truncatedEnd[i]: self._rest[i] = self._rest_n

    # Live  
    self._stateCode[isLive] = 0            
    # Rest
    self._stateCode[self._rest>=1] = 1
    # Warm up
    self._stateCode[self._warm<= self._warm_n] = 3    
    # Reset
    self._stateCode[self._warm==0] = 2

    return self.state, self.reward, self._isEnd, self.info

  def reset(self, index=None):    
    for i in range(self._batch_size) if index is None else index.nonzero()[0]:
      self._state[i] = self._env[i].reset()
      self._reward[i] = 0
    if index is None: index = slice(None)  
    self._rest[index] = 0
    self._warm[index] = 0
    self._truncatedEnd[index] = False
    self._isEnd[index] = False
    return self._state

class uniform_policy():
  def __init__(self, batch_size, actions):
    self.actions = np.array(actions, np.int)
    self.batch_size = batch_size
  
  def __call__(self, state):
    return np.random.choice(self.actions, size=self.batch_size)    

def mv(a, n=1000) :
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n      

def plot(curves, names, mv_n=100, end_n=1000, title=None, xlabel="Episode", 
         ylabel="Episode Return", ylim=None, legend=False, loc=4, ax=None, size=1,
         color=None):  
  lines = []
  new_fig = ax is None
  if new_fig: plt.figure(figsize=(5*size, 3.5*size), dpi=80) 
  for i, m in enumerate(names.keys()):    
    v = np.array([mv(j[:end_n], mv_n) for j in curves[m]])
    r_std = np.std(v, axis=0)/np.sqrt(len(curves[m]))        
    r_avg = np.mean(v, axis=0)    
    r_std = np.concatenate([np.full([mv_n-1,],np.nan), r_std])
    r_avg = np.concatenate([np.full([mv_n-1,],np.nan), r_avg])
    
    k = names[m]
    if new_fig: ax = plt.gca()         
    x = np.arange(0, len(r_avg), len(r_avg)//50)
    
    cmap = plt.get_cmap("tab10")
    c = cmap(i) if color is None else color[i]
    
    #lines.append(ax.errorbar(x, r_avg[x], r_std[x], label=None, color=c))      
    lines.append(ax.plot(x, r_avg[x], label=k, color=c))
    c = tuple(i*0.2+0.8 for i in c)
    ax.fill_between(x, (r_avg-r_std)[x], (r_avg+r_std)[x], label=None, color=c)
      
  if title is not None: ax.set_title(title)
  if xlabel is not None: ax.set_xlabel(xlabel)  
  if ylabel is not None: ax.set_ylabel(ylabel)    
  if legend: ax.legend(loc=loc)    
  return lines  

def print_stat(curves, names, end_n=None):
  for m in names.keys():
    print("Stat on %s(%s):" % (names[m],m))
    if end_n is None: end_n = len(curves[m][0])
    r = np.array(curves[m])[:, :end_n]
    r = np.average(r, axis=1)
    print("Avg. Return: mean %.2f median %.2f min %.2f max %.2f std %.2f" % (np.average(r),
                                                                            np.median(r),
                                                                            np.amin(r),
                                                                            np.amax(r),
                                                                            np.std(r)))#/np.sqrt(len(r)))) 