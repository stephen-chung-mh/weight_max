#v2.3

import numpy as np 
from util import *

L_SOFTPLUS = 0
L_RELU = 1
L_LINEAR = 2
L_SIGMOID = 3
L_DISCRETE = 4
L_BINARY_Z = 5
L_BINARY_N = 6
LS_REAL = [L_SOFTPLUS, L_RELU, L_LINEAR, L_SIGMOID]
LS_DISCRETE = [L_DISCRETE, L_BINARY_Z, L_BINARY_N, ]


ACT_F = {L_SOFTPLUS: softplus,
                 L_RELU: relu,
                 L_SIGMOID: sigmoid,
                 L_LINEAR: lambda x: x,
                 L_BINARY_Z: sigmoid,
                 L_BINARY_N: lambda x: 2*sigmoid(x)-1,
                 }

ACT_D_F = {L_SOFTPLUS: sigmoid,
                     L_RELU: relu_d,
                     L_SIGMOID: sigmoid_d,
                     L_LINEAR: lambda x: 1,
                     L_BINARY_Z: sigmoid_d,
                     L_BINARY_N: lambda x: 2*sigmoid_d(x),
                     }

class eq_prop_layer():
    def __init__(self, name, input_size, output_size, optimizer, var, temp, l_type):
        if l_type not in [L_SOFTPLUS, L_RELU, L_LINEAR, L_SIGMOID, L_DISCRETE, L_BINARY_Z, L_BINARY_N, ]:
            raise Exception('l_type (%d) not implemented' % l_type)

        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.l_type = l_type
        self.temp = temp if l_type in LS_DISCRETE else 1
        
        lim = np.sqrt(6 / (input_size + output_size))       
        if l_type == L_DISCRETE: output_size -= 1
        self._w = np.random.uniform(-lim, lim, size=(input_size, output_size))
        self._b = np.random.uniform(-1e-3, 1e-3, size=(output_size))
        self._inv_var = np.full(output_size, 1/var) if var > 0 else None
        
        self.prev_layer = None # Set manually
        self.next_layer = None # Set manually
        self.values = np.zeros((1, output_size))
        
        self.w_trace = np.zeros((1, input_size, output_size,))
        self.b_trace = np.zeros((1, output_size,)) 
                
    def sample(self, inputs, det=False):    
        self.compute_pot_mean(inputs)          
        if self.l_type in LS_REAL:
            if self._inv_var is None or det:
                self.values = self.mean
            else:
                sigma = np.sqrt(1/self._inv_var)
                self.values = self.mean + sigma * np.random.normal(size=self.pot.shape)              
        elif self.l_type == L_DISCRETE:      
            self.values = multinomial_rvs(n=1, p=self.mean)
        elif self.l_type in [L_BINARY_Z]:      
            if not det:
                self.values = np.random.binomial(1, self.mean, size=self.mean.shape)
            else:
                self.values = (self.mean > 0.5).astype(np.float)
        elif self.l_type == L_BINARY_N:
            if not det:
                self.values = np.random.binomial(1, (self.mean+1)/2, size=self.mean.shape)*2-1
            else:
                self.values = (self.mean > 0).astype(np.float)
                
        return self.values
    
    def compute_pot_mean(self, inputs):
        # Compute potential (pre-activated mean value) and mean value of layer
        self.inputs = inputs    
        self.pot = (inputs.dot(self._w) + self._b)/self.temp
        if self.l_type in LS_REAL + [L_BINARY_Z, L_BINARY_N]:        
            self.mean = ACT_F[self.l_type](self.pot)
        elif self.l_type == L_DISCRETE:    
            self.pot = np.concatenate([self.pot, np.zeros((inputs.shape[0], 1))], axis=-1)
            self.mean = softmax(self.pot, axis=-1)
            
    def record_trace(self, lambda_=0, det=False, zero_learn=False):
        dev = 1 if det else (self.values - self.mean)    
        if self.l_type in LS_REAL:
            v_ch = dev * ACT_D_F[self.l_type](self.pot) * self._inv_var
        elif self.l_type in [L_BINARY_Z, L_BINARY_N]:
            v_ch = dev / self.temp            
                
        elif self.l_type == L_DISCRETE:  
            v_ch = (dev / self.temp)[:, :-1]     
        
        if det: v_ch = v_ch * recip_z(self.values)
        if zero_learn: v_ch[self.values == 0] = 0.   
        
        self.w_trace = lambda_ * self.w_trace + self.inputs[:, :, np.newaxis] * v_ch[:, np.newaxis, :] 
        self.b_trace = lambda_ * self.b_trace + v_ch    
            
    def learn_trace(self, reward, lr=0.01, w_reg=0, w_reg_p=0, h_reg=0, entro_reg=0): 
        self.reward = reward
        if len(reward.shape) == 1: reward = reward[:, np.newaxis]
        w_update = self.w_trace * reward[:, np.newaxis, :]   
        b_update = self.b_trace * reward
        
        self.w_update = w_update
        self.b_update = b_update
                
        if h_reg > 0:            
            v_ch = h_reg * ACT_D_F[self.l_type](self.pot) / self.temp
            w_update -= self.inputs[:, :, np.newaxis] * v_ch[:, np.newaxis, :] 
            b_update -= v_ch     
            
        if entro_reg > 0:
            if self.l_type == L_BINARY_Z:            
                en = entro_reg*(-softplus(-self.mean) + softplus(-1+self.mean))*sigmoid_d(self.pot)                
                w_update -= self.inputs[:, :, np.newaxis] * en[:, np.newaxis, :] 
                b_update -= en
                self.en = en
            else:
                raise Exception("Not defined.")
        
        if w_reg > 0:
            w_update -= w_reg * w_reg_p * np.power(np.abs(self._w), w_reg_p-1) * np.sign(self._w) 
        
        w_update = np.average(w_update, axis=0) 
        b_update = np.average(b_update, axis=0)
        delta_w = self.optimizer.delta(grads=[w_update], name=self.name+"_w", learning_rate=lr)[0]    
        delta_b = self.optimizer.delta(grads=[b_update], name=self.name+"_b", learning_rate=lr)[0]    
        
        self._w += delta_w    
        self._b += delta_b
            
    def clear_trace(self, mask):    
        self.w_trace = self.w_trace * (mask.astype(np.float))[:, np.newaxis, np.newaxis]
        self.b_trace = self.b_trace * (mask.astype(np.float))[:, np.newaxis]
 
    def backprop(self, grad, w_reg=0, w_reg_p=0, h_reg=0, entro_reg=0, lambda_=0):   
        if self.next_layer is None:
            v_ch = grad #* ACT_D_F[self.l_type](self.pot) 
        else:
            v_ch = grad * ACT_D_F[self.l_type](self.pot) 
        if self.l_type == L_DISCRETE: v_ch = v_ch[:, :-1]    
        w_update = self.inputs[:, :, np.newaxis] * v_ch[:, np.newaxis, :]    
        b_update = v_ch            
        self.w_trace = self.w_trace * lambda_ + w_update
        self.b_trace = self.b_trace * lambda_ + b_update        
        return v_ch.dot(self._w.T)          
            
class Network():
    def __init__(self, state_n, action_n, hidden, var, temp, hidden_l_type, 
                 output_l_type, opt="adam", beta_1=0.9, beta_2=0.999):
        
        self.layers = []    
        in_size = state_n
        
        if opt == "adam":
            optimizer = adam_optimizer(learning_rate=0.01, beta_1=beta_1, beta_2=beta_2, epsilon=1e-08)            
        else:
            optimizer = simple_grad_optimizer(learning_rate=0.01)        

        for d, n in enumerate(hidden + [action_n]):
            a = eq_prop_layer(name="layer_%d"%d, input_size=in_size, output_size=n, 
                                                optimizer=optimizer, var=getl(var,d), temp=getl(temp,d),
                                                l_type=(output_l_type if d==len(hidden) else hidden_l_type))
            if d > 0: 
                a.prev_layer = self.layers[-1]        
                self.layers[-1].next_layer = a 
            self.layers.append(a)                  
            in_size = n

    def forward(self, state, det=False):        
        self.state = state
        h = state   
        for n, a in enumerate(self.layers): h = a.sample(h, det=getl(det,n))                    
        self.action = h                
        return self.action      

    def backprop(self, targ, grad_mean, w_reg=0, w_reg_p=0, h_reg=0, entro_reg=0, lambda_=0):        
        ll = self.layers[-1]
        if targ is None:      
            if grad_mean:
                grad = np.array([[1.]])
            else:
                grad = (ll.values - ll.mean) / ll.temp                   
        else:
            grad = (targ - ll.mean) / ll.temp      
        
        for n in range(len(self.layers)-1, -1, -1):
            grad = self.layers[n].backprop(grad, w_reg=getl(w_reg, n), w_reg_p=getl(w_reg_p, n), 
                                                                         h_reg=getl(h_reg, n), entro_reg=getl(entro_reg, n), lambda_=lambda_)            
            
    def learn(self, reward, lr=0.01, w_reg=0, w_reg_p=0, h_reg=0, entro_reg=0, weight_max=0, 
                        baseline=False, weight_beta=0, det=False,):    
        for n in range(len(self.layers)-1, -1, -1):
            a = self.layers[n]            
            a.learn_trace(reward=reward, lr=getl(lr, n), w_reg=getl(w_reg, n), w_reg_p=getl(w_reg_p, n), 
                                        h_reg=getl(h_reg, n), entro_reg=getl(entro_reg,n))
            if n > 0  and weight_max != 0:        
                w = a._w[np.newaxis, :, :]
                ch_w = weight_max * np.power(np.abs(w), weight_max-1) * np.sign(w) *a.w_update
                if baseline:
                    ch_w -=  ch_w * (self.layers[n-1].mean / self.layers[n-1].values)[..., np.newaxis]
                reward = np.sum(ch_w, axis=-1)                
                    
    def clear_trace(self, mask):
        for n, a in enumerate(self.layers): a.clear_trace(mask)
            
    def clear_values(self, mask):
        for n, a in enumerate(self.layers): a.clear_values(mask)         
