#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:30:53 2019

@author: fsc
"""

import tensorflow as tf
import timeit
import numpy as np
import time
from pyDOE import lhs

tf.random.set_random_seed(1234)
np.random.seed(1234)


        
class Eikonal2DnetCV2RPF:
    # Initialize the class
    def __init__(self, x, y, x_e, y_e, T_e, layers, CVlayers, Batch,
                 C = 1.0, alpha = 1e-6, alphaL2 = 1e-6, jobs = 8, noise_level = 0.01):
        
        self.Batch = Batch
        self.noise_level = noise_level
        X = np.concatenate([x, y], 1)
  #      X_e = np.concatenate([x_e, t_e], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
     #   self.X_e = X_e
        #normalization
        self.x = x#2.0*(x - self.lb[0])/(self.ub[0] - self.lb[0]) - 1.0
        self.y = y#2.0*(y - self.lb[1])/(self.ub[1] - self.lb[1]) - 1.0
        
        self.N_data = T_e.shape[0]
        
        T_e = np.tile(T_e[None,:], (self.Batch,1,1)) + self.noise_level*np.random.rand(self.Batch, self.N_data, 1)
        
        self.T_e = T_e
        self.x_e = x_e#2.0*(x_e - self.lb[0])/(self.ub[0] - self.lb[0]) - 1.0
        self.y_e = y_e#2.0*(y_e - self.lb[1])/(self.ub[1] - self.lb[1]) - 1.0
        
        self.layers = layers
        self.CVlayers = CVlayers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers) 
        self.prior_weights, self.prior_biases = self.initialize_NN(layers, trainable = False)  
        self.CVweights, self.CVbiases = self.initialize_NN(CVlayers)  
        self.CVprior_weights, self.CVprior_biases = self.initialize_NN(layers, trainable = False) 
        
        self.C = tf.constant(C)
        self.alpha = tf.constant(alpha)
        self.alphaL2 = alphaL2
        

        
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True,
                                                     intra_op_parallelism_threads=jobs,
                                                     inter_op_parallelism_threads=jobs,
                                                     device_count={'CPU': jobs}))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        
        self.T_e_tf = tf.placeholder(tf.float32, shape=[Batch, None, self.x_e.shape[1]]) 
        self.x_e_tf = tf.placeholder(tf.float32, shape=[None, self.x_e.shape[1]]) 
        self.y_e_tf = tf.placeholder(tf.float32, shape=[None, self.y_e.shape[1]]) 
        self.dummy_x0_tf = tf.placeholder(tf.float32, shape=(None, Batch))
        

        self.T_pred, self.CV_pred, self.f_T_pred, self.f_CV_pred = self.net_eikonal(self.x_tf, self.y_tf)
                
        self.T_e_pred = self.net_T(self.x_e_tf, self.y_e_tf)
                        
        self.loss = np.float32(self.Batch)*(tf.reduce_mean(tf.square(self.T_e_tf - self.T_e_pred)) + \
                    tf.reduce_mean(tf.square(self.f_T_pred)) + \
                    self.alpha*tf.reduce_mean(tf.square(self.f_CV_pred)))  + \
                    sum([self.alphaL2*tf.nn.l2_loss(w) for w in self.weights])
                    

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) 
        # Define optimizer (use L-BFGS for better accuracy)       
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 10000,
                                                                           'maxfun': 10000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.lossit = []

        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers, trainable = True):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim + self.Batch) / 3.)
            return tf.Variable(tf.random_normal([self.Batch, in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32, trainable = trainable)   

        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([self.Batch, 1, layers[l+1]], dtype=tf.float32), dtype=tf.float32, trainable = trainable)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_eikonal(self, x, y):
        X = tf.concat([x,y], 1)
        X_temp = tf.tile(tf.expand_dims(X, 0),[self.Batch,1,1])

        T = self.neural_net(X_temp, self.weights, self.biases)\
          + self.neural_net(X_temp, self.prior_weights, self.prior_biases)
        CV = self.neural_net(X_temp, self.CVweights, self.CVbiases)\
           + self.neural_net(X_temp, self.CVprior_weights, self.CVprior_biases)
        
        T = tf.squeeze(T, [2])
        T = tf.transpose(T)
        
        CV = tf.squeeze(CV, [2])
        CV = tf.transpose(CV)
        
        CV = self.C*tf.sigmoid(CV)
        
        T_x = self.fwd_gradients_0(T, x)
        T_y = self.fwd_gradients_0(T, y)
        
        CV_x = self.fwd_gradients_0(CV, x)
        CV_y = self.fwd_gradients_0(CV, y)
        
        f_T = tf.sqrt(T_x**2 + T_y**2) - 1.0/CV
        f_CV = tf.sqrt(CV_x**2 + CV_y**2)
        
        return T, CV, tf.transpose(f_T), tf.transpose(f_CV)
    
    def net_T(self, x, y):
        X = tf.concat([x,y], 1)
        X_temp = tf.tile(tf.expand_dims(X, 0),[self.Batch,1,1])

        T = self.neural_net(X_temp, self.weights, self.biases)\
          + self.neural_net(X_temp, self.prior_weights, self.prior_biases)
        
        return T
    

    def fwd_gradients_0(self, U, x):        
        g = tf.gradients(U, x, grad_ys=self.dummy_x0_tf)[0]
        return tf.gradients(g, self.dummy_x0_tf)[0]
    
    def callback(self, loss):
        self.lossit.append(loss)
        print('Loss: %.5e' % (loss))
      
    def train(self): 

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, 
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.T_e_tf: self.T_e,
                   self.dummy_x0_tf: np.ones((self.x.shape[0], self.Batch))}
        
        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)
        
    def train_Adam(self, nIter): 
        
        self.lossit = []

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, 
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.T_e_tf: self.T_e,
                   self.dummy_x0_tf: np.ones((self.x.shape[0], self.Batch))}      
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.lossit.append(loss_value)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                C_value = np.exp(self.sess.run(self.C))
                print('It: %d, Loss: %.3e, C: %.3f, Time: %.2f' % 
                      (it, loss_value, C_value, elapsed))
                start_time = time.time()
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
        
    def train_Adam_minibatch(self, nIter, size = 50): 
        
        init_start_time = time.time()
        start_time = time.time()
        for it in range(nIter):
            X = lhs(2, size)
            tf_dict = {self.x_tf: X[:,:1], self.y_tf: X[:,1:], 
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.T_e_tf: self.T_e,
                   self.dummy_x0_tf: np.ones((size, self.Batch))} 
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.lossit.append(loss_value)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                C_value = np.exp(self.sess.run(self.C))
                print('It: %d, Loss: %.3e, C: %.3f, Time: %.2f' % 
                      (it, loss_value, C_value, elapsed))
                start_time = time.time()
        total_time = time.time() - init_start_time
        print('total time:', total_time)
        return total_time
        
 
    def predict(self, x_star, y_star):
#        x_star = 2.0*(x_star - self.lb[0])/(self.ub[0] - self.lb[0]) - 1.0
#        y_star = 2.0*(y_star - self.lb[1])/(self.ub[1] - self.lb[1]) - 1.0
#        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star,
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e}
        
        T_star = self.sess.run(self.T_pred, tf_dict)
        CV_star = self.sess.run(self.CV_pred, tf_dict)

        
        return T_star, CV_star
    
    def get_adaptive_points(self, N = 1000, M = 10):
        
        X = lhs(2, N)
        tf_dict = {self.x_tf: X[:,:1], self.y_tf: X[:,1:],
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e}
        
        f_T_star = self.sess.run(self.f_T_pred, tf_dict)
        
        ind = f_T_star[:,0].argsort()[-M:]
        
        return X[ind], f_T_star[ind]
    def add_point(self,x_new, y_new, T_new):
        self.x_e = np.concatenate((self.x_e, x_new[:,None]))
        self.y_e = np.concatenate((self.y_e, y_new[:,None]))
        T_new = np.tile(T_new[None,:,None], [self.Batch, 1,1]) + self.noise_level*np.random.randn(self.Batch,1,1)
        self.T_e = np.concatenate((self.T_e, T_new), axis = 1)
        
#%%
        
class Eikonalnet3DRPF:
    # Initialize the class
    def __init__(self, X, normals, X_e, T_e,
                 layers, CVlayers, Batch, Tmax,
                 C = 1.0, alpha = 1e-5, alphaL2 = 1e-6, jobs = 4, noise_level = 0.01):
        
        self.Batch = Batch
        self.noise_level = noise_level
        self.Tmax = Tmax
        
  #      X_e = np.concatenate([x_e, t_e], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        self.normals = normals
     #   self.X_e = X_e
        

        
        self.N_data = T_e.shape[0]
        
        T_e = np.tile(T_e[None,:], (self.Batch,1,1)) + self.noise_level*np.random.rand(self.Batch, self.N_data, 1)
        
        self.T_e = T_e
        self.X_e = X_e

        
        self.layers = layers
        self.CVlayers = CVlayers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers) 
        self.prior_weights, self.prior_biases = self.initialize_NN(layers, trainable = False)  
        self.CVweights, self.CVbiases = self.initialize_NN(CVlayers)  
        self.CVprior_weights, self.CVprior_biases = self.initialize_NN(layers, trainable = False) 
 
        
        
        self.C = tf.constant(C)
        self.alpha = tf.constant(alpha)
        self.alphaL2 = alphaL2
        

        
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True,
                                                     intra_op_parallelism_threads=jobs,
                                                     inter_op_parallelism_threads=jobs,
                                                     device_count={'CPU': jobs}))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.normals_tf = tf.placeholder(tf.float32, shape=[None, self.normals.shape[1]])


        self.T_e_tf = tf.placeholder(tf.float32, shape=[Batch, None, self.T_e.shape[-1]]) 
        self.X_e_tf = tf.placeholder(tf.float32, shape=[None, self.X_e.shape[1]]) 
        self.dummy_x0_tf = tf.placeholder(tf.float32, shape=(None, Batch))


        self.CV_pred, self.f_T_pred, self.f_CV_pred, self.f_N_pred, self.gradT = self.net_eikonal(self.x_tf, self.y_tf, self.z_tf, self.normals_tf)
                
        self.T_e_pred, self.CV_e_pred = self.net_data(self.X_e_tf)

       
        self.pde_loss = tf.reduce_mean(tf.square(self.f_T_pred))  
        self.normal_loss = 1e-3*tf.reduce_mean(tf.square(self.f_N_pred))              
        self.loss = np.float32(self.Batch)*(tf.reduce_mean(tf.square(self.T_e_tf - self.T_e_pred)) + \
                    self.pde_loss + \
                    self.normal_loss + \
                    self.alpha*tf.reduce_mean(tf.square(self.f_CV_pred))  + \
                    sum([self.alphaL2*tf.nn.l2_loss(w) for w in self.weights]))
                    

                    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) 
        # Define optimizer (use L-BFGS for better accuracy)       
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 10000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        self.lossit = []
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers, trainable = True):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim + self.Batch) / 2.)
            return tf.Variable(tf.random_normal([self.Batch, in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32, trainable = trainable)   

        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([self.Batch, 1, layers[l+1]], dtype=tf.float32), dtype=tf.float32, trainable = trainable)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_eikonal(self, x, y, z, normals):

        X_temp = tf.tile(tf.expand_dims(tf.concat((x,y,z), axis = 1), 0),[self.Batch,1,1])

        T = self.neural_net(X_temp, self.weights, self.biases)\
          + self.neural_net(X_temp, self.prior_weights, self.prior_biases)
        CV = self.neural_net(X_temp, self.CVweights, self.CVbiases)\
           + self.neural_net(X_temp, self.CVprior_weights, self.CVprior_biases)
        
        T = tf.squeeze(T, [2])
        T = tf.transpose(T)
        
        CV = tf.squeeze(CV, [2])
        CV = tf.transpose(CV)
        
        CV = self.C*tf.sigmoid(CV)
        
        T_x = self.fwd_gradients_0(T, x) # batch, N_data,
        T_y = self.fwd_gradients_0(T, y) # batch, N_data,
        T_z = self.fwd_gradients_0(T, z) # batch, N_data,
#        
        CV_x = self.fwd_gradients_0(CV, x) # batch, N_data,
        CV_y = self.fwd_gradients_0(CV, y) # batch, N_data,
        CV_z = self.fwd_gradients_0(CV, z) # batch, N_data,
#       
        gradT = tf.stack((tf.transpose(T_x), tf.transpose(T_y), tf.transpose(T_z)), axis = -1)
        f_T = CV*self.Tmax*tf.sqrt((T_x**2 + T_y**2 + T_z**2)) - 1.0
      #  f_T = CV*tf.norm(self.Tmax*(gradT), axis = -1, keepdims = True) - 1.0

        f_CV = tf.sqrt(CV_x**2 + CV_y**2 + CV_z**2)
        f_N = self.C*self.Tmax*(T_x*normals[:,0:1] + T_y*normals[:,1:2] + T_z*normals[:,2:])
      #  f_N = self.C*self.Tmax*tf.reduce_sum(gradT*normals, axis = -1)

    # f_N = 0.0 
    #    f_T = f_CV = f_N = T_x = 0.0
        return CV, f_T, f_CV, f_N, gradT
    def net_data(self,X):
        X_temp = tf.tile(tf.expand_dims(X, 0),[self.Batch,1,1])
        T = self.neural_net(X_temp, self.weights, self.biases)
        CV = self.C*tf.sigmoid(self.neural_net(X_temp, self.CVweights, self.CVbiases))
        return T, CV
        
    def fwd_gradients_0(self, U, x):        
        g = tf.gradients(U, x, grad_ys=self.dummy_x0_tf)[0]
        return tf.gradients(g, self.dummy_x0_tf)[0]
    
    def batch_grad(self,y, X):
        J = tf.map_fn(lambda m: tf.gradients(y[:,m:m+1], X)[0], tf.range(tf.shape(y)[-1]), tf.float32)
        return J
    
    def callback(self, loss):
        self.lossit.append(loss)
        print('Loss: %.5e' % (loss))
      
    def train(self): 

        tf_dict = {self.x_tf: self.X[:,0:1], self.y_tf: self.X[:,1:2], self.z_tf: self.X[:,2:],
                   self.X_e_tf: self.X_e, 
                   self.T_e_tf: self.T_e,
                   self.dummy_x0_tf: np.ones((self.X.shape[0], self.Batch)),
                   self.normals_tf: self.normals
                   }
        
        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)
        
    def train_Adam(self, nIter): 
        

        tf_dict = {self.x_tf: self.X[:,0:1], self.y_tf: self.X[:,1:2], self.z_tf: self.X[:,2:],
                   self.X_e_tf: self.X_e, 
                   self.T_e_tf: self.T_e,
                   self.dummy_x0_tf: np.ones((self.X.shape[0], self.Batch)),
                   self.normals_tf: self.normals
                   }
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.lossit.append(loss_value)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                C_value = np.exp(self.sess.run(self.C))
                print('It: %d, Loss: %.3e, C: %.3f, Time: %.2f' % 
                      (it, loss_value, C_value, elapsed))
                start_time = time.time()
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
        
    def train_Adam_minibatch(self, nEpoch, size = 50): 
        
        init_start_time = time.time()
        start_time = time.time()
        idx_global = np.arange(self.X.shape[0])
        np.random.shuffle(idx_global)
        splits = np.array_split(idx_global, idx_global.shape[0]//size)
        for ep in range(nEpoch):
            for it, idx in enumerate(splits):
                tf_dict = {self.x_tf: self.X[idx,0:1], self.y_tf: self.X[idx,1:2], self.z_tf: self.X[idx,2:],
                           self.X_e_tf: self.X_e, 
                           self.T_e_tf: self.T_e,
                           self.dummy_x0_tf: np.ones((idx.shape[0], self.Batch)),
                           self.normals_tf: self.normals[idx]
                           }
                self.sess.run(self.train_op_Adam, tf_dict)
                loss_value = self.sess.run(self.loss, tf_dict)
                self.lossit.append(loss_value)

            # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    pde_loss = self.sess.run(self.pde_loss, tf_dict)
                    normal_loss = self.sess.run(self.normal_loss, tf_dict)
                    print('Epoch: %d, It: %d, Loss: %.3e, pde loss: %.3e, normal loss: %.3e, Time: %.2f' % 
                          (ep, it  + ep*idx_global.shape[0]//size, loss_value, pde_loss, normal_loss,elapsed))
                    start_time = time.time()
                    
        print('total time:', time.time() - init_start_time)
            
 
    def predict(self, X_star):
        
        tf_dict = {self.X_e_tf: X_star}
        
        T_star = self.sess.run(self.T_e_pred, tf_dict)
        CV_star = self.sess.run(self.CV_e_pred, tf_dict)

        
        return T_star, CV_star
    
    def get_adaptive_points(self, N = 1000, M = 10):
        
        X = lhs(2, N)
        tf_dict = {self.x_tf: X[:,:1], self.y_tf: X[:,1:],
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e}
        
        f_T_star = self.sess.run(self.f_T_pred, tf_dict)
        
        ind = f_T_star[:,0].argsort()[-M:]
        
        return X[ind], f_T_star[ind]
    def add_point(self,X_new, T_new):
        self.X_e = np.concatenate((self.X_e, X_new[None,:]))
        T_new = np.tile(T_new[None,:,None], [self.Batch, 1,1]) + self.noise_level*np.random.randn(self.Batch,1,1)
        self.T_e = np.concatenate((self.T_e, T_new), axis = 1)


            