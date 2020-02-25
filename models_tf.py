#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:15:14 2018

@author: Paris
"""


import tensorflow as tf
import numpy as np
import time
from pyDOE import lhs

tf.random.set_random_seed(1234)
np.random.seed(1234)


        
class Eikonal2DnetCV2:
    # Initialize the class
    def __init__(self, x, y, x_e, y_e, T_e, layers, CVlayers, C = 1.0, alpha = 1e-5, alphaL2 = 1e-6, jobs = 4):
        
        X = np.concatenate([x, y], 1)
  #      X_e = np.concatenate([x_e, t_e], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
     #   self.X_e = X_e
        
        self.x = x
        self.y = y
        
        self.T_e = T_e
        self.x_e = x_e
        self.y_e = y_e
        
        self.layers = layers
        self.CVlayers = CVlayers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)  
        self.CVweights, self.CVbiases = self.initialize_NN(CVlayers)  
        
        
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
        
        self.T_e_tf = tf.placeholder(tf.float32, shape=[None, self.T_e.shape[1]]) 
        self.x_e_tf = tf.placeholder(tf.float32, shape=[None, self.x_e.shape[1]]) 
        self.y_e_tf = tf.placeholder(tf.float32, shape=[None, self.y_e.shape[1]]) 
        

        self.T_pred, self.CV_pred, self.f_T_pred, self.f_CV_pred = self.net_eikonal(self.x_tf, self.y_tf)
                
        self.T_e_pred, self.CV_e_pred, self.f_T_e_pred, self.f_CV_e_pred = self.net_eikonal(self.x_e_tf, self.y_e_tf)
                        
        self.loss = tf.reduce_mean(tf.square(self.T_e_tf - self.T_e_pred)) + \
                    tf.reduce_mean(tf.square(self.f_T_e_pred)) + \
                    tf.reduce_mean(tf.square(self.f_T_pred)) + \
                    self.alpha*tf.reduce_mean(tf.square(self.f_CV_e_pred)) + \
                    self.alpha*tf.reduce_mean(tf.square(self.f_CV_pred))  + \
                    sum([self.alphaL2*tf.nn.l2_loss(w) for w in self.weights])
                    

                    
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

        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
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
        C = self.C
        T = self.neural_net(tf.concat([x,y], 1), self.weights, self.biases)
        CV = self.neural_net(tf.concat([x,y], 1), self.CVweights, self.CVbiases)
        
        CV = C*tf.sigmoid(CV)
        
        T_x = tf.gradients(T, x)[0]
        T_y = tf.gradients(T, y)[0]
        
        CV_x = tf.gradients(CV, x)[0]
        CV_y = tf.gradients(CV, y)[0]
        
        f_T = tf.sqrt(T_x**2 + T_y**2) - 1.0/CV
        f_CV = tf.sqrt(CV_x**2 + CV_y**2)
        
        return T, CV, f_T, f_CV
        

    
    def callback(self, loss):
        self.lossit.append(loss)
        print('Loss: %.5e' % (loss))
      
    def train(self): 

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, 
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.T_e_tf: self.T_e}
        
        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)
        
    def train_Adam(self, nIter): 
        
        self.lossit = []

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, 
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.T_e_tf: self.T_e}        
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
        
        self.lossit = []
       
        start_time = time.time()
        for it in range(nIter):
            X = lhs(2, size)
            tf_dict = {self.x_tf: X[:,:1], self.y_tf: X[:,1:], 
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.T_e_tf: self.T_e} 
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
            
 
    def predict(self, x_star, y_star):
        
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
    
    

class Eikonal3DnetCV2:
    # Initialize the class
    def __init__(self, X, normals,X_e, T_e, layers, CVlayers, Tmax, C = 1.0, alpha = 1e-5, alphaL2 = 1e-6, jobs = 4):
        
  #      X_e = np.concatenate([x_e, t_e], 1)
        self.Tmax = Tmax
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        self.normals = normals
     #   self.X_e = X_e

        
        self.T_e = T_e
        self.X_e = X_e

        
        self.layers = layers
        self.CVlayers = CVlayers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)  
        self.CVweights, self.CVbiases = self.initialize_NN(CVlayers)  
        
        
        self.C = tf.constant(C)
        self.alpha = tf.constant(alpha)
        self.alphaL2 = alphaL2
        

        
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True,
                                                     intra_op_parallelism_threads=jobs,
                                                     inter_op_parallelism_threads=jobs,
                                                     device_count={'CPU': jobs}))
        
        self.X_tf = tf.placeholder(tf.float32, shape=[None, self.X.shape[1]])
        self.normals_tf = tf.placeholder(tf.float32, shape=[None, self.X.shape[1]])
        
        self.T_e_tf = tf.placeholder(tf.float32, shape=[None, self.T_e.shape[1]]) 
        self.X_e_tf = tf.placeholder(tf.float32, shape=[None, self.X_e.shape[1]]) 
        

        self.T_pred, self.CV_pred, self.f_T_pred, self.f_CV_pred, self.f_N_pred = self.net_eikonal(self.X_tf, self.normals_tf)
                
        self.T_e_pred, self.CV_e_pred = self.net_data(self.X_e_tf)
        self.pde_loss = tf.reduce_mean(tf.square(self.f_T_pred))  
        self.normal_loss = 1e-2*tf.reduce_mean(tf.square(self.f_N_pred))               
        self.loss = tf.reduce_mean(tf.square(self.T_e_tf - self.T_e_pred)) + \
                    self.pde_loss + \
                    self.normal_loss + \
                    self.alpha*tf.reduce_mean(tf.square(self.f_CV_pred))  + \
                    sum([self.alphaL2*tf.nn.l2_loss(w) for w in self.weights])
                    

                    
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

        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
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
    
    def net_eikonal(self, X, normals):
        C = self.C
        T = self.neural_net(X, self.weights, self.biases)
        CV = self.neural_net(X, self.CVweights, self.CVbiases)
        
        CV = C*tf.sigmoid(CV)
        
        T_x = tf.gradients(T, X)[0]
       
        CV_x = tf.gradients(CV, X)[0]

        
        f_T = CV*tf.norm(self.Tmax*(T_x), axis = -1, keepdims = True) - 1.0
        f_CV = tf.norm(CV_x, axis = -1)
        f_N = self.C*self.Tmax*tf.reduce_sum(T_x*normals, axis = -1)
        #f_N = 0.0
        return T, CV, f_T, f_CV, f_N
    
    def net_data(self, X):
        C = self.C
        T = self.neural_net(X, self.weights, self.biases)
        CV = self.neural_net(X, self.CVweights, self.CVbiases)
        
        CV = C*tf.sigmoid(CV)
        
        return T, CV
        

    
    def callback(self, loss):
        self.lossit.append(loss)
        print('Loss: %.5e' % (loss))
      
    def train(self): 

        tf_dict = {self.X_tf: self.X,  
                   self.normals_tf: self.normals,
                   self.X_e_tf: self.X_e,
                   self.T_e_tf: self.T_e}
        
        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)
        
    def train_Adam(self, nIter): 
        
        self.lossit = []

        tf_dict = {self.X_tf: self.X,  
                   self.normals_tf: self.normals,
                   self.X_e_tf: self.X_e,
                   self.T_e_tf: self.T_e}
              
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
        
        self.lossit = []
       
        start_time = time.time()
        idx_global = np.arange(self.X.shape[0])
        np.random.shuffle(idx_global)
        splits = np.array_split(idx_global, idx_global.shape[0]//size)
        for ep in range(nEpoch):
            for it, idx in enumerate(splits):
                tf_dict = {self.X_tf: self.X[idx],
                           self.normals_tf: self.normals[idx],
                           self.X_e_tf: self.X_e, 
                           self.T_e_tf: self.T_e}
                self.sess.run(self.train_op_Adam, tf_dict)
                loss_value = self.sess.run(self.loss, tf_dict)
                self.lossit.append(loss_value)
            # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    pde_loss = self.sess.run(self.pde_loss, tf_dict)
                    normal_loss = self.sess.run(self.normal_loss, tf_dict)
                    print('Epoch: %d, It: %d, Loss: %.3e, pde loss: %.3e, normal_loss: %.3e, Time: %.2f' % 
                          (ep, it  + ep*idx_global.shape[0]//size, loss_value, pde_loss, normal_loss, elapsed))
                    start_time = time.time()
            
 
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

