#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:17:28 2019

@author: fsc
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

from models_para_tf import Eikonal2DnetCV2RPF
import entropy_estimators as ee


np.random.seed(1234)

def plot_ensemble(T_star, CV_star, X_train, Y_train, filename = None):
    plt.set_cmap('jet_r')
    scale = np.linspace(0,0.75, 16)
    scaleCV = np.linspace(0.9,1.5, 16)
    plt.close('all')
    fig = plt.figure(1)
    fig.set_size_inches((10,15))
    plt.subplot(321)
    plt.contourf(X_m, Y_m, T_star.mean(1).reshape(X_m.shape), scale)  
    plt.colorbar()
    plt.scatter(X_train, Y_train)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('predicted activation times [ms]')

    plt.subplot(322)
    plt.contourf(X_m, Y_m, (T_star.mean(1) - T[:,0]).reshape(X_m.shape))  
    plt.colorbar()
    plt.scatter(X_train, Y_train) 
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('activation time error [ms]')

    plt.subplot(323)
    plt.contourf(X_m, Y_m, CV_star.mean(1).reshape(X_m.shape))  
    plt.colorbar()
    plt.scatter(X_train, Y_train)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('predicted conduction velocity [mm/ms]')

    plt.subplot(324)
    plt.contourf(X_m, Y_m, (CV[:,0] - CV_star.mean(1)).reshape(X_m.shape))  
    plt.colorbar()
    plt.scatter(X_train, Y_train)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('conduction velocity error [mm/ms]')

    plt.set_cmap('jet')
    plt.subplot(325)
    plt.contourf(X_m, Y_m, T_star.std(1).reshape(X_m.shape))  
    plt.colorbar()
    plt.scatter(X_train, Y_train)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('activation times std [ms]')

    ent = np.apply_along_axis(lambda x: ee.entropy(x[:,None]), 1, T_star)

    plt.subplot(326)
    plt.contourf(X_m, Y_m, ent.reshape(X_m.shape))  
    plt.colorbar()
    plt.scatter(X_train, Y_train)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('activation times entropy [-]')
    
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)

if __name__ == "__main__":
    
    def exact(X, Y):
        return np.minimum(np.sqrt(X**2 + Y**2), 0.7*np.sqrt((X - 1)**2 + (Y - 1)**2))
    
    def CVexact(X, Y):
        mask = np.less_equal(np.sqrt(X**2 + Y**2), 0.7*np.sqrt((X - 1)**2 + (Y - 1)**2))
        return mask*1.0 + ~mask*1.0/0.7
    
# create the data
    N_grid = 50         
    x = y = np.linspace(0,1,N_grid)[:,None]
    # start training with a low number of samples
    N_train = 10
  
    X_m, Y_m = np.meshgrid(x,y)
    X = X_m.flatten()[:,None]
    Y = Y_m.flatten()[:,None]
    T = exact(X,Y)
    CV = CVexact(X,Y)
            
    X_train_all = lhs(2, N_train)
    X_train = X_train_all[:,:1]
    Y_train = X_train_all[:,1:]
    T_train = exact(X_train, Y_train)
    
#   network parameters     
    X_pde = X
    Y_pde = Y

    layers = [2,20,20,20,20,20,1]
    CVlayers = [2,5,5,5,5,1]

    Batch = 30 # this is the number of networks to train in parallel
    
    CVmax = 1.5

    model = Eikonal2DnetCV2RPF(X_pde, Y_pde, X_train, Y_train, T_train, 
                               layers, CVlayers, Batch, C = CVmax, alpha = 1e-7, alphaL2 = 1e-9)

    #start training the model with the initial dataset
    model.train_Adam_minibatch(20000 + 5000*(N_train - 10), size = 50)

    T_star, CV_star = model.predict(X,Y) 
    plot_ensemble(T_star, CV_star, X_train, Y_train, filename = 'results/AL_NNpara_0.pdf')

    N_AL = 40 # number of samples to acquire with active learning

    # store how the error is evolving
    errorsT = [np.sqrt(np.average((T_star.mean(1) - T[:,0])**2))]
    errorsCV = [np.average(np.abs(CV_star.mean(1) - CV[:,0]))]
    T_stars = [T_star.mean(1)]
    CV_stars = [CV_star.mean(1)]
    print('RMSE:',errorsT[-1])
    print('RMSE CV:',errorsCV[-1])
    
    # list of available candidates for sample during active learning
    available = list(range(X.shape[0]))
    # start active learning
    for i in range(N_AL):
        # compute the entropy for the available candidates
        ent = np.apply_along_axis(lambda x: ee.entropy(x[:,None]), 1, T_star[available])

        idx_next = ent.argmax() # find the point of maximum entropy
        
        # add it to the dataset
        x_next, y_next = X[available][idx_next], Y[available][idx_next] 
        T_next = exact(x_next, y_next)
        model.add_point(x_next, y_next, T_next)
        available.remove(available[idx_next])
        
        # and continue training
        model.train_Adam_minibatch(5000, size = 96)

        # predict and evaluate the error
        T_star, CV_star = model.predict(X,Y) 
        plot_ensemble(T_star, CV_star, model.x_e, model.y_e, filename = 'results/AL_NNpara_%i.pdf' % (i+1))

        errorsT.append(np.sqrt(np.average((T_star.mean(1) - T[:,0])**2)))
        errorsCV.append(np.average(np.abs(CV_star.mean(1) - CV[:,0])))
        print(i,'RMSE:',errorsT[-1])
        print(i,'RMSE CV:',errorsCV[-1])
            
        
        
        
        
    
            