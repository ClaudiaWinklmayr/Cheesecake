# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import absolute_import, division, print_function
import time
import pandas as pd
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)


def lidar2bin(lidar_data, max_value, bin_lenght): 
    '''Inputs: 
    lider_data........ numpy array with timrlines from n lidars
    max_value......... maximum value of the bins to which the measured values are assigned
    bin length........ length of bins
    
    Maps a numpy array with timelines from n lidars to a binary array which for 
    each point in time indicates the bin in which a certain lidar value falls'''
    
    bins = np.arange(0, max_value, bin_lenght) # last entry is for everything larger than  max_value
    Nlidars = lidar_data.shape[1]
    lidar_onehot = np.zeros((lidar_data.shape[0], (len(bins)+1)*Nlidars))
    for i in range(Nlidars):
        lid = lidar_data[:, i]
        bin_idx = np.digitize(lid, bins)
        lidar_onehot[np.arange(0, len(lidar_data), 1), bin_idx + (i*Nlidars)] = 1 
    return lidar_onehot


def bin2lidar(lidar_onehot, max_value, bin_lenght): 
    ''' Inputs: 
    lidar_onehot........ binary representation of lidar timeline optained from lidar2bin()
    max_value........... maximum value of the bins to which the measured values are assigned
    bin length.......... length of bins 
    
    Reverses the procedure of lidar2bin()''''
    
    bins = np.arange(0, max_value+1, bin_lenght)
    lidar_rec = np.array([bins[np.argmax(lidar_onehot[t, :])] for t in range(lidar_onehot.shape[0])])
    return lidar_rec   


def lidar_preprocessing(lidar_data, cutoff = 2000, normalize = True): 
    '''Inputs: 
    lidar_data......... raw lidar measurements usually between 0 and 8000
    cutoff............. all values above are clipped
    normalize.......... if True the data is normalized to lie between 0 and 1 (after clipping)
    
    Typical lidar data lies between 0 and 8000 but all meaningful values are <2000. 
    Therefore this function clips the raw measurement. Normalization is useful to keep the weights
    of the classification network from exploding'''
    
    lidar = np.zeros(lidar_data.shape)
    for i in range(lidar.shape[1]):
        a = lidar_data[:, i].astype(float)
        a[np.where(a>cutoff)[0]] = cutoff
        if normalize:             
            a = a/np.max(a)
        lidar[:, i] = a
    return lidar


def joint_preprocessing(joint_data, normalize = True): 
    '''Inputs: 
    joint_data......... raw joint angle measurements usually between -pi and pi
    normalize.......... if True the data is normalized to lie between 0 and 1 
    
    Normalizes joint angle measurements (in rad) to values between 0 and 1 by 
    first shifting by pi and then dividing by the maxiaml value'''
    
    joints = np.zeros(joint_data.shape)
    for i in range(joints.shape[1]):
        a = joint_data[:, i].astype(float)
        a += np.pi
        if normalize:             
            a = a/np.max(a)
        joints[:, i] = a
    return joints


def build_model(inp_shape, outp_shape):
    ''' Creates a feed forward neural network for classifying lidar data (binarized with lidar2bin)
    using 2 hidden layers and a softmax output layer,. Weigths are manually initialized from a 
    normal distribution'''

    new_model = keras.Sequential()
    new_model.add(keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(inp_shape,)))
    new_model.add(keras.layers.Dense(outp_shape))
    new_model.add(keras.layers.Dense(outp_shape, activation='softmax'))

    new_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

    a = [wm.shape for wm in new_model.get_weights()]
    rand_weights = [np.random.normal(loc = 0, scale = 0.05, size = aa) for aa in a]
    new_model.set_weights(rand_weights)
    
    return new_model


class PrintDot(keras.callbacks.Callback):
    ''' callback to indicate training progress '''
    
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


def training(joints_data, lidar_data, epochs = 800, 
             binning_params = {'max_value':1, 'bin_lenght':0.05 }, model_name = 'model_lidar_'): 
    '''Inputs: 
    joints_data........... raw joint angle measurements (in rad)
    lidar_data............ raw lidar measurements 
    epochs................ number of traing epochs for the classifiers
    binning_params........ parameters for binarizing the lidar data 
    model_name............ prefix for model saving names
    
    the joint and lidar data are normalized using the preporcessing functions. Then for each 
    lidar a new instance of the classifier network is created, trained and saved. ''''
    
    joints = joint_preprocessing(joints_data)
    lidars = lidar_preprocessing(lidar_data)
    Nlidars = int(lidar_data.shape[1])
    Nbins = len(np.arange(0, binning_params['max_value'], binning_params['bin_lenght'])) +1
    Njoints = int(joints.shape[1])
    
    for lid in range(Nlidars): 
        print('\ntraining lidar {}'.format(lid))       
        model = build_model(Njoints, Nbins)
        binned_lidar = lidar2bin(np.reshape(lidars[:, lid], (lidars.shape[0], 1)), max_value=binning_params['max_value'], bin_lenght=binning_params['bin_lenght'])
        history = model.fit(joints, binned_lidar, epochs=epochs, validation_split=0.2, verbose=0, callbacks=[PrintDot()])
        model.save(model_name + str(lid) + '.h5')


def load_models(model_names = ['model_lidar_{}.h5'.format(i) for i in range(9)]):
    '''A list of models is loades using a list of names and the keras function load_model''' 
    models = []
    for i in range(len(model_names)): 
        models.append(keras.models.load_model(model_names[i]))
        print('Done loading model {}'.format(i))
    return models


def do_prediction(joint_data, models, binning_params = {'max_value':1, 'bin_lenght':0.05 }): 
    '''Inputs: 
    joints_data........... raw joint angle measurements (in rad)
    models................ list of classifier models loaded  with load models
    binning_params........ parameters for binarizing the lidar data 
    
    Loads the trained models for each lidar and performs prediction based on the 
    normalized joints data. The predicted liadr values are then transformed back from the 
    binary representation and returned as a numpy array of shape (timesteps x lidars)'''
    
    joints = joint_preprocessing(joint_data)    
    predicted_lidars = np.zeros((joints.shape[0], len(models)))
    
    for i in range(len(models)):
        yhat = models[i].predict(joints, verbose=0)
        binary_yhat = np.zeros(yhat.shape)
        for t in range(yhat.shape[0]): 
            amax = np.argmax(yhat[t, :])
            binary_yhat[t, amax] = 1 
        val = bin2lidar(binary_yhat, binning_params['max_value'], binning_params['bin_lenght'])
        predicted_lidars[:, i] = val
        
    return predicted_lidars 


if __name__ == "__main__":
    
    df = p.load(open('robot_random_data_large.p', 'rb'))
    orig_lidar = np.array(list(df['lidar_data'].values))
    orig_joints = np.array(list(df['joint_pos'].values))
    
    BP = {'max_value':1, 'bin_lenght':0.01 }
    EPOCHS = 1200
    
    training(orig_joints, orig_lidar, binning_params=BP, epochs=EPOCHS)
    mymodels = load_models()
    pred_lid = do_prediction(orig_joints, mymodels, binning_params=BP)
    
    true_lidar = lidar_preprocessing(orig_lidar)
    
    
    plt.figure(figsize=(12, 12))
    for k in range(pred_lid.shape[1]): 
        plt.subplot(3, 3, k+1)
        plt.plot(pred_lid[:100, k], label = 'predicted')
        plt.plot(true_lidar[:100, k], label = 'true')
        plt.legend()
        plt.title('lidar: {}'.format(k))
    plt.show()