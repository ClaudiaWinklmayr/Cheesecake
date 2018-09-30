# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:08:33 2018

@author: claud
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from collections import Counter
import time
import pickle as p
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def depth_image_preprocessing(images, res = 0.01): 
    '''Inputs: 
    images........... a np.array conataining depth measurements of size samples x pixels
    
    Takes an array of depth measurements (usually between 0 and 8000 mm), 
    normalizes them to lie between 0 and 1 and creates a probability distribution od the 
    normalized values by sorting them to bins of withd 0.01. The resulting distribution 
    is normalized to sum to 100.'''
    
    Nimgs = images.shape[0]
    bins = np.arange(0, 1+2*res, res)
    processed_imgs = np.zeros((Nimgs, len(bins)))
    
    for idx in range(Nimgs): 
        img_array = images[idx, :]
        processed_img = img_array
        processed_img = processed_img/np.max(processed_img)        
        ranges = np.digitize(processed_img, bins)
        count = Counter(ranges)
        img_distr = np.zeros(len(bins))
        img_distr[np.array(list(count.keys()))] = np.array(list(count.values()))
        img_distr = (img_distr/np.sum(img_distr))*100
        processed_imgs[idx, :] = img_distr
        
    return processed_imgs


def joint_preprocessing(joint_data, normalize = True): 
    '''Inputs: 
    joint_data......... raw joint angle measurements usually between -pi and pi
    normalize.......... if True the data is normalized to lie between 0 and 1 
    
    Normalizes joint angle measurements (in rad) to values between 0 and 1 by 
    first shifting by pi and then dividing by 2pi'''
    
    joints = np.zeros(joint_data.shape)
    for i in range(joints.shape[1]):
        a = joint_data[:, i].astype(float)
        a += np.pi
        if normalize:             
            #a = a/np.max(a)
            a = a/(2*np.pi) # divide my max possible value
        joints[:, i] = a
    return joints



def build_depth_model(inp_shape, outp_shape):
    ''' Creates a feed formward network for classifying the distribution of a depth
    image from normalized joint data. '''
    
    #working with: 3 dense layers with size outpshape & activation relu, 
    # optimizer = sgd, lr 0.1, momentum = 0, decay = 0 (or very very small)
    # loss lms, init weights randomly, train for at least 5000 epochs
    model = keras.Sequential()
    model.add(keras.layers.Dense(outp_shape, activation=tf.nn.relu, input_shape=(inp_shape,)))
    model.add(keras.layers.Dense(outp_shape, activation=tf.nn.relu))
    model.add(keras.layers.Dense(outp_shape, activation=tf.nn.relu))
    # TODO: needs a normalization layer
    optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0)#0.000001, nesterov=False)
    #loss = keras.losses.categorical_crossentropy
    loss = keras.losses.mean_squared_error
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
   
    a = [wm.shape for wm in model.get_weights()]
    rand_weights = [np.random.normal(loc = 0, scale = 0.05, size = aa) for aa in a]
    model.set_weights(rand_weights)
    return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
        
def find_support_overlap(array1, array2, thresh): 
    ''' this function takes the  distribution of depth values calclulated from a depth image and 
    compares it to the density predicted by the network. To this end we identify the bins which hold 
    a given perectage (normally 95%) of the distributions enegery and calculate the overlap 
    between these two lists of inidces'''
    
    sorted_array1 = np.flip(np.sort(array1), axis = 0)   
    indices_sorted_array1 = np.flip(np.argsort(array1), axis = 0)    
    max_support_idx_array1 = np.where(np.cumsum(sorted_array1)>thresh)[0][0]
    support_idxs_array1 = set(indices_sorted_array1[:max_support_idx_array1])

    sorted_array2 = np.flip(np.sort(array2), axis = 0)
    indices_sorted_array2 = np.flip(np.argsort(array2), axis = 0)    
    max_support_idx_array2 = np.where(np.cumsum(sorted_array2)>thresh)[0][0]
    support_idxs_array2 = set(indices_sorted_array2[:max_support_idx_array2])
 
    #overlap = list(support_idxs_array1.intersection(support_idxs_array2))symmetric_difference
    non_obverlap = list(support_idxs_array1.symmetric_difference(support_idxs_array2))
    return float(len(non_obverlap))/len(list(sorted_array1)), non_obverlap


class depth_predictor(): 
    
    def __init__(self, model_name='depth_predictor.h5', train=False, training_data={}):
        
        if train == False: 
            self.model = tf.keras.models.load_model(model_name)
        else: 
            joints_data = training_data['joints']
            joints_training = joint_preprocessing(joints_data)
            image_data = training_data['depth_images']
            images_training = depth_image_preprocessing(image_data)
            
            model = build_depth_model(joints_training.shape[1], images_training.shape[1])
            print('Training the depth predictor')
            history = model.fit(joints_training, images_training, epochs=training_data['epochs'], validation_split=0.2, verbose=0, callbacks=[PrintDot()])
            plt.plot(history.history['loss'])
            model.save(model_name)
            self.model = tf.keras.models.load_model(model_name)

    def predict_timeseries(self, joints_raw, depth_img_raw): 
        joints = joint_preprocessing(joints_raw)
        depth_img = depth_image_preprocessing(depth_img_raw)
        prediction = self.model.predict(joints)   
        
        plt.plot(prediction.T, label = 'predict')
        plt.plot(depth_img.T, label = 'true')
        plt.legend()
        plt.show()
                
    def predict(self, joints_raw, depth_img_raw): 
        joints = joint_preprocessing(np.reshape(joints_raw, (len(joints_raw), 1)))
        depth_img = depth_image_preprocessing(np.reshape(depth_img_raw, (1, len(depth_img_raw))))
        prediction = self.model.predict(np.reshape(joints, (1, len(joints))))   
        prediction = 100*(prediction/np.sum(prediction))
        
        percent, indices = find_support_overlap(np.reshape(prediction, (prediction.shape[1])), 
                            np.reshape(depth_img, (depth_img.shape[1])), 95)

        plt.plot(prediction.T, label = 'predict')
        plt.plot(depth_img.T, label = 'true')
        plt.scatter(indices, np.zeros(len(indices)), c = 'k', s = 50)
        plt.title('{}% non - overlap'.format(np.round(percent, 2)))
        plt.legend()
        plt.show()
        



if __name__ == "__main__":
    from IPython.display import clear_output

    df = p.load(open('00_robot_random_joint_depth_longrecording_1.p', 'rb'))
    unseen_joints = np.array(list(df['joint_pos'].values))
    unseen_images = np.array(list(df['depth_img'].values))
    #predictor = depth_predictor(train=True, training_data={'joints': unseen_joints, 'depth_images': unseen_images, 'epochs':9000})
    predictor = depth_predictor()
    for idx in range(len(unseen_joints)):
        print(idx)
        predictor.predict(unseen_joints[idx, :], unseen_images[idx, :])
        time.sleep(2)
        clear_output() 