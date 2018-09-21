import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# this is a toolbox for transforming joint_states of the Fraka robot
# to cartesian coordinats using hartenberg-denavit transformation


def init_params():
    A=np.array([ 0,0,0,0.0825,-0.0825,0,0.088,0])
    D=np.array([ 0.333,0,0.316,0,0.384,0,0,0.107])
    Alph=(np.pi/2)*(np.array([0,-1,1,1,-1,1,1,0]))
    Thet=np.zeros(8)
    d = {'a':A, 'd':D,'alpha':Alph, 'theta': Thet}
    df = pd.DataFrame(d)

    return df


def Rx(alpha):
    ''' rotations around x axis'''
    transform=np.eye(4)

    transform[1,1]=np.cos(alpha)
    transform[2,2]=np.cos(alpha)
    transform[2,1]=np.sin(alpha)
    transform[1,2]=-np.sin(alpha)

    return transform


def Rz(theta):
    ''' rotations around z axis'''
    transform=np.eye(4)

    transform[0,0]=np.cos(theta)
    transform[1,1]=np.cos(theta)
    transform[1,0]=np.sin(theta)
    transform[0,1]=-np.sin(theta)

    return transform


def Transl(x,y, z):
    ''' shift in xyz-space'''
    transform=np.eye(4)

    transform[0,3]=x
    transform[1,3]=y
    transform[2,3]=z

    return transform


def HDtransfromation(params):
    ''' Complete transformation: (i) shift in z-direction (ii) rotation around z-axis (iii) shift in x-direction
	(iv) rotation around x-axis '''

    d = params['d']
    theta = params['theta']
    a = params['a']
    alpha = params['alpha']

    return np.dot(Rx(alpha), np.dot(Transl(a, 0, 0), np.dot(Rz(theta), Transl(0,0,d))))


def Transform2Base(df):
    ''' return list of seven matrices, each of which is the transformation of the i-th joint's coordinate system
	back to the base'''

    single_trans_list = [HDtransfromation(dict(df.loc[i])) for i in range(len(df))] # transformmation matrices from i-th to (i-1)-th coordinate system

    total = []
    total.append(single_trans_list[0])
    for i in range(1, len(single_trans_list)):
        total.append(np.dot(total[i-1], single_trans_list[i]))
    return total
