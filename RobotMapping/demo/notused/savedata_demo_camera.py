#Broker imports:
"""These scripts can only be run in the workstation """
import py_at_broker as pab


import pylab as pl
import pickle as p
import matplotlib.pyplot as plt
b = pab.broker()

from toolbox import data_processing as dp
from toolbox import movementtoolbox as mt



#Non broker imports:
"""These scripts can only be run elsewhere """
from pynput import keyboard
import math as math
import pandas as pd
import time
import numpy as np
from toolbox import kinematic_mapping as km

##

b.request_signal("franka_state", pab.MsgType.franka_state, True)
b.request_signal("realsense_images", pab.MsgType.realsense_image)
#b.request_signal("franka_lidar", pab.MsgType.franka_lidar)

cols = ['timestamp_franka', 'joint_pos', 'joint_vel', 'endeff_pos', 'rgb_img', 'depth_img', 'demo']
df = pd.DataFrame(columns = cols)
print(df)
####
iter=0

#Drop of code adapted from: https://pypi.org/project/pynput/
def on_press(key):

    global iter, df

    state_msg = b.recv_msg("franka_state", 0)
    #lidar_msg = b.recv_msg("franka_lidar", 0) #-1 waits for the lidar, then...
    camera_msg = b.recv_msg("realsense_images", 0)

    df2 = pd.DataFrame([state_msg.get_timestamp(), state_msg.get_j_pos(), state_msg.get_j_vel(),
    state_msg.get_c_pos(),camera_msg.get_rgb(), camera_msg.get_depth(),iter],cols)
    df = df.append(df2.T, ignore_index = True)
    #print(state_msg.get_j_pos())
    print(iter)

def on_release(key):
    global iter
    print('{0} released'.format(key))
    iter=iter+1
    if key == keyboard.Key.esc or iter==4:
        p.dump(df, open(filename, 'wb'))
        switch=True
        return False



#####

filename = 'JointsDemonstration_cameraClaudia1.p'

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()