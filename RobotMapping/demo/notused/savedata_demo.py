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
cols = ['timestamp_franka', 'joint_pos', 'joint_vel', 'endeff_pos', 'demo']
df = pd.DataFrame(columns = cols)
print(df)
####
itera=0
switch=False
breakflag=False
#Drop of code adapted from: https://pypi.org/project/pynput/



while True:
	def on_press(key):
		global switch
		switch=True



	def on_release(key):
		global itera, switch, breakflag
		switch=False
		print('{0} released'.format(key))
		itera=itera+1
		if key == keyboard.Key.esc or itera==3:
			p.dump(df, open(filename, 'wb'))
			breakflag=True
			return False


	if breakflag:
		break

	if switch:
		state_msg = b.recv_msg("franka_state", -1)
		df2 = pd.DataFrame([state_msg.get_timestamp(), state_msg.get_j_pos(), state_msg.get_j_vel(), state_msg.get_c_pos(),itera],cols)
		df = df.append(df2.T, ignore_index = True)

		print("yihaaaaaaaaaaa")
#####

filename = 'JointsDemonstration_swing.p'
print("Dump done")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
