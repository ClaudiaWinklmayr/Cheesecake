import numpy as np
import py_at_broker as pab

import time

def create_spiral(t, a): 
    x = a*(np.cos(t) + t*np.sin(t))
    y = a*(np.sin(t) - t*np.cos(t))
    return x, y


def create_message(counter=1000, timestamp=time.CLOCK_MONOTONIC, \
                   ctrl_t=0, pos = np.array([0.5, 0.7, 0.7]), go_time=0.15):
    msg=pab.target_pos_msg()
    msg.set_fnumber(counter)
    msg.set_timestamp(timestamp)
    msg.set_ctrl_t(ctrl_t)
    msg.set_pos(pos)
    msg.set_time_to_go(go_time)
    current_counter=counter
    return msg, current_counter


