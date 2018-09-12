import time
import numpy as np
import py_at_broker as pab
b = pab.broker()

b.register_signal('franka_target_pos', pab.MsgType.target_pos)

b.request_signal("franka_state", pab.MsgType.franka_state, True)

counter = 2000
time.sleep(0.3)

posA = np.array([0.5, 0.5, 0.3])
posB = np.array([0.3, 0.3, 0.5])

msg = b.recv_msg("franka_state", 0)

for i in range(10): 
	target_msg = pab.target_pos_msg()
	target_msg.set_fnumber(counter + i*5)
	target_msg.set_timestamp(time.CLOCK_MONOTONIC)
	target_msg.set_ctrl_t(0)
	
	if i%2 == 0: 
		pos = posA
	else: 
		pos = posB

	target_msg.set_pos(pos)
	target_msg.set_time_to_go(0.5)


	b.send_msg('franka_target_pos', target_msg)
	time.sleep(1)

