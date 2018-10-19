import numpy as np
import py_at_broker as pab
import time

##############################################################################
# This script shows how to use the gravity compensation of the robot while
# guiding it by hand to collect demonstrations
##############################################################################
if __name__ == "__main__":
    ###########################################################################
    # Script Settings
    robot_name = "franka"  # for use in signals
    max_sync_jitter = 0.2
    
    ###########################################################################
    # Set up communication
    b = pab.broker()
    # Register target position signal
    rb = b.register_signal(robot_name + "_des_tau", pab.MsgType.des_tau)
    if not rb:
        print(
            "Could not register signal <{}_des_tau>\n Maybe try"
            " restarting the address broker".format(robot_name))
    # Request robot signal in a blocking fashion
    b.request_signal(robot_name + "_state", pab.MsgType.franka_state, True)

    counter = 0
    time.sleep(1)  # Allow some time for the communication setup

    ###########################################################################
    # Executive loop
    while True:
        # Receive franka state
        recv_start = time.clock_gettime(time.CLOCK_MONOTONIC)
        msg_panda = b.recv_msg(robot_name + "_state", -1)
        recv_stop = time.clock_gettime(time.CLOCK_MONOTONIC)

        # Check that the messages we're getting are not too old, reset the
        # entire script if we get out of sync
        if (2 * recv_stop - recv_start - msg_panda.get_timestamp()) > max_sync_jitter:
            print("De-synced, messages too old\n Resetting...")

        # Create and fill message
        msg = pab.des_tau_msg()
        msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
        msg.set_fnumber(counter)
        msg.set_j_torque_des(np.zeros(7))
        # Send message
        b.send_msg(robot_name + "_des_tau", msg)

        if counter % 500 == 0:
            print(msg_panda.get_j_pos())
        # Increment counter
        counter += 1
