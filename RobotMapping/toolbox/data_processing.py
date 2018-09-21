import pandas as pd
import pickle as p 



def write2file(broker, filename = 'testdata.p'):

    #filename = 'testdata.p'
    cols = ['timestamp_lidar', 'lidar_data', 'timestamp_franka', 'joint_pos', 'joint_vel', 'endeff_pos']
    df = pd.DataFrame(columns = cols)
    state_msg = broker.recv_msg("franka_state", 0)  
    ref = state_msg.get_fnumber()
    while True: 
        lidar_msg = broker.recv_msg("franka_lidar", 5) #-1 waits for the lidar, then...
        state_msg = broker.recv_msg("franka_state", 0)        # then we get a signal from the state as fast as possible (tested)
        if state_msg.get_fnumber() == ref: 
            break
        df2 = pd.DataFrame([lidar_msg.get_timestamp(), lidar_msg.get_data(), state_msg.get_timestamp(), 
                            state_msg.get_j_pos(), state_msg.get_j_vel(), state_msg.get_c_pos()],cols)

        df = df.append(df2.T, ignore_index = True)
        ref = state_msg.get_fnumber()
    p.dump(df, open(filename, 'wb'))
    print('simulation stopped, wrote data to {}'.format(filename))
