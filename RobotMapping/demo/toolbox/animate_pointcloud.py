import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import time
import numpy as np

# This script reads the robots joint positions from a txt file and 
# anaimates the resulting scatter plot. 

# script should be run in parallel with a simulating file which constantly 
# creates new joint positions


fig = plt.figure()
ax1 = p3.Axes3D(fig)


def read_data_array(filename, typ = 'joint'): 
    ''' reads through a file of joint positions and returns the final one as an array of shape (joints x 4) '''

    f = open(filename, 'r')
    a = None
    for line in f:
        s = line.split(',')
        a = np.array([float(ss) for ss in s[:-1]])
        if typ == 'joint': 
            a = a.reshape((8, 4))
        elif typ == 'lidar':
            a = a.reshape((9, 4))
    return a 



def animate(i):
    joint_file = "joint_positions.txt"
    lidar_file = "lidar_positions.txt"
    
    jp = read_data_array(joint_file, typ = 'joint')
    lp = read_data_array(lidar_file, typ = 'lidar')
    lp = lp
    ax1.clear()
    ax1.scatter(jp[:,0], jp[:,1], jp[:,2], s = 50, c = 'b')    
    ax1.plot(jp[:,0], jp[:,1], jp[:,2], color = 'b')
    ax1.scatter(lp[:,0], lp[:,1], lp[:,2], s = 50, c = 'r')    
    
    ax1.set_xlim(0,1.1)
    ax1.set_ylim(0,1.1)
    ax1.set_zlim(0,1.1)


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()


