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

def ExecuteTransform(matrix, vector):
    return np.round(np.dot(matrix , vector), 3)


def Zbounds(a):
    ztol=0.2
    minZ=0.02+ztol
    maxZ=1.08
    return (a>minZ and a<maxZ)

def Xbounds(a):
    xtol=0.2 # Extra safety for selfcrash
    minX=0.28+xtol
    maxX=0.82-xtol
    return (a>minX and a<maxX)

def Ybounds(a):
    minY=-0.78
    maxY=0.78
    return (a>minY and a<maxY)

def Bounds(a, coord):
    if coord == 'X':
        return Xbounds(a)
    elif coord == 'Y':
        return Ybounds(a)
    elif coord == 'Z':
        return Zbounds(a)
    else:
        print("ERROR")
        return 0


def IsBetweenBounds(theta, df=None):
    if df is None:
        df=init_params()

    """This algorithm checks:
    1. That the robot stump doesn't hit the table during joint space explorations.
    2.  Guarantees that the robot samples trajectories within the safe box.


    Felix bounds:
    const double vWallMin[3] = {0.28, -0.78, 0.02};
    const double vWallMax[3] = {0.82,  0.78, 1.08};"""

    df['theta'].values[0:7]=theta
    #print(df)
    Matrices=Transform2Base(df)


    standard_jointcoordinateaxis=np.array((0,0,0,1)) #with repsect to all the joints
    """The most likely part of the robot that can crash is the small stump that has a keypad and a light, which is
    part of joint six, and is measured (and always constant with respect to) coordinate axis #6. In the next two lines,
    we measure the position of the two bounds of this stump """
    keypad_jointcoordinateaxis_border1=np.array((0.10,0.09,0.05,1)) #The keypad position With respect to the sixth coordinate axis
    keypad_jointcoordinateaxis_border2=np.array((0.10,0.09,-0.05,1))
    endef_jointcoordinateaxis_center=np.array((0,0,0.13,1)) #With respect to the last coordinate axis


    criticalpoint1=ExecuteTransform(Matrices[5],standard_jointcoordinateaxis)
    criticalpoint2=ExecuteTransform(Matrices[5],keypad_jointcoordinateaxis_border1)
    criticalpoint3=ExecuteTransform(Matrices[5],keypad_jointcoordinateaxis_border2)
    criticalpoint4=ExecuteTransform(Matrices[7],endef_jointcoordinateaxis_center)



    safe=True #InitialAssumption

    for coordinate, axis in enumerate(['X', 'Y', 'Z']):
        CPs=np.array([criticalpoint1[coordinate], criticalpoint2[coordinate], criticalpoint3[coordinate], criticalpoint4[coordinate]]) # A vector of the coordinates of our points of interest
        for cp in CPs:

            if not Bounds(cp, axis):
                safe =False
                break
        if safe==False:
            break
    #if safe==True:
        #print(safe, theta, criticalpoint4)
    return safe



import SLRobot
from scipy.interpolate import interp1d
from scipy import interpolate


pgain_null = 0.002 * np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], dtype=np.float64)


# Null-space theta configuration

target_th_null = np.zeros(7, dtype=np.float64)
target_th_null[3] = -1.55
target_th_null[5] = 1.9


def quatToEulerAngles(quat):

    eulerAngles = np.array([0.0]*3)
    threshold = 0.001
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]


    heading = np.arctan2(2.0*qy*qw-2.0*qx*qz , 1.0 - 2.0*qy*qy - 2.0*qz*qz)
    attitude = np.arcsin(2.0*qx*qy + 2.0*qz*qw)
    bank = np.arctan2(2.0*qx*qw-2.0*qy*qz , 1.0 - 2.0*qx*qx - 2.0*qz*qz)



    if (qx*qy + qz*qw - 0.5)*(qx*qy + qz*qw - 0.5) < threshold:
        heading = 2.0 * np.arctan2(qx,qw)
        bank = 0


    if (qx*qy + qz*qw + 0.5)*(qx*qy + qz*qw + 0.5) < threshold:
        heading = -2.0 * np.arctan2(qx,qw)
        bank = 0



    eulerAngles[0] = attitude
    eulerAngles[1] = heading
    eulerAngles[2] = bank



    return(eulerAngles)





def inverse_kin_6d(c_pos,q):

    J_reg = 1e-8
    # initialize the position

    J_w = np.diag([50, 50, 25, 10, 10, 5, 1])



    # compute the velocity in joint space and update the joint position
    qi = []
    EF_=[]
    qi.append(q)


    results_ =[]
    c_vel_all = []

    #while 1:
    plt.figure()

    for t in np.arange(100000):


        EF_FK = SLRobot.FK(np.array([q]))
        #print(EF_FK[0,50:56])
        c_vel = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.01])*(c_pos - EF_FK[0,50:56])

        #print(c_vel)
        J = SLRobot.Jacobian(np.array([q]), 6)
        target_th_null_out  = target_th_null
        A = J.dot(J_w).dot(J.transpose())
        A += J_reg * np.eye(6)
        qd_null = pgain_null * (target_th_null_out - q)



        qd_d = np.linalg.solve(A, c_vel - J.dot(qd_null))
        qd_d = J_w.dot(J.transpose()).dot(qd_d) + qd_null



        q = q + 0.002*qd_d




        #set the joint range

        #mask_max = q > np.pi
        #mask_min = q<-np.pi
        #q[mask_max] = q[mask_max] - 2*np.pi
        #q[mask_min] = q[mask_min] + 2*np.pi





        qi.append(q)
        EF_.append(EF_FK[0,50:56])
        c_vel_all.append(c_vel)


        # times the arrary to scale the orientation range.

        error_ = np.sqrt(np.sum(np.square(np.array([1.0, 1.0, 1.0, 0.08, 0.08, 0.08])*(c_pos-EF_FK[0,50:56]))))
        #print(error_)
        if error_ <0.020:

            break



    print('error:', error_)


    qi = np.array(qi)
    plt.plot(qi)

    EF_ = np.array(EF_)

    c_vel_all = np.array(c_vel_all)

    return qi[-1], EF_, qi, c_vel_all
