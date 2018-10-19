import numpy as np
from toolbox import kinematic_mapping as km
from toolbox import lidar_predictor as lp
import time
import pandas as pd


def init_lidar_predictor(broker, timesteps = 5, model_prefix = ''): 
    predictor = lp.lidar_predictor(binning_params = {'max_value':1, 'bin_lenght':0.01 }, 
                   model_names=[model_prefix+'_{}.h5'.format(i) for i in range(9)])
    
    measurements = []
    predictions = []
    
    print('initializing measurements: ', end ='')
    for t in range(timesteps): 
        lidar_msg = broker.recv_msg("franka_lidar", 0)
        state_msg = broker.recv_msg("franka_state", 0)  
        prediction, measurement = predictor.predict_single(joint_data=state_msg.get_j_pos(), true_lidar=lidar_msg.get_data())
        predictions.append(prediction)
        measurements.append(measurement)
        time.sleep(0.5)
        print('.', end = '')
    print('done')
    return predictor, predictions, measurements    

def calc_error(predictor, predictions, measurements, new_lidar, new_joints, threshold): 
    #prediction, measurement = predictor.predict_single(joint_data=state_msg.get_j_pos(), true_lidar=lidar_msg.get_data())
    prediction, measurement = predictor.predict_single(joint_data=new_lidar, true_lidar=new_lidar)
    
    predictions.append(prediction)
    predictions.pop(0)
    
    measurements.append(measurement)
    measurements.pop(0)
    
    error = compare2threshold(np.array(predictions), np.array(measurements), threshold)
    return error, predictions, measurements

def compare2threshold(prediction, measurement, threshold, all_errors=False): 
    # dim prediction = timestamps x lidars
    
    #prediction = np.reshape(prediction, (prediction.shape[0], prediction.shape[2]))
    #measurement = np.reshape(measurement, (measurement.shape[0], measurement.shape[1]))
    error = prediction-measurement # positive error means something is closer than expected
    measurement_variance = np.var(measurement, axis = 0)
    scaling = np.zeros(error.shape)
    scaling[np.where(prediction >=0.5)] = 1
    scaling[np.where(prediction >=1)] = 2   
    scaling[np.where(prediction <0.5)] = 0.5
    #error = np.divide(error, prediction)
    error = np.divide(error, scaling)
    med_error = np.median(error, axis = 0) # calculate median over timestamps will ignore negantive and small values 
    med_error[np.where(measurement_variance>0.8)] = 0

    
    if (med_error>threshold).any():
        if all_errors: 
            idxs = np.where(med_error>threshold)[0]
            for i in idxs : 
                print('Deviation of {} in liadar {}.'.format(np.round(med_error[i], 2), i))
            return True, idxs            
        else: 
            print('Deviation of {} in liadar {}.'.format(np.round(np.max(med_error), 2), np.argmax(med_error)))
            #print(med_error)
            return True, np.argmax(med_error)
    else: 
        return False, None

def check_approaching(measurements, index): 
    diff = np.median(np.diff(np.array(measurements)[:, index]))
    if diff < 0: 
        print('approaching')

class grid():       
    def __init__(self, x_min=0, x_max=1, y_min=-0.5, y_max=0.5, z_min=0, z_max=15, grid_width=0.2):
        self.xs = np.arange(x_min, x_max+grid_width, grid_width)
        self.ys = np.arange(y_min, y_max+grid_width, grid_width)
        self.zs = np.arange(z_min, z_max+grid_width, grid_width)        
        
    def is_in_grid(self, point): 
        idxs = np.zeros(3)
        
        if (point[0]<self.xs[0]) or (point[0]>self.xs[-1]): 
            return (False, None)
        else: 
            idxs[0] = np.argwhere(point[0]<self.xs)[0][0]
            
        if (point[1]<self.ys[0]) or (point[1]>self.ys[-1]): 
            return (False, None)
        else: idxs[1] = np.argwhere(point[1]<self.ys)[0][0]
        
        if (point[2]<self.zs[0]) or (point[2]>self.zs[-1]): 
            return (False, None)
        else: 
            idxs[2] = np.argwhere(point[2]<self.zs)[0][0]
            
        return (True, idxs.astype(int))

class localizer(): 
    def __init__(self): 
        self.angle_df = km.init_params()
        
        self.lidarDF = self.init_lidar_constants()
        #joint the lidar is attached to
        self.lidar_joint = {str(self.lidarDF['lidar_id'][i]):self.lidarDF['joint_id'][i] for i in range(len(self.lidarDF))}
        
        self.lidar_base = {str(self.lidarDF['lidar_id'][i]): 
                           0.01*np.array([float(self.lidarDF['lidar_x'][i]), self.lidarDF['lidar_y'][i], self.lidarDF['lidar_z'][i]])
                           for i in range(len(self.lidarDF))}
        
        self.lidar_direction = {str(self.lidarDF['lidar_id'][i]): 
                           np.array([float(self.lidarDF['point_x'][i]), self.lidarDF['point_y'][i], self.lidarDF['point_z'][i]])
                           for i in range(len(self.lidarDF))}
        self.world_grid = grid()
    
    def init_lidar_constants(self): 
        lidar_id = [0,1,2,3,4,5,6,7,8]
        joint_id = [5,5,5,5,5,4,3,3,4]
        lidar_x = [0,-7,0,7,0,0,8,14,0]
        lidar_y = [-8,5,4,5,12,0,5,5,0]
        lidar_z = [0,0,6,0,0,12,6,-1,-12]
        point_x = [0,-1,0,1,0,0,0,0,0]
        point_y = [-1,0,0,0,1,0,0,1,0]
        point_z = [0,0,1,0,0,1,1,0,-2]

        d = {'lidar_id':lidar_id, 'joint_id':joint_id, 'lidar_x':lidar_x, 'lidar_y':lidar_y, 
         'lidar_z':lidar_z, 'point_x':point_x, 'point_y':point_y, 'point_z':point_z}

        return pd.DataFrame(d)       
    
    def create_transformation(self, joint_angles): 
        self.angle_df['theta'].values[0:7] = joint_angles
        TF_Base = km.Transform2Base(self.angle_df)
        return TF_Base
        
    def localize_joints(self, joint_angles):         
        TF_Base = self.create_transformation(joint_angles)
        joint_base = np.array([0,0,0,1])
        joints = [km.ExecuteTransform(TF_Base[i], joint_base) for i in range(len(TF_Base))]
        return np.array(joints)[:,:3]
    
    def localize_lidars(self, joint_angles): 
        TF_Base = self.create_transformation(joint_angles)   
        lidar_poses = []
        for lid in range(9): 
            joint = self.lidar_joint[str(lid)]
            lidar_tmp = np.ones(4)
            lidar_tmp[:3] = self.lidar_base[str(lid)]
            lidar_poses.append(km.ExecuteTransform(TF_Base[joint-1], lidar_tmp))
        return np.array(lidar_poses)[:,:3]

    
    def localize_obstacle(self, joint_pos, lidar_measurement, lidar_index):
        measurement = lidar_measurement[lidar_index]
        distance = np.min([measurement, 2000])/1000. # clip and transform to meters 
        location_joint = self.lidar_base[str(lidar_index)] + distance*self.lidar_direction[str(lidar_index)]
        location_joint_tmp = np.ones(4)
        location_joint_tmp[:3] = location_joint
        TF_Base = self.create_transformation(joint_pos)
        
        location = km.ExecuteTransform(TF_Base[self.lidar_joint[str(lidar_index)]-1], location_joint_tmp)[:3]
        self.identify_area(location)
        return location

    
    def identify_area(self, location):         
        x_dir = ['back', 'front']
        y_dir = ['right','left']
        z_dir = ['bottom', 'top']        
        comparison = np.array([0, 0, 0.4])
        select = (location>comparison).astype(int)       
        print('obstacle found in {} {} {}'.format(x_dir[select[0]], y_dir[select[1]], z_dir[select[2]]))

        
    def predict_collision(self, obstacle_location, future_joint_angels): 
        #future_joint_angels = joints x timestamps
        T = future_joint_angels.shape[1]  #number of future timestamps
        J = future_joint_angels.shape[0] 
        obstacle_in_grid = self.world_grid.is_in_grid(obstacle_location)        
        if obstacle_in_grid[0] == False: 
            print('Obstacle outside world')
            return (False, None)       
        else: 
            for t in range(T): 
                #print(obstacle_in_grid[1])
                joint_poses = self.localize_joints(future_joint_angels[:, t])               
                #print(joint_poses)#
                grid_locs = [self.world_grid.is_in_grid(joint_poses[j, :])[1] for j in range(J)]
                #print(grid_locs)
                collision = np.array([(grid_loc == obstacle_in_grid[1]).all() for grid_loc in grid_locs])
                if collision.any(): 
                    print('possible collision in {} steps'.format(t))
                    return (True, t)  
            return (False, None)
        
        
