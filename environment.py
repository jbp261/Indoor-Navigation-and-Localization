import numpy as np
import pandas as pd
from pandas import DataFrame
from keras.models import load_model
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
import random
import csv



class Environment():
    #'''Approximate Environment model in which the agent operates'''
    def __init__(self, grid_file,iBeacon_loc,labeled_data, runtime=5., init_pose=None):
        
        self.init_pose = init_pose
        self.grid = np.load(grid_file)
        self.runtime = runtime
        self.b_loc = np.load(iBeacon_loc)
        #Data with 13 BLE values and the encoded columns and rows
        self.data = pd.read_csv(labeled_data)
        self.dt = 1 / 10.0  # Timestep
        
        self.lower_bounds = np.array([0,0])
        self.upper_bounds = np.array([self.grid.shape[0]-1,self.grid.shape[1]-1])
        #self.env_model = load_model('model_weights/weights.environment.h5')
        self.reset()

    def reset(self):
        #'''reset or initialize all the environment variables'''
        self.time = 0.0
        self.pose = np.array([2, 14]) if self.init_pose is None else np.copy(self.init_pose)
        self.cols = self.grid.shape[1]
        self.rows = self.grid.shape[0]
        self.distance = 0
        self.BLE_vals = self.calc_dis_BLE (self.pose)#self.calc_BLE (self.pose)
        self.done = False
        self.last_pose = np.array([2, 14]) if self.init_pose is None else np.copy(self.init_pose)
        
    def deep_inferred_BLE(self,position):
        #'''prediction of 13 iBeacon values for a given position based on deep neural network model'''
        ph1 = []
        ph1.clear()
        ph1.append(position[1])
        ph1.append(position[0])
        for j in range(0,self.b_loc.shape[0]):
            ph1.append(3 * distance.euclidean(np.array([position[1],position[0]]), self.b_loc[j]))#Column first!
        x = np.array(ph1).reshape(-1,15)
        prediction = self.env_model.predict(x)
        for i in range(0,prediction.shape[1]):
            if (prediction[0,i] < 0.25):
                prediction[0,i] = 0
            else:
                prediction[0,i] = 1 - ((x[0,i+2])/24)
            if (prediction[0,i] < 0):
                prediction[0,i] = 0
        return np.array(prediction)
    
    def inferred_BLE(self,position):
        #'''prediction of 13 iBeacon values for a given position based on mathematical model'''
        ph2 = []
        ph2.clear()
        for j in range(0,self.b_loc.shape[0]):
            ph2.append(3 * distance.euclidean(np.array([position[1],position[0]]), self.b_loc[j]))#Column first!
        array = np.array(ph2)
        array = array - array[np.argmin(array)]
        min_index = np.argmin(array)
        min_val  = array[min_index]
        array[min_index] = 1000
        s_min_index = np.argmin(array)
        s_min_val  = array[s_min_index]
        array[s_min_index] = 1000
        if s_min_val > 5.5:
            s_min_val  = 0
        t_min_index = np.argmin(array)
        t_min_val  = array[t_min_index]
        if t_min_val > 5.5:
            t_min_val  = 0
        result = np.zeros((array.shape))
        result[min_index] = (1.1/np.exp(min_val*0.1/1))-(0.03*5)
        if s_min_val > 0:
            result[s_min_index] = (1.1/np.exp(s_min_val*0.1/1))-(0.03*5)
        if t_min_val > 0:
            result[t_min_index] = (1.1/np.exp(t_min_val*0.1/1))-(0.03*5)
        return result
    
    def calc_BLE (self,position):
        #'''assign 13 iBeacon values for a given position'''
        search = self.data[(self.data['col']==position[1]) & (self.data['row']==position[0])]
        search_arr = search.values
        if search_arr.shape[0] > 0:
            rn = random.randint(0,search_arr.shape[0]-1)
            return search_arr[rn,0:13]
        else:
            return self.inferred_BLE(position)
           #return self.deep_inferred_BLE(position)
    
    def calc_dis_BLE (self,position):
        #'''calculate distance between a given position and the 13 iBeacon locations '''
        ph2 = []
        ph2.clear()
        for j in range(0,self.b_loc.shape[0]):
            ph2.append(3 * distance.euclidean(np.array([position[1],position[0]]), self.b_loc[j]))#Column first!
        return np.array(ph2)
    
    def next_timestep(self, direction):
       
       # '''
        #if direction == 0: #move east
        #    position = np.array([self.pose[0],(self.pose[1]+1)])
        #elif direction == 1: #move south-east
        #    position = np.array([self.pose[0]+1,self.pose[1]+1])    
        #elif direction == 2: #move south
        #    position = np.array([self.pose[0]+1,self.pose[1]])  
        #elif direction == 3: #move south-west
        #    position = np.array([self.pose[0]+1,self.pose[1]-1]) 
        #elif direction == 4: #move west
        #    position = np.array([self.pose[0],self.pose[1]-1]) 
        #elif direction == 5: #move north-west
        #    position = np.array([self.pose[0]-1,self.pose[1]-1])     
        #elif direction == 6: #move north
        #    position = np.array([self.pose[0]-1,self.pose[1]]) 
        #elif direction == 7: #move north-east
        #    position = np.array([self.pose[0]-1,self.pose[1]+1]) 
        #else:
        #    position = self.pose
        #'''
        #change the position based on a given action (direction)
        if direction == 0: #move east
            position = np.array([self.pose[0],(self.pose[1]+1)])   
        elif direction == 1: #move south
            position = np.array([self.pose[0]+1,self.pose[1]])
        elif direction == 2: #move west
            position = np.array([self.pose[0],self.pose[1]-1])     
        elif direction == 3: #move north
            position = np.array([self.pose[0]-1,self.pose[1]]) 
        #else:
        #    position = self.pose
        
       #check for out of bounds
        for ii in range(2):
            if position[ii] < self.lower_bounds[ii]:
                self.done = True
                self.BLE_vals = np.zeros((1,13))
            elif position[ii] >= self.upper_bounds[ii]:
                self.done = True
                self.BLE_vals = np.zeros((1,13))
        # calculate BLE RSSI values for a given position
        if not self.done:
            self.BLE_vals = self.calc_dis_BLE (position)#self.calc_BLE (position)
        
        #check to see if the position is occupied with an object
        if (self.grid[position[0],position[1]] == -10):
            self.done = True
        #check to see if the position has a datapoint     
        #if (self.grid[position[0],position[1]] == 0):
        #    self.done = True
        
        self.pose = position
        self.time += self.dt
        #increment and check the time
        if self.time > self.runtime:
            self.done = True
        
        if self.done is True:
            self.last_pose = position
            self.distace = 3 * distance.euclidean(self.last_pose, np.array([2, 14]))
        return self.done