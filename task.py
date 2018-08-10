import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from environment import Environment

class Task():
    #"""Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, grid_file,iBeacon_loc,labeled_data,runtime=5., target_pos=None, init_pose=None):
        #"""Initialize a Task object.
         # Arguments:
        #    init_pose: initial position of the user in (x,y) dimensions 
        #    runtime: time limit for each episode
        #    target_pos: target/goal (x,y) position for the agent
        #"""
        # Simulation
        self.sim = Environment(grid_file,iBeacon_loc,labeled_data, runtime, init_pose) 
        self.action_repeat = 1
        #"""
        #States: The state of the agent is represented as a tuple of these observations.
        #1) A vector of 13 RSSI values.
        #2) Current location (identified by row and column numbers).
        #3) Distance to the target.
        #"""
        self.state_size = self.action_repeat * 16
        self.action_size = 1
        self.action_categories = 4
        #Statistics data variables
        self.prev_dis = 0
        self.total_dis = 0
        self.positions = []
        self.best_pos = []
        self.best_score = -np.inf
        self.score = 0
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([17, 10])
        self.init_dis = self.calc_distance(self.sim.pose, self.target_pos)
        self.dis_to_target = self.calc_distance(self.sim.pose, self.target_pos)
        
    def calc_distance(self,a,b):
        return 3 * distance.euclidean(a,b)
    
    def get_reward(self, done):
        #"""Uses current pose of sim to return reward."""
        reward = 0
        distance = self.calc_distance(self.sim.pose, self.target_pos)
        self.total_dis += abs(self.prev_dis - distance)
   
        #positive reward for getting close to the target and neagtive for getting far
        if self.prev_dis > distance:
            reward += (self.prev_dis - distance)
        if (distance < 12 and distance != 0): #reward for being close to target
            reward += (10)
        elif distance <= 4: # reward for getting to the target
            reward += 20
        #elif distance is not 0:
        #    reward += 1/distance
        else:
            reward -= 1 #penalty for hovering away from the target
            
        #penalty for being done without reaching target
        if done is True:
            if not np.array_equal(self.sim.pose,self.target_pos):
                reward -= 0
        
        self.prev_dis = distance
        return reward
  
    
    def step(self, direction):
        #"""Uses action to obtain next state, reward, done."""
        reward = 0
        list_ = []
        
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(direction)
            self.dis_to_target = self.calc_distance(self.sim.pose, self.target_pos)
            if np.array_equal(self.sim.pose,self.target_pos):
                done = True
            reward += self.get_reward(done)
            self.score += reward
            self.positions.append(self.sim.pose)
            y1 = self.sim.BLE_vals.reshape(-1,)
            for i in range(0,13):
                list_.append(y1[i])
            y2 = self.sim.pose.reshape(-1,)
            for i in range(0,2):
                list_.append(y2[i])
                
            list_.append((3 * distance.euclidean(self.sim.pose,self.target_pos)))
            
        next_state =np.array(list_)
        list_.clear()
        
        if done is True:
            self.update_positions(self.score)
        return next_state, reward, done

    def reset(self):
        #"""Reset the sim to start a new episode."""
        self.sim.reset()
        self.prev_dis = 0
        self.total_dis = 0
        self.score = 0
        self.dis_to_target = self.calc_distance(self.sim.pose, self.target_pos)
        list_ = []
        y1 = self.sim.BLE_vals.reshape(-1,)
        for i in range(0,13):
            list_.append(y1[i])
        y2 = self.sim.pose.reshape(-1,)
        for i in range(0,2):
            list_.append(y2[i])    

        list_.append((3 * distance.euclidean(self.sim.pose,self.target_pos)))

        state = np.concatenate([np.array(list_)]* self.action_repeat)
        list_.clear()
        return state
    
    def update_positions(self,reward):
        #"""saves the best path found in terms of rewards"""
        if reward > self.best_score:
            self.best_pos.clear()
            self.best_pos = self.positions
            self.positions.clear()
            self.best_score = reward
        
        
        
        
        