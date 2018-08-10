from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer
from task import Task
from VAE_action import VAE_action
import numpy as np

class Agent():
    #"""Reinforcement Learning agent that learns using Q-Learning and experiance replay."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_categories = task.action_categories

        #VAE model
        self.VAE_act = VAE_action(self.state_size, self.action_size, self.action_categories)
        self.VAE_act.model.load_weights('model_weights/weights.trainedagent_2.h5')
        # Replay memory
        self.buffer_size = 10000
        self.batch_size = 80
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        self.steps = 0
        # Algorithm parameters
        self.gamma = 0.95  # discount factor

    def reset_episode(self):
        #'''reset the task and agent variables'''
        state = self.task.reset()
        self.steps = 0
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        #'''save and learn from experiences'''
        self.steps += 1
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        #"""Returns actions for given state(s) as per current policy."""
        input_state = np.reshape(state, [-1, self.state_size])
        action = np.argmax(self.VAE_act.model.predict(input_state)[0])
        return action

    def learn(self, experiences):
        #"""Update policy and value parameters using given batch of experience tuples."""
        
        for state, action, reward, next_state, done in experiences:
            # if done, make our target reward
            target = reward
            if not done:
              # predict the future discounted reward
                input_state = np.reshape(next_state, [-1, self.state_size])
                target = reward + self.gamma * np.max(self.VAE_act.model.predict(input_state)[0])
            # make the agent to approximately map
            # the current state to future discounted reward
            # save the value as target_final
            input_state = np.reshape(state, [-1, self.state_size])
            target_final = self.VAE_act.model.predict(input_state)
            target_final[0][action] = target
            # Train the Neural Net with the state and target_final
            self.VAE_act.model.fit(input_state, target_final, epochs=1, verbose=0)
