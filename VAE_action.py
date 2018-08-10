import keras
from keras import layers, models, optimizers
from keras import metrics
from keras.utils import plot_model
from keras import backend as K
import pydot
#import graphviz 
from VAE import VAE

class VAE_action:
    #"""VAE Action Model to direct the agent for an appropriate action given input"""

    def __init__(self, state_size, action_size, action_categories):
        #"""Initialize parameters and build model.

        #Argumets:
        #    state_size (int): Dimension of each state
        #    action_size (int): Dimension of each action
        #"""
        self.state_size = state_size
        self.action_size = action_size
        self.action_categories = action_categories
        self.vae_model = VAE(self.state_size)
        self.lr = 0.001
        #vae_model.vae.load_weights('model_weights/weights.vae.h5')
        #vae_model.vae.trainable = False
        #vae_model.encoder.trainable = False
        #vae_model.decoder.trainable = False
        self.build_model()

    def build_model(self):
        #"""Build VAE network such that VAE model = encoder + decoder."""
        #l = self.vae_model.vae.output
        #input layer
        inputs = layers.Input(shape=(self.state_size, ), name='action_input')
        #dense layers
        l = layers.Dense(64, kernel_initializer='random_normal', activation="relu")(inputs)
        l = layers.Dense(128, kernel_initializer='random_normal', activation="relu")(l)
        l = layers.Dense(128, kernel_initializer='random_normal', activation="relu")(l)
        #output layer
        actions = layers.Dense(self.action_categories, kernel_initializer='random_normal', activation='linear', name='actions')(l)
        
        #model creation
        self.model = models.Model(inputs, actions, name='action_model')
        plot_model(self.model, to_file='images/vae_action_model.png', show_shapes=True)
        
        #model compilation
        self.model.compile(loss='mse',optimizer=optimizers.Adam(lr=self.lr))
        