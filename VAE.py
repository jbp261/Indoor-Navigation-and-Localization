import keras
from keras import layers, models, optimizers
from keras import metrics
from keras.utils import plot_model
import pydot
#import graphviz 
from keras import backend as K 

class VAE:
    
    #"""VAE (Variational Auto Encoder) Model"""
    
    
    #"""Initialize parameters and build model.

     # Arguments:
    #    input_size (int): Dimension of each state
    #"""
    def __init__ (self, input_size):
        
        self.input_size = input_size
        self.stddev = 1.0
        self.vae_trained = False
        self.build_model()
         
    def build_model(self):
        #"""Build VAE network such that VAE model = encoder + decoder."""
        
        # Define input layer & build encoder model
  
        self.inputs = layers.Input(shape=(self.input_size, ), name='encoder_input')
        
        self.encoder_model()
        self.decoder_model()
        
        # instantiate VAE model
        outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae = models.Model(self.inputs, outputs, name='vae')
        #plot_model(self.vae, to_file='images/vae_model.png', show_shapes=True)
                
   
    def sampling(self,args):
        #"""Reparameterization by sampling from an isotropic unit Gaussian.
        # Arguments:
        #    args (tensor): mean and log of variance of Q(z|X)
        # Returns:
        #    z (tensor): sampled latent vector
        #"""
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        #random_normal: mean=0 and std=1.0
        #instead of sampling from Q(z|X), sample eps = N(0,I)
        epsilon = K.random_normal(shape=(batch, dim),stddev = self.stddev)
        # z = z_mean + sqrt(var)*eps
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def encoder_model(self):
        #"""Build an encoder model with dense layers"""
        # Add hidden layers
       
        x1 = layers.Dense(32, kernel_initializer='random_normal', activation='relu',name='dense1')(self.inputs)
        x1 = layers.Dense(64, kernel_initializer='random_normal', activation='relu',name='dense2')(x1)
        x1 = layers.Dense(64, kernel_initializer='random_normal', activation='relu',name='dense3')(x1)
        
        # Mean and log_variation layers
   
        self.z_mean = layers.Dense(2, kernel_initializer='random_normal', name='z_mean')(x1)
        self.z_log_var = layers.Dense(2, kernel_initializer='random_normal', name='z_log_var')(x1)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z = layers.Lambda(self.sampling, output_shape=(2,), name='z')([self.z_mean, self.z_log_var])

        # instantiate encoder model
        self.encoder = models.Model(self.inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
        #plot_model(self.encoder, to_file='images/vae_encoder.png', show_shapes=True)
        
    def decoder_model(self):
        #"""Build a decoder model with dense layers"""
        # build decoder model
        latent_inputs = layers.Input(shape=(2,), name='z_sampling')
        x2 = layers.Dense(64, kernel_initializer='random_normal', activation='relu',name='dense4')(latent_inputs)
        x2 = layers.Dense(64, kernel_initializer='random_normal', activation='relu',name='dense5')(x2)
        x2 = layers.Dense(32, kernel_initializer='random_normal', activation='relu',name='dense6')(x2)
        outputs = layers.Dense(self.input_size, kernel_initializer='random_normal', activation='sigmoid')(x2)

        # instantiate decoder model
        self.decoder = models.Model(latent_inputs, outputs, name='decoder')
        #plot_model(self.decoder, to_file='images/vae_decoder.png', show_shapes=True)
        
    def train_off(self):
        #"""turn all the layers of the model to not-trainable"""
        self.vae.trainable = False
   
    def loss_vae(self,y_true, y_pred):
        #"""Custom loss function for VAE model based on KL divergence principle"""
        xent_loss = keras.metrics.mse(y_true, y_pred)
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        
        loss = K.mean(xent_loss + kl_loss)
        
        return loss
        