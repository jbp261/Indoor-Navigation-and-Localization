import keras
from keras import layers, models, optimizers
from keras import metrics
from keras.utils import plot_model
from keras import backend as K 

class VAE:
    
    """VAE Model."""
    """Initialize parameters and build model.

    Params
    ======
        input_size (int): Dimension of each state
        action_size (int): Dimension of each action
    """
    def __init__ (self, input_size, output_size):
        
        self.input_size = input_size
        self.action_size = output_size
        self.stddev = 1.0
        self.vae_trained = False
        self.build_model()
         
    def build_model(self):
        """Build VAE network such that VAE model = encoder + decoder."""
        
        # Define input layer & build encoder model
  
        self.inputs = layers.Input(shape=(self.input_size, ), name='encoder_input')
        
        self.encoder_model()
        self.decoder_model()
        
        # instantiate VAE model
        outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae = models.Model(self.inputs, outputs, name='vae')
        #plot_model(vae, to_file='vae_model.png', show_shapes=True)
        
        self.action_dense = layers.Dense(1024, activation='relu', name = 'action_dense')(outputs)
        
        actions = layers.Dense(self.action_size, activation='sigmoid', name='actions')(self.action_dense)
        
        self.model = models.Model(self.inputs, actions, name='model')
        #plot_model(model, to_file='vae_action_model.png', show_shapes=True)
        
        #custom loss function
        state_gradients = layers.Input(shape=(self.input_size,))
        z_decoded = self.decoder(self.z)
        # Reconstruction loss
        xent_loss = keras.metrics.binary_crossentropy(state_gradients, z_decoded)
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        loss_vae = K.mean(xent_loss + kl_loss)
        
        # Define optimizer and training functions
       
        #The optimizer and training function for the VAE model
        self.vae.compile(loss=loss_vae,optimizer='adam')
        #updates_op_vae = optimizer.get_updates(params=self.vae.trainable_weights, loss=loss_vae)       
            
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])
        
        
    # reparameterization 
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def sampling(self,args):
        """Reparameterization by sampling from an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim),stddev = self.stddev)
        
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def encoder_model(self):
        # Add hidden layers
       
        x1 = layers.Dense(256, activation='relu')(self.inputs)
        x1 = layers.Dense(512, activation='relu')(x1)
        x1 = layers.Dense(256, activation='relu')(x1)
        
        # Mean and log_variation layers
   
        self.z_mean = layers.Dense(2, name='z_mean')(x1)
        self.z_log_var = layers.Dense(2, name='z_log_var')(x1)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z = layers.Lambda(self.sampling, output_shape=(2,), name='z')([self.z_mean, self.z_log_var])

        # instantiate encoder model
        self.encoder = models.Model(self.inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
        #plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)
        
    def decoder_model(self):
        # build decoder model
        latent_inputs = layers.Input(shape=(2,), name='z_sampling')
        x2 = layers.Dense(256, activation='relu')(latent_inputs)
        outputs = layers.Dense(self.input_size, activation='sigmoid')(x2)

        # instantiate decoder model
        self.decoder = models.Model(latent_inputs, outputs, name='decoder')
        #plot_model(decoder, to_file='vae_decoder.png', show_shapes=True)
        
        
    def tain_off(self):
        if (self.vae_trained is True):
            self.vae.trainable = False