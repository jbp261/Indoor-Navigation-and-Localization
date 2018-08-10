from keras import layers, models, optimizers
from keras.utils import plot_model
from keras import backend as K

class QNetwork:
    """QNetwork Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        """Build a Q network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=128, activation='relu')(states)
        net_states = layers.Dense(units=128, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=128, activation='relu')(actions)
        net_actions = layers.Dense(units=128, activation='relu')(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        #plot_model(QNetwork, to_file='QNetwork.png', show_shapes=True)

        # Define optimizer and compile model for training with built-in loss function
        self.model.compile(optimizer='adam', loss='mse')