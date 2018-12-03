#import tensorflow as tf
from collections import deque
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, action_size):
        self.input_shape = (80,80,4)
        #self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Convolution2D(16, kernel_size=(8, 8), strides=(4, 4),
                 activation='relu',
                 input_shape=self.input_shape))
        model.add(Convolution2D(32, (4, 4), strides=(2,2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(4))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state,(1,80,80,4))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state,(1,80,80,4))
            next_state = np.reshape(next_state,(1,80,80,4))
            target = reward
            if not done:
                target = reward + self.gamma * \
                        np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
           
