import gym
import numpy as np
import random
import keras
import cv2
from ReplayBuffer import ReplayBuffer
from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

class DQN(object):
    def __init__(self, number_of_frames, number_of_actions):
        self.learning_rate = 0.99
        self.tau = 0.01
        self.number_of_frames = number_of_frames
        self.number_of_actions = number_of_actions
        self.construct_q_network()

    def construct_q_network(self):
        self.model = Sequential()
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, self.number_of_frames)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.number_of_actions))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))

        self.target_model = Sequential()
        self.target_model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, self.number_of_frames)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 3, 3))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.target_model.add(Dense(self.number_of_actions))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
        self.target_model.set_weights(self.model.get_weights())

    def predict(self, data, epsilon):
        q_actions = self.model.predict(data.reshape(1, 84, 84, self.number_of_frames), batch_size = 1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, self.number_of_actions)
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch):
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, self.number_of_actions))

        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, 84, 84, self.number_of_frames), batch_size = 1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, 84, 84, self.number_of_frames), batch_size = 1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += self.learning_rate * np.max(fut_action)

        loss = self.model.train_on_batch(s_batch, targets)

    def save_network(self, path):
        self.model.save(path)

    def load_network(self, path):
        self.model = load_model(path)

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)
