# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 12:08:28 2017

@author: Gebruiker
"""

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import PReLU

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# Building a simple model that can be extended with the layers
# Acitivation can be (relu, sigmoid, linear)
# model.add(Dense(16)), model.add(Activation('relu')) can be repeated between 1 and 10
# Neuron amounts are 16, 32, 64, 128, 256
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
#model.add(PReLU())
#model.add(Dropout(0.1))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

print(model.summary())

# Finally, we configure and compile the agent.
memory = EpisodeParameterMemory(limit=1000, window_length=1)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()

# Fitting the training
cem.fit(env, nb_steps=100000, visualize=False, verbose=2)

# Saving the final weights
cem.save_weights('cem_{}_params.h5f'.format(ENV_NAME), overwrite=True)

# Visualize the best 10 episodes
cem.test(env, nb_episodes=10, visualize=True)