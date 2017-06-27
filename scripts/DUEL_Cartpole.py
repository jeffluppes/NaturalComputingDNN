# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 20:31:54 2017

@author: Gebruiker
"""

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import PReLU

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'CartPole-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n

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
model.add(Dense(nb_actions, activation='softmax'))
print(model.summary())

# Finally, we configure and compile the agent.
memory = SequentialMemory(limit=100000, window_length=1)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Fitting the training
dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)

# Saving the final weights
dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Visualize the best 10 episodes
dqn.test(env, nb_episodes=10, visualize=False)