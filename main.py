import gym
from models import get_model
from funcs import decay_and_normalize
import random
import tensorflow as tf
import numpy as np


# create gym
env = gym.make("CartPole-v0")
env.reset()

# assign model to variable
model = get_model()

# totaal
all_obs, all_rewards, all_done, all_actions = [], [], [], []
for i in range(100):
    # game niveau
    observations, rewards, dones, actions = [], [], [], []
    done = False
    obs = env.reset()
    while not done:
        # turn niveau, we hebben info niet nodig
        action = 1 if random.random() < model(obs.reshape(1, -1))[0] else 0
        obs, reward, done, _ = env.step(action)

        observations.append(obs)
        dones.append(done)
        rewards.append(reward)
        actions.append(action)

        # env.render()

    # laatste turn voor game over
    all_done.append(dones)
    all_rewards.append(rewards)
    all_obs.append(observations)


all_rewards = decay_and_normalize(all_rewards, 0.9)
print(all_rewards)

optimizer = tf.keras.optimizers.Adam(3e-4)
with tf.GradientTape() as tape:
    predictions = model(np.concatenate(all_obs))
    loss = tf.keras.losses.mse(all_actions, predictions)

train_vars = model.trainable_variables
grads = tape.gradient(loss, train_vars)
optimizer.apply_gradients(zip(grads, train_vars))

x =1


