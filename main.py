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
optimizer = tf.keras.optimizers.Adam(3e-4)
for _ in range(1000):
    all_obs, all_rewards, all_done, all_actions = [], [], [], []
    for i in range(25):
        # game niveau
        observations, rewards, dones, actions = [], [], [], []
        done = False
        obs = env.reset()
        reward = 1
        while not done:
            # turn niveau, we hebben info niet nodig
            action = 1 if random.random() < model(obs.reshape(1, -1))[0] else 0
            actions.append(action)
            observations.append(obs)
            dones.append(done)
            rewards.append(reward)
            obs, reward, done, _ = env.step(action)

            # env.render()

        # laatste turn voor game over
        all_done.append(dones)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_obs.append(observations)

    all_rewards = decay_and_normalize(all_rewards, 0.97)

    all_actions = np.concatenate(all_actions)
    all_obs = np.concatenate(all_obs)
    batch_size = 250

    indices_list = [x for x in range(len(all_actions))]
    random.shuffle(indices_list)

    all_actions_sorted = all_actions[indices_list]
    all_observations_sorted = all_obs[indices_list]
    all_rewards_sorted = all_rewards[indices_list]
    # all_actions_sorted, all_observations_sorted, all_rewards_sorted = [], [], []
    # for idx, index_from_list in enumerate(indices_list):
    #     all_actions_sorted.append(all_actions[index_from_list])
    #     all_observations_sorted.append(all_obs[index_from_list])
    #     all_rewards_sorted.append(all_rewards[index_from_list])

    print(len(all_actions) / 25)
    aantal_batches = np.ceil(len(all_actions) / batch_size)
    batch_actions = np.array_split(all_actions_sorted, aantal_batches)
    batch_observations = np.array_split(all_observations_sorted, aantal_batches)
    batch_decayed_rewards = np.array_split(all_rewards_sorted, aantal_batches)

    for batch_action, batch_observation, batch_decayed_reward in zip(batch_actions, batch_observations, batch_decayed_rewards):
        with tf.GradientTape() as tape:
            predictions = model(batch_observation)
            loss = tf.keras.losses.mse(batch_action, predictions) * batch_decayed_reward
        train_vars = model.trainable_variables
        grads = tape.gradient(loss, train_vars)
        optimizer.apply_gradients(zip(grads, train_vars))


print('einde verwerking')


