# Jullie zijn wel erg ver, prachtig om die indeling in meerdere files te zien, goed bezig!




import gym
from models import get_model
from funcs import decay_and_normalize
import random


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

        env.render()

    # laatste turn voor game over
    all_done.append(dones)
    all_rewards.append(rewards)
    all_obs.append(observations)


all_rewards = decay_and_normalize(all_rewards, 0.9)
print(all_rewards)
