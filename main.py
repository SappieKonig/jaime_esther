import gym
from models import get_model
from funcs import decay_and_normalize


# create gym
env = gym.make("CartPole-v0")
env.reset()

# assign model to variable
model = get_model()

# totaal
all_obs = []
all_rewards = []
all_done = []
for i in range(100):
    # game niveau
    observations = []
    rewards = []
    dones = []
    done = False
    while not done:
        # turn niveau, we hebben info niet nodig
        observation, reward, done, _ = env.step(0)
        observations.append(observation)
        rewards.append(reward)
        dones.append(done)
        env.render()

    # laatste turn voor game over
    env.reset()
    all_done.append(dones)
    all_rewards.append(rewards)
    all_obs.append(observations)


all_rewards = decay_and_normalize(all_rewards, 0.9)
print(all_rewards)
