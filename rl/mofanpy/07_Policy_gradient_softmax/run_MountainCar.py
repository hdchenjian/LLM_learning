import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore',category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import gymnasium as gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = False  # rendering wastes time
RENDER = 0

env = gym.make('MountainCar-v0', render_mode="human")
#env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.995,
    # output_graph=True,
)

for i_episode in range(1000):
    observation, info = env.reset(seed=42)
    step_count = 0
    car_left = 0
    while True:
        if RENDER: env.render()
        action = RL.choose_action(observation)
        if observation[0] < -0.9: car_left = 1
        #if car_left: action = 2
        observation_, reward, done, truncated, info = env.step(action)     # reward = -1 in all cases
        #print('reward, done, truncated, info', reward, done, truncated, info, i_episode, step_count, observation, car_left)
        RL.store_transition(observation, action, reward)
        step_count += 1

        if done:
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            #if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
            print("episode:", i_episode, "  reward:", int(running_reward))
            vt = RL.learn()  # train

            if i_episode == 30:
                plt.plot(vt)  # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                #plt.show()

            break
        observation = observation_

env.close()
