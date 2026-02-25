"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/tabular-q1
"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(20):
        observation = env.reset()
        while True:
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_
            if done:
                break

    print('game over')
    env.destroy()
    print('\nself.q_table\n', RL.q_table)

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
