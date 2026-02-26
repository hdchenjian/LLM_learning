"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from RL_brain import SarsaLambdaTable
import sys
sys.path.insert(0, '../02_Q_Learning_maze')
from maze_env import Maze

def update():
    for episode in range(63):
        env.title('maze: ' + str(episode))
        observation = env.reset()
        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        # initial all zero eligibility trace
        RL.eligibility_trace *= 0

        step_count = 0
        while True:
            #env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            step_count += 1
            if done:
                print('episode', episode, reward, step_count)
                break
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)), e_greedy = 0.8)

    env.after(100, update)
    env.mainloop()
