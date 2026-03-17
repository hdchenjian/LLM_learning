import random
import numpy as np

from utils import MazeEnv

def sarsa(env, num_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    使用SARSA算法对环境进行训练
    :param env: 训练环境
    :param num_episodes: 训练的轮数
    :param alpha: 学习率
    :param gamma: 折扣因子
    :param epsilon: epsilon-greedy 策略中的探索概率
    :return: Q表 (Q[state_row, state_col, action])
    """
    # Q表的维度：[row, col, action]
    # 这里的迷宫是5x5，动作空间是4
    Q = np.zeros((env.maze_size[0], env.maze_size[1], env.action_space.n))

    def epsilon_greedy_action(state):
        """
        在状态state下使用 epsilon-greedy 策略选择动作
        :param state: (row, col)
        :return: 动作 (int)
        """
        row, col = state
        if np.random.rand() < epsilon:
            return np.random.randint(env.action_space.n)
        else:
            return np.argmax(Q[row, col, :])

    for episode in range(num_episodes):
        # 重置环境，得到初始状态
        state = env.reset()
        state_row, state_col = state
        # 选择初始动作
        action = epsilon_greedy_action((state_row, state_col))

        done = False
        # 这里设定一个最大步数，防止某些情况下无限循环
        for _ in range(200):
            next_state, reward, done, _ = env.step(action)
            next_state_row, next_state_col = next_state
            
            if done:
                # 如果已经到达终点, 直接更新Q并跳出循环
                Q[state_row, state_col, action] += alpha * (reward - Q[state_row, state_col, action])
                break
            else:
                # 选择下一步动作(基于下一个状态)
                next_action = epsilon_greedy_action((next_state_row, next_state_col))
                # SARSA 更新
                Q[state_row, state_col, action] += alpha * (
                    reward + gamma * Q[next_state_row, next_state_col, next_action] - Q[state_row, state_col, action]
                )
                # 状态和动作往前推进
                state_row, state_col = next_state_row, next_state_col
                action = next_action

    return Q

if __name__ == "__main__":
    env = MazeEnv()
    Q = sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    # 测试训练结果：让智能体使用贪心策略走迷宫
    state = env.reset()
    env.render()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done and step_count < 50:
        row, col = state
        # 选取Q值最大的动作
        action = np.argmax(Q[row, col, :])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1
        state = next_state
        env.render()
    
    print(f"测试结束，步数: {step_count}, 总奖励: {total_reward}")
