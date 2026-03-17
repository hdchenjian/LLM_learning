import random
import numpy as np

from utils import MazeEnv

def epsilon_greedy_action(state, Q, epsilon):
    """ ε-贪心动作选择：以概率 epsilon 随机选取动作，否则选择 Q 值最大的动作 """
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[-1])
    else:
        return np.argmax(Q[state[0], state[1], :])

def q_lambda_train(env,
                   alpha=0.1,      # 学习率
                   gamma=0.99,     # 折扣因子
                   lam=0.9,        # 资格迹衰减系数
                   epsilon=0.1,    # ε-贪心系数
                   num_episodes=500,
                   max_steps_per_episode=100):
    """
    使用 Watkins's Q(λ) 算法来训练智能体。

    参数说明：
    -----------
    env : OpenAI Gym 兼容环境
    alpha : float
        学习率
    gamma : float
        折扣因子
    lam : float
        资格迹衰减系数
    epsilon : float
        ε-贪心系数
    num_episodes : int
        训练的总回合数
    max_steps_per_episode : int
        每个回合的最大步数
    """
    # 建立一个 Q 表，状态空间 5x5, 动作空间 4
    Q = np.zeros((env.maze_size[0], env.maze_size[1], env.action_space.n))

    # 记录每个回合结束时的总奖励，便于观察收敛情况
    episode_rewards = []

    for episode in range(num_episodes):
        # 重置环境
        state = env.reset()         # 当前状态 (row, col)
        # 将资格迹初始化为全零
        E = np.zeros((env.maze_size[0], env.maze_size[1], env.action_space.n))

        # ε-贪心动作选择
        action = epsilon_greedy_action(state, Q, epsilon)

        total_reward = 0
        done = False

        for _ in range(max_steps_per_episode):
            # 交互一步
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # 下一个动作
            next_action = epsilon_greedy_action(next_state, Q, epsilon)

            # 计算 TD 误差：Q-learning 使用 max_a' Q(next_state, a')
            td_target = reward + gamma * np.max(Q[next_state[0], next_state[1], :]) if not done else reward
            td_error = td_target - Q[state[0], state[1], action]

            # 资格迹更新（ Watkins’s Q(λ) “替换迹”）
            # 1) 让所有 E 都衰减
            E *= gamma * lam
            # 2) 将当前状态动作对的 E 置为 1
            E[state[0], state[1], action] = 1

            # 对所有 (s,a) 执行批量 Q 更新
            # Watkins’s Q(λ) 中，如果下一步动作不是贪心，可以选择截断迹；此处简单实现不对迹进行截断。
            Q += alpha * td_error * E

            # 如果使用严格的 Watkins’s Q(λ)，当下一步动作不是 greedy 时，可以将 E 置零
            # 但是也可以使用 Peng’s Q(λ) 之类的变体。此处代码为了简洁，不再细分。

            # 转移到下一状态与动作
            state = next_state
            action = next_action

            if done:
                break

        episode_rewards.append(total_reward)

    return Q, episode_rewards


if __name__ == "__main__":
    env = MazeEnv()
    # 训练智能体
    Q, episode_rewards = q_lambda_train(env, alpha=0.1, gamma=0.99, lam=0.9, epsilon=0.1, num_episodes=500, max_steps_per_episode=100)

    # 观察训练后每个回合的奖励
    print("训练结束。每个回合的总奖励如下所示：")
    print(episode_rewards)

    # 测试阶段：让智能体基于学到的 Q 表走一些步，看是否能到达目标
    state = env.reset()
    done = False
    step_count = 0
    while not done and step_count < 50:  # 测试上限 50 步
        env.render()
        action = np.argmax(Q[state[0], state[1], :])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        step_count += 1

    env.render()
    print(f"测试结束，共执行了 {step_count} 步，是否到达目标：{done}")
