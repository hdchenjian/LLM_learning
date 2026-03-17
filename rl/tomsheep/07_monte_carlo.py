import random
import numpy as np

from utils import MazeEnv

def mc_control_on_policy(env, num_episodes=5000, gamma=1.0, epsilon=0.1):
    """
    基于第一访问蒙特卡洛的 on-policy 控制（ε-贪心）。
    :param env: 自定义迷宫环境
    :param num_episodes: 训练的回合数
    :param gamma: 折扣因子
    :param epsilon: 探索率
    :return: Q, 最优的状态-动作价值函数
    """
    # Q 表示状态-动作价值函数，大小为 [行, 列, 动作数]
    Q = np.zeros((env.maze_size[0], env.maze_size[1], env.action_space.n))

    # 这里使用一个字典来存储每个状态-动作对的回报（列表），方便后续取平均做更新
    returns = dict()
    for r in range(env.maze_size[0]):
        for c in range(env.maze_size[1]):
            for a in range(env.action_space.n):
                returns[((r, c), a)] = []

    def epsilon_greedy_policy(state):
        """
        给定当前的 Q 和 explored state, 采用 ε-贪心策略选择动作
        """
        r, c = state
        if random.random() < epsilon:
            # 随机探索
            return np.random.choice(env.action_space.n)
        else:
            # 贪心选择
            return np.argmax(Q[r, c])

    for episode in range(num_episodes):
        # 生成一条回合（episode）
        state = env.reset()
        episode_trace = []  # 存储 (state, action, reward) 元组

        done = False
        while not done:
            action = epsilon_greedy_policy(tuple(state))
            next_state, reward, done, _ = env.step(action)
            episode_trace.append((tuple(state), action, reward))
            state = next_state

        # 回溯回合，更新 Q
        visited_state_actions = set()
        G = 0  # 从后往前计算折扣回报
        # 在这里从后向前计算更简洁（若想从前向后可先沿 episode_trace 再次扫一遍计算回报）
        for t in reversed(range(len(episode_trace))):
            s_t, a_t, r_t = episode_trace[t]
            G = gamma * G + r_t
            # 检查是否是该回合中首次出现的 (s_t, a_t)
            if (s_t, a_t) not in visited_state_actions:
                visited_state_actions.add((s_t, a_t))
                returns[(s_t, a_t)].append(G)
                # 增量方式更新 Q(s, a)
                Q[s_t[0], s_t[1], a_t] = np.mean(returns[(s_t, a_t)])
    return Q

if __name__ == "__main__":
    # 创建环境
    env = MazeEnv()

    # 使用蒙特卡洛方法进行训练
    Q = mc_control_on_policy(env, num_episodes=3000, gamma=1.0, epsilon=0.1)

    # 打印最终学到的 Q
    print("训练结束后学到的状态-动作价值函数 Q：")
    for r in range(env.maze_size[0]):
        for c in range(env.maze_size[1]):
            print(f"State=({r},{c}) -> Q={Q[r, c]}")
        print()

    # 根据学到的 Q 构造出一个贪心策略并测试
    def greedy_policy(state):
        return np.argmax(Q[state[0], state[1]])

    # 测试智能体在环境中的表现
    state = env.reset()
    env.render()
    done = False
    step_count = 0
    while not done and step_count < 50:  # 做一个简单的步数限制，防止卡死
        action = greedy_policy(tuple(state))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        step_count += 1

    if tuple(state) == env.goal_pos:
        print("智能体成功到达目标！")
    else:
        print("智能体未能到达目标。")
