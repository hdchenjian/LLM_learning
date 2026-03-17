# 1. 将迷宫中的每个可行位置(非墙壁)视作一个状态，终点视作终止状态。  
# 2. 对每个状态定义一个策略(在这里可以用“每个状态随机选择一个动作”作为初始化策略)。  
# 3. 进行策略迭代，包括以下循环：  
#   • 策略评估：在固定策略下，反复迭代计算每个状态的价值，直到收敛。  
#   • 策略改进：在每个状态上选取能够最大化该状态价值的动作，更新策略。如果策略不再发生变化则停止迭代。  

import numpy as np

from utils import MazeEnv

def policy_iteration(env, gamma=0.9, theta=1e-5, max_iter=1000):
    """
    使用策略迭代求解迷宫。
    gamma: 折扣因子(小于1，避免负循环)
    theta: 收敛阈值
    max_iter: 策略迭代的最大迭代次数，防止死循环
    """

    # 1) 收集所有非墙壁状态
    states = []
    for r in range(env.maze_size[0]):
        for c in range(env.maze_size[1]):
            if env.maze[r, c] != -1: 
                states.append((r, c))

    actions = [0, 1, 2, 3]  # 上、下、左、右

    def step_in_model(state, action):
        # 终点不需要再动
        if state == env.goal_pos:
            return state, 0.0, True

        directions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }
        move = directions[action]
        new_state = (state[0] + move[0], state[1] + move[1])

        if not (0 <= new_state[0] < env.maze_size[0] and 0 <= new_state[1] < env.maze_size[1]):
            # 越界
            return state, -5, False
        if env.maze[new_state] == -1:
            # 撞墙
            return state, -5, False

        # 正常移动
        reward = -1
        done = False
        if new_state == env.goal_pos:
            reward = 10
            done = True
        return new_state, reward, done

    # 2) 初始化策略、价值函数
    pi = {}
    V = {}
    for s in states:
        if s == env.goal_pos:
            pi[s] = None
            V[s] = 0.0
        else:
            pi[s] = np.random.choice(actions)
            V[s] = 0.0

    # 3) 策略迭代
    iter_count = 0
    while True:
        iter_count += 1
        if iter_count > max_iter:
            print("超过最大迭代次数，提前退出，可能未完全收敛。")
            break

        # ========== (A) 策略评估 ==========
        while True:
            delta = 0
            for s in states:
                if s == env.goal_pos:
                    continue
                v_old = V[s]
                a = pi[s]
                s_next, r, done = step_in_model(s, a)
                if done:
                    V[s] = r
                else:
                    V[s] = r + gamma * V[s_next]
                delta = max(delta, abs(V[s] - v_old))
            if delta < theta:
                break

        # ========== (B) 策略改进 ==========
        policy_stable = True
        for s in states:
            if s == env.goal_pos:
                continue
            old_a = pi[s]

            best_a = None
            best_q = float('-inf')
            for a in actions:
                s_next, r, done = step_in_model(s, a)
                q_sa = r if done else (r + gamma * V[s_next])
                if q_sa > best_q:
                    best_q = q_sa
                    best_a = a

            pi[s] = best_a
            if best_a != old_a:
                policy_stable = False

        if policy_stable:
            print(f"策略在迭代 {iter_count} 次后稳定。")
            break

    return pi, V

if __name__ == "__main__":
    env = MazeEnv()
    policy, value = policy_iteration(env, gamma=0.9, theta=1e-5, max_iter=200)

    # 查看价值函数与策略
    print("最终价值函数(部分):")
    for s in sorted(value.keys()):
        print(f"State {s}: V = {value[s]:.2f}")

    print("\n最终策略：")
    action_dict = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    for s in sorted(policy.keys()):
        if policy[s] is None:
            print(f"{s} -> 终点")
        else:
            print(f"{s} -> {action_dict[policy[s]]}")

    # 用最终策略跑一遍环境
    obs = env.reset()
    env.render()
    done = False
    step_count = 0
    while not done and step_count < 50:
        s = tuple(obs)
        action = policy[s]
        obs, reward, done, _ = env.step(action)
        env.render()
        step_count += 1

    print("Episode finished!")
