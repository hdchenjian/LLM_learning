import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from utils import MazeEnv

class DQN(nn.Module):
    def __init__(self, state_dim=2, action_dim=4):
        super(DQN, self).__init__()
        # 对于迷宫环境，输入是 (x, y) 两维（整数），所以输入层大小为 2
        # 输出层大小为 action_dim (4), 对应上下左右四个动作的 Q-value
        hidden_dim = 64
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim))

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------------
# 3. 定义经验回放（Replay Buffer）
# --------------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# --------------------------------------------------------------------------------
# 4. 训练 DQN
# --------------------------------------------------------------------------------
def train_dqn(model_name):
    env = MazeEnv()
    # 定义超参数
    num_episodes = 500
    batch_size = 32
    gamma = 0.99
    lr = 1e-3

    # epsilon 贪心相关参数
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 300  # 调整衰减速度

    target_update_interval = 50  # 每隔多少个 episode 同步一次目标网络
    replay_buffer_capacity = 10000

    # 创建网络
    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # 优化器
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # 经验回放缓冲区
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    # 记录奖励信息
    all_rewards = []

    # 训练过程
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        episode_reward = 0
        done = False

        # 计算当前 epsilon
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
            np.exp(-1. * episode / epsilon_decay)

        while not done:
            # 根据 epsilon 贪心选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.argmax(dim=1).item()

            # 与环境进行一步交互
            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # 将 transition 存到经验回放中
            replay_buffer.push(
                state.squeeze(0).numpy(),
                action,
                reward,
                next_state_tensor.squeeze(0).numpy(),
                done
            )

            episode_reward += reward
            state = next_state_tensor

            # 每步都尝试训练（如果缓冲区够大）
            if len(replay_buffer) >= batch_size:
                # 从回放缓冲区采样
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
                states_b = torch.FloatTensor(states_b)
                actions_b = torch.LongTensor(actions_b)
                rewards_b = torch.FloatTensor(rewards_b)
                next_states_b = torch.FloatTensor(next_states_b)
                dones_b = torch.FloatTensor(dones_b)

                # 计算 Q(s, a)
                q_values = policy_net(states_b)
                # 选出与动作对应的 Q-value
                q_values = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                # 计算 Q'(s', a') 来 更新目标
                with torch.no_grad():
                    # 使用target_net来计算 max Q'(s', a')
                    next_q_values = target_net(next_states_b)
                    max_next_q_values = next_q_values.max(dim=1)[0]
                    # 如果结束，那么目标是 reward；否则是 reward + gamma * max Q'(s', a')
                    target_q_values = rewards_b + gamma * (1 - dones_b) * max_next_q_values

                # 计算损失
                loss = nn.MSELoss()(q_values, target_q_values)

                # 反向传播和更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        all_rewards.append(episode_reward)

        # 每隔一段时间更新目标网络
        if (episode + 1) % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 打印训练信息
        print(f"Episode {episode+1}, Epsilon: {epsilon:.3f}, Reward: {episode_reward}")

    # 保存训练好的网络
    torch.save(policy_net.state_dict(), model_name)
    print(f"model saved: {model_name}")

    return all_rewards


def test_dqn(trained_model_path, num_episodes=1):
    """
    使用训练好的DQN模型在迷宫环境中测试 num_episodes 次，
    并通过 env.render() 在控制台打印出路径。
    参数:
        trained_model_path: str, 已保存的模型文件路径，例如 'dqn_policy.pth'
        num_episodes: int, 测试的回合数
    """
    # 1. 创建与训练时相同的迷宫环境
    env = MazeEnv()

    # 2. 构建与训练时相同的网络结构，并加载训练好的模型参数
    policy_net = DQN(state_dim=2, action_dim=4)
    policy_net.load_state_dict(torch.load(trained_model_path))
    policy_net.eval()  # 推断模式

    for episode in range(num_episodes):
        state = env.reset()                    # 环境初始化
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        episode_reward = 0

        print(f"===== 测试 Episode {episode + 1} 开始 =====")
        while not done:
            # 打印环境的当前状态
            env.render()

            # 使用训练好的策略网络，选择最优动作（贪心）
            with torch.no_grad():
                q_values = policy_net(state)
                action = q_values.argmax(dim=1).item()

            # 与环境交互
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            episode_reward += reward
            state = next_state

        # 最后再渲染一次，以便显示终止状态
        env.render()
        print(f"Episode {episode + 1} 结束，总奖励: {episode_reward}\n")


if __name__ == "__main__":
    model_name = 'dqn_policy.pth'
    rewards = train_dqn(model_name)
    test_dqn(model_name)
