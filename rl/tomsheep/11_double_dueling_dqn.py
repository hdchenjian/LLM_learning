import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from utils import MazeEnv

class DuelingDQN(nn.Module):
    def __init__(self, state_dim=2, action_dim=4):
        super(DuelingDQN, self).__init__()
        self.hidden_dim = 64

        # 共有的特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.ReLU()
        )

        # 价值函数 (Value) 分支
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        # 优势函数 (Advantage) 分支
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, state_dim]
        features = self.feature_layer(x)
        values = self.value_stream(features)           # [batch_size, 1]
        advantages = self.advantage_stream(features)   # [batch_size, action_dim]
        # Dueling DQN 的合并公式: Q(s,a) = V(s) + A(s,a) - mean(A(s,a), dim=1, keepdim=True)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# 4. 训练函数 (Double DQN + Dueling DQN)
def train_double_dueling_dqn():
    env = MazeEnv()
    # ------------------
    # 超参数设置
    # ------------------
    num_episodes = 500
    batch_size = 32
    gamma = 0.99
    lr = 1e-3

    # epsilon 贪心相关参数
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 300  # 调整衰减速度

    target_update_interval = 50   # 每隔多少个 episode 同步一次目标网络
    replay_buffer_capacity = 10000

    # 创建网络 (policy_net 和 target_net)
    policy_net = DuelingDQN()
    target_net = DuelingDQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # 优化器
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # 经验回放缓冲区
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    # 记录奖励信息
    all_rewards = []

    # ------------------
    # 训练过程
    # ------------------
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        episode_reward = 0

        # 计算当前 epsilon (随训练进程衰减)
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
            np.exp(-1. * episode / epsilon_decay)

        while not done:
            # 根据 epsilon 贪心策略选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.argmax(dim=1).item()

            # 与环境进行一步交互
            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            episode_reward += reward

            # 存储到 Replay Buffer
            replay_buffer.push(state.squeeze(0).numpy(), 
                               action, 
                               reward, 
                               next_state_tensor.squeeze(0).numpy(), 
                               done)

            state = next_state_tensor

            # 每步都进行一次学习 (如果缓冲区大小足够)
            if len(replay_buffer) >= batch_size:
                (states_b, actions_b, rewards_b, 
                 next_states_b, dones_b) = replay_buffer.sample(batch_size)

                states_b = torch.FloatTensor(states_b)
                actions_b = torch.LongTensor(actions_b)
                rewards_b = torch.FloatTensor(rewards_b)
                next_states_b = torch.FloatTensor(next_states_b)
                dones_b = torch.FloatTensor(dones_b)

                # ---------------------------------------------
                # 计算当前 Q(s, a) 
                # ---------------------------------------------
                q_values = policy_net(states_b)
                # 选出与动作对应的 Q-value
                q_values = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                # ---------------------------------------------
                # Double DQN 的目标 Q 值计算:
                #  a. 使用 policy_net 选择使 Q 值最大的动作 a' = argmax(Q_online(s'), a')
                #  b. 使用 target_net 评估该动作对应的 Q'(s',a')
                # ---------------------------------------------
                with torch.no_grad():
                    # a. 使用 policy_net 在下一状态上选择动作
                    next_q_online = policy_net(next_states_b)            # [batch_size, action_dim]
                    next_actions = next_q_online.argmax(dim=1)           # [batch_size]

                    # b. 使用 target_net 来获取 Q'(s', a')
                    next_q_target = target_net(next_states_b)            # [batch_size, action_dim]
                    next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

                    # 如果 done=True，则没有后续奖励
                    target_q_values = rewards_b + gamma * (1 - dones_b) * next_q_values

                # 计算损失 (均方误差)
                loss = nn.MSELoss()(q_values, target_q_values)

                # 反向传播和更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 记录本回合奖励
        all_rewards.append(episode_reward)

        # 间隔一定周期后更新目标网络
        if (episode + 1) % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 打印训练信息
        print(f"Episode {episode+1}, Epsilon: {epsilon:.3f}, Reward: {episode_reward}")

    # 训练完成后，可选择保存模型
    torch.save(policy_net.state_dict(), "double_dueling_dqn.pth")
    return all_rewards


def test_double_dueling_dqn(trained_model_path, num_episodes=1):
    """
    使用训练好的Double Dueling DQN模型在迷宫环境中测试 num_episodes 次，
    并通过 env.render() 在控制台打印出路径。
    参数:
        trained_model_path: str, 已保存的模型文件路径，例如 'double_dueling_dqn.pth'
        num_episodes: int, 测试的回合数
    """
    env = MazeEnv()

    # 构建与训练时相同的网络结构，并加载模型参数
    policy_net = DuelingDQN(state_dim=2, action_dim=4)
    policy_net.load_state_dict(torch.load(trained_model_path))
    policy_net.eval()  # 推断模式

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        episode_reward = 0

        print(f"===== 测试 Episode {episode + 1} 开始 =====")
        while not done:
            env.render()

            with torch.no_grad():
                q_values = policy_net(state)
                action = q_values.argmax(dim=1).item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            episode_reward += reward
            state = next_state

        # 渲染最终位置
        env.render()
        print(f"Episode {episode + 1} 结束，总奖励: {episode_reward}\n")


if __name__ == "__main__":
    # 1. 训练
    print("开始训练 Double Dueling DQN ...")
    rewards = train_double_dueling_dqn()
    print("训练结束！")

    # 2. 测试
    print("开始测试 ...")
    test_double_dueling_dqn("double_dueling_dqn.pth", num_episodes=3)
