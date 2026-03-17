import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from utils import MazeEnv

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=4, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        # 输入: (x, y) 2维状态
        # 输出: 对应 4 个离散动作的概率分布
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # 输出动作概率
        )

    def forward(self, x):
        return self.layers(x)


def compute_discounted_returns(rewards, gamma=0.99):
    """计算从每个时间步开始的折扣回报 G_t。"""
    discounted_returns = []
    G = 0
    # 从后往前计算
    for r in reversed(rewards):
        G = r + gamma * G
        discounted_returns.insert(0, G)  # 头部插入
    return discounted_returns


def train_reinforce(
    num_episodes=500,
    gamma=0.99,
    lr=1e-3,
    render_interval=0,
    model_save_path=None
):
    """
    使用 REINFORCE 方法训练策略网络。
    参数:
    - num_episodes: 训练的总回合数
    - gamma: 折扣因子
    - lr: 学习率
    - render_interval: 若 > 0，则每隔多少回合渲染一次迷宫
    - model_save_path: 若不为 None, 则在训练结束保存模型到该路径
    """
    env = MazeEnv()
    policy_net = PolicyNetwork()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    reward_history = []

    for i_episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        done = False
        step_count = 0

        # ----------------------------
        # 生成一条完整的回合(episode)
        # ----------------------------
        while not done:
            # 是否需要渲染
            if render_interval > 0 and (i_episode + 1) % render_interval == 0:
                env.render()

            # 转换状态为张量
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # 前向传播, 得到动作概率分布
            action_probs = policy_net(state_tensor)
            dist = Categorical(action_probs)
            # 依据分布采样动作
            action = dist.sample()
            # 记录该动作的对数概率，以用于梯度更新
            log_prob = dist.log_prob(action)
            # 与环境交互
            next_state, reward, done, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            step_count += 1

        # -----------------------------
        # 计算回合总折扣回报并回传梯度
        # -----------------------------
        discounted_returns = compute_discounted_returns(rewards, gamma)

        # 标准化 returns (可选, 常见做法)
        discounted_returns = torch.FloatTensor(discounted_returns)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / \
                             (discounted_returns.std() + 1e-9)

        # 计算 loss = - Σ (log_pi(a_t|s_t) * Gt)
        loss = 0
        for log_prob, Gt in zip(log_probs, discounted_returns):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        reward_history.append(episode_reward)

        print(f"Episode {i_episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")

    # 如果指定了保存路径，保存训练好的策略网络
    if model_save_path is not None:
        torch.save(policy_net.state_dict(), model_save_path)
        print(f"模型已保存到 {model_save_path}")

    return reward_history


def test_reinforce(model_path, num_episodes=5):
    """
    加载训练好的策略网络，并在迷宫环境中测试 num_episodes 回合。
    测试时会将迷宫渲染打印出来，以查看智能体的走法。
    """
    env = MazeEnv()
    policy_net = PolicyNetwork()
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    for i in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        print(f"===== 测试 Episode {i+1} =====")
        while not done:
            env.render()

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            state = next_state
            total_reward += reward

        env.render()
        print(f"Episode {i+1} 结束，总奖励: {total_reward}\n")


if __name__ == '__main__':
    # 1. 训练
    train_episodes = 500
    model_path = "reinforce_policy.pth"
    reward_history = train_reinforce(num_episodes=train_episodes,
                                     gamma=0.99,
                                     lr=1e-3,
                                     render_interval=0,   # 可以设为例如 50 看看训练中间的走法
                                     model_save_path=model_path)

    # 2. 测试
    test_reinforce(model_path, num_episodes=3)
