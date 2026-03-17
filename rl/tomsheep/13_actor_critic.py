import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from utils import MazeEnv

class ActorCritic(nn.Module):
    """
    一个简单的Actor-Critic结构:
      - actor_head: 输出对各动作的概率分布 (logits)
      - critic_head: 输出该状态的价值 V(s)
    """
    def __init__(self, state_dim=2, action_dim=4, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Actor 部分输出对每个动作的logits (还需经过Softmax)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        # Critic 部分输出状态价值 V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        前向传播：
        输入： x (batch_size, state_dim)
        输出： actor_logits (batch_size, action_dim),
              critic_value (batch_size, 1)
        """
        shared_out = self.shared(x)
        actor_logits = self.actor_head(shared_out)
        critic_value = self.critic_head(shared_out)
        return actor_logits, critic_value


def train_actor_critic():
    env = MazeEnv()

    # 超参数
    num_episodes = 100
    gamma = 0.99
    lr = 1e-3

    # 创建网络
    model = ActorCritic(state_dim=2, action_dim=4, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 用于记录所有回合的总奖励
    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)  # shape: (1,2)
        done = False

        # 记录一整个回合的 (state, action, reward, log_probs, values)
        transitions = []
        episode_reward = 0

        while not done:
            # 前向传播，得到logits和价值
            logits, value = model(state)   # logits: shape (1, action_dim)
            # 根据logits采样动作
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # 与环境交互
            next_state, reward, done, _ = env.step(action.item())
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)

            transitions.append({'state': state, 'action': action, 'reward': reward, 'log_prob': log_prob, 'value': value})

            episode_reward += reward
            state = next_state_t

        # 回合结束后，计算 returns 并更新网络
        # 1) 计算每个时间步的回报 G_t
        returns = []
        G = 0
        for t in reversed(transitions):
            G = t['reward'] + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)  # shape: (T,1)

        # 2) 计算 Actor 和 Critic 的loss并反向传播
        actor_loss = 0
        critic_loss = 0
        for i, trans in enumerate(transitions):
            advantage = returns[i] - trans['value']
            # Actor损失: -log(pi(a|s)) * advantage (策略梯度)
            actor_loss += -trans['log_prob'] * advantage.detach()
            # Critic损失: MSE( V(s) - G_t )
            critic_loss += advantage.pow(2)

        loss = actor_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_rewards.append(episode_reward)
        # 打印训练信息
        print(f"Episode {episode+1}, Reward: {episode_reward}")

    # 训练结束后，保存模型
    torch.save(model.state_dict(), 'actor_critic_model.pth')
    return all_rewards


def test_actor_critic(model_path='actor_critic_model.pth', num_episodes=1):
    """
    使用训练好的Actor-Critic模型在迷宫环境中测试 num_episodes 次，
    并通过 env.render() 在控制台打印出路径。
    参数:
        model_path: str, 已保存的模型文件路径，例如 'actor_critic_model.pth'
        num_episodes: int, 测试的回合数
    """
    env = MazeEnv()

    # 构建网络并加载参数
    model = ActorCritic(state_dim=2, action_dim=4, hidden_dim=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        episode_reward = 0
        print(f"===== 测试 Episode {episode + 1} 开始 =====")

        while not done:
            env.render()
            with torch.no_grad():
                logits, _ = model(state)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward
            state = torch.FloatTensor(next_state).unsqueeze(0)

        # 渲染最终状态
        env.render()
        print(f"Episode {episode + 1} 结束，总奖励: {episode_reward}\n")


if __name__ == "__main__":
    # 1) 训练
    #   如果你已经训练过并保存了模型，可以注释掉此行并只执行测试。
    rewards = train_actor_critic()

    # 2) 测试
    #   如果已经存在 "actor_critic_model.pth"，可以直接进行测试
    #   也可以在训练完后直接测试
    test_actor_critic(model_path='actor_critic_model.pth', num_episodes=3)
