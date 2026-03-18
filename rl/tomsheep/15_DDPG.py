import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

from utils import MazeEnv
import gymnasium as gym

class DiscreteToContinuousWrapper(gym.Wrapper):
    """
    将离散动作空间包装成一个连续动作空间，用于演示 TD3。
    思路：让策略网络输出 4 维 logits，通过 softmax 得到离散动作的概率分布，再选取动作。
    在环境外部看，是一个连续动作空间 Box(-inf, inf, (4,))。
    在与环境交互时，会将该连续向量映射到离散动作{0,1,2,3}。
    """
    def __init__(self, env):
        super(DiscreteToContinuousWrapper, self).__init__(env)
        # 连续动作的维度，就定为 (4,)  对应四个离散动作的 logits
        # 取值范围可以是任意，这里为了简单，设为 [-1, 1]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
    def step(self, action):
        """
        接收连续动作，转为离散动作，然后传给原始环境
        """
        # 将 logits 做 softmax 得到概率分布
        # 注意，这里只是一个简单演示；真实的实现中应考虑数值稳定性
        logits = action
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # 这里可以做随机采样，也可以选取 argmax
        discrete_action = np.random.choice(len(probs), p=probs)
        # 或者 discrete_action = np.argmax(probs)

        return self.env.step(discrete_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# 3.1 策略网络 (Actor): 输入状态，输出 4 维 logits
#     Critic 网络: 输入状态和连续动作，输出 Q 值
# （为简化，Actor、Critic 这里都用非常小的网络）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # 输出 4 维 logits
        )
        
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        这里的 action_dim 指的是 Actor 所输出的连续动作维度(=4)。
        Critic 需要联合 (state, action) 做输入。
        """
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出 Q(s,a)
        )
        
    def forward(self, state, action):
        # state: [batch_size, state_dim]
        # action: [batch_size, action_dim]
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# 3.2 经验回放池
class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states), 
                np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
    ):
        """
        参数：
        state_dim: 状态维度
        action_dim: 连续动作维度 (在本示例中是 4)
        lr: 学习率
        gamma: 折扣因子
        tau: 软更新参数
        policy_noise: 给目标 Actor 的噪声幅度
        noise_clip: 噪声范围 [-noise_clip, noise_clip]
        policy_delay: 每多少步更新一次 Actor
        """
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        # 创建两个 Critic，两个目标 Critic
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        
        # 创建 Actor 和目标 Actor
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)

        # 初始化目标网络参数与主网络相同
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.total_it = 0

    @torch.no_grad()
    def select_action(self, state):
        """
        给定单个状态 state (numpy 数组)，输出 Actor 连续动作 (logits)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]
        action = self.actor(state_tensor)
        return action.squeeze(0).cpu().numpy()
    
    def train_step(self, replay_buffer, batch_size=64):
        self.total_it += 1

        # 1. 从经验池采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(-1)
        
        # 2. 根据 Actor_target 产生 next_actions，加噪声
        with torch.no_grad():
            next_actions = self.actor_target(next_states_tensor)
            # 加噪声
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise)
            
            # 用 Critic_target 计算目标 Q 值
            target_Q1 = self.critic1_target(next_states_tensor, next_actions)
            target_Q2 = self.critic2_target(next_states_tensor, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards_tensor + (1 - dones_tensor) * self.gamma * target_Q
        
        # 3. 更新 Critic
        current_Q1 = self.critic1(states_tensor, actions_tensor)
        current_Q2 = self.critic2(states_tensor, actions_tensor)
        
        critic1_loss = nn.MSELoss()(current_Q1, target_Q)
        critic2_loss = nn.MSELoss()(current_Q2, target_Q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 4. 延迟更新 Actor 和目标网络
        if self.total_it % self.policy_delay == 0:
            # Actor loss: 让 Critic1 对 (s, actor(s)) 的 Q 值最大
            actor_actions = self.actor(states_tensor)
            actor_loss = -self.critic1(states_tensor, actor_actions).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            self._soft_update(self.actor, self.actor_target)
    
    def _soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

def train_td3_on_discrete_maze_env(num_episodes=200, max_steps=100, batch_size=64):
    # 1. 创建迷宫环境，并用离散到连续的包装器包起来
    env = MazeEnv()
    env = DiscreteToContinuousWrapper(env)

    # 2. 获取状态、动作空间的维度 (包装器后动作维度是 4)
    state_dim = env.observation_space.shape[0]   # 原始 2 维状态
    action_dim = env.action_space.shape[0]       # 4
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # 3. 创建 TD3 Agent 和经验回放池
    agent = TD3Agent(state_dim, action_dim)
    replay_buffer = ReplayBuffer()

    # 4. 预填充一些随机数据 (可选)
    pre_fill_steps = 500
    state = env.reset()
    for _ in range(pre_fill_steps):
        # 随机动作
        random_action = env.action_space.sample()
        next_state, reward, done, _ = env.step(random_action)
        replay_buffer.push(state, random_action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

    # 5. 正式训练
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
            # 与环境交互
            next_state, reward, done, info = env.step(action)
            # 存入经验池
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward

            # 更新网络
            if len(replay_buffer) > batch_size:
                agent.train_step(replay_buffer, batch_size)

            if done:
                break
        
        print(f"Episode {episode}, Reward: {episode_reward}")
    return agent

if __name__ == "__main__":
    agent = train_td3_on_discrete_maze_env()

    # 测试: 用训练好的 agent 玩几回合
    test_env = MazeEnv()
    test_env = DiscreteToContinuousWrapper(test_env)
    for i in range(3):
        state = test_env.reset()
        done = False
        total_reward = 0
        while not done:
            #test_env.render()       # 打印迷宫，可观察 Agent 的位置
            action = agent.select_action(state)
            state, reward, done, _ = test_env.step(action)
            total_reward += reward
        print(f"[Test Episode {i}] Reward: {total_reward}")
