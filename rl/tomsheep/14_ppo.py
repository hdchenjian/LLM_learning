import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 设置随机种子，便于复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class MazeEnv(gym.Env):
    """
    自定义迷宫环境，继承自 gym.Env
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MazeEnv, self).__init__()
        # 定义动作空间和状态空间
        self.action_space = gym.spaces.Discrete(4)  # 上、下、左、右
        self.maze_size = (5, 5)
        self.observation_space = gym.spaces.Box(
            low=0, high=4, shape=(2,), dtype=np.int32
        )

        # 定义迷宫（0 表示空地，-1 表示墙壁）
        self.maze = np.zeros(self.maze_size)
        self.maze[0, 3] = -1
        self.maze[1, 1] = -1
        self.maze[1, 3] = -1
        self.maze[2, 1] = -1
        self.maze[3, 3] = -1
        self.maze[4, 1] = -1

        # 起点和终点
        self.start_pos = (0, 0)
        self.goal_pos = (0, 4)
        # 智能体初始位置
        self.agent_pos = self.start_pos

    def step(self, action):
        # 定义动作对应的移动
        directions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        move = directions[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        # 调整后的奖励/惩罚
        step_penalty = -0.1    # 每步行动的负奖励
        wall_penalty = -1      # 撞墙或越界惩罚
        goal_reward = 10       # 到达终点

        reward = step_penalty
        done = False

        # 检查新位置是否在迷宫范围内
        if (0 <= new_pos[0] < self.maze_size[0]) and (0 <= new_pos[1] < self.maze_size[1]):
            # 检查新位置是否是墙壁
            if self.maze[new_pos] == -1:
                # 撞到墙壁
                reward += wall_penalty
            else:
                self.agent_pos = new_pos  # 更新位置
        else:
            # 越界
            reward += wall_penalty

        # 是否到达终点
        if self.agent_pos == self.goal_pos:
            reward += goal_reward
            done = True

        obs = np.array(self.agent_pos)
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.agent_pos = self.start_pos
        return np.array(self.agent_pos)

    def render(self, mode='human'):
        maze_render = np.copy(self.maze)
        maze_render[self.agent_pos] = 2  # 智能体
        maze_render[self.start_pos] = 3  # 起点
        maze_render[self.goal_pos] = 4   # 终点

        symbol_map = {
            -1: 'W',  # 墙壁
            0: ' ',   # 空地
            2: 'A',   # 智能体
            3: 'S',   # 起点
            4: 'G'    # 终点
        }
        print("\n".join(["".join([symbol_map[item] for item in row]) for row in maze_render]))
        print("\n")


class ActorCritic(nn.Module):
    def __init__(self, state_dim=2, action_dim=4, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # 公共特征提取层
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor 分支：输出对每个动作的 logits
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic 分支：输出状态价值 V(s)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.base(x)
        logits = self.actor(features)
        state_value = self.critic(features)
        return logits, state_value


# ------------------------------------------------------------------------------
# 3. Rollout Buffer
# ------------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_done = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.is_done.clear()
        self.values.clear()

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.is_done.append(done)
        self.values.append(value)

    def get_size(self):
        return len(self.states)


# ------------------------------------------------------------------------------
# 4. PPO 核心
# ------------------------------------------------------------------------------
class PPOTrainer:
    def __init__(
        self,
        state_dim=2,
        action_dim=4,
        hidden_dim=64,
        gamma=0.99,
        lr=3e-4,
        clip_eps=0.2,
        update_epochs=5,
        lmbda=0.95,
        vf_coef=0.5,
        ent_coef=0.02  # 略微增大，促进探索
    ):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.lmbda = lmbda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def select_action(self, state):
        """
        输入单个state (numpy array)，输出一个action，以及log_prob等信息
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.ac(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def get_value(self, state):
        """
        给定一个状态，返回 Critic 估计的价值
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.ac(state_t)
        return value.item()

    def compute_gae(self, rewards, values, dones, final_value):
        """
        使用 GAE-lambda 计算优势和回报
        如果最后状态没 done，就用 final_value 作为 bootstrap
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        values = np.append(values, [final_value])

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lmbda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        return advantages, returns

    def update(self, buffer: RolloutBuffer, final_value):
        states = torch.FloatTensor(buffer.states).to(self.device)
        actions = torch.LongTensor(buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(buffer.log_probs).to(self.device)
        rewards = np.array(buffer.rewards, dtype=np.float32)
        dones = np.array(buffer.is_done, dtype=np.float32)
        values = np.array(buffer.values, dtype=np.float32)

        # 计算 GAE
        advantages, returns = self.compute_gae(rewards, values, dones, final_value)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        for _ in range(self.update_epochs):
            logits, value_pred = self.ac(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # 计算比率 ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO Clip
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            value_pred = value_pred.squeeze(-1)
            value_loss = nn.MSELoss()(value_pred, returns)

            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def train_ppo(num_episodes=600, max_steps_per_episode=100):
    """
    使用 PPO 训练迷宫环境：
    - 增加训练回合数
    - 每回合限制 100 步
    """
    env = MazeEnv()

    trainer = PPOTrainer(
        state_dim=2,
        action_dim=4,
        hidden_dim=64,
        gamma=0.99,
        lr=3e-4,
        clip_eps=0.2,
        update_epochs=5,
        lmbda=0.95,
        vf_coef=0.5,
        ent_coef=0.02  # 提高熵系数，引导更多探索
    )

    rollout_buffer = RolloutBuffer()
    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        rollout_buffer.clear()

        for step in range(max_steps_per_episode):
            action, log_prob, value = trainer.select_action(state)
            next_state, reward, done, _ = env.step(action)

            rollout_buffer.add(
                state=state,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=float(done),
                value=value
            )
            state = next_state
            episode_reward += reward

            if done:
                break

        # 如果回合中途没有 done，就从 Critic 得到 bootstrap 价值
        if done:
            final_value = 0.0
        else:
            final_value = trainer.get_value(state)

        # 用 rollouts 中的数据更新网络
        trainer.update(rollout_buffer, final_value)
        all_rewards.append(episode_reward)

        # 打印训练信息
        if (episode+1) % 10 == 0:
            avg_rew = np.mean(all_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Avg10: {avg_rew:.2f}")

    # 训练结束后，保存模型参数
    torch.save(trainer.ac.state_dict(), "ppo_actor_critic_fixed.pth")
    return all_rewards


# ------------------------------------------------------------------------------
# 测试：加载训练好的模型，在迷宫中走若干回合并渲染
# ------------------------------------------------------------------------------
def test_ppo(model_path="ppo_actor_critic_fixed.pth", num_episodes=3):
    env = MazeEnv()

    state_dim = 2
    action_dim = 4
    ac = ActorCritic(state_dim=state_dim, action_dim=action_dim)
    ac.load_state_dict(torch.load(model_path))
    ac.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ac.to(device)

    for epi in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        print(f"===== 测试 Episode {epi+1} =====")
        while not done:
            env.render()

            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, critic_value = ac(state_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()

            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward

        env.render()
        print(f"Episode {epi+1} 结束，回合总奖励: {episode_reward}\n")


if __name__ == "__main__":
    # 1. 训练
    rewards = train_ppo(num_episodes=600, max_steps_per_episode=100)
    # 2. 测试
    test_ppo("ppo_actor_critic_fixed.pth", num_episodes=3)
