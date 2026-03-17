import gymnasium as gym
import numpy as np

class MazeEnv(gym.Env):
    """
    自定义迷宫环境，继承自 gym.Env
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MazeEnv, self).__init__()
        # 定义动作空间和状态空间
        # 动作空间：上、下、左、右
        self.action_space = gym.spaces.Discrete(4)
        # 状态空间：智能体在迷宫中的位置（二维坐标）
        self.maze_size = (5, 5)
        self.observation_space = gym.spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)

        # 定义迷宫（0 表示空地，-1 表示墙壁）
        self.maze = np.zeros(self.maze_size)
        self.maze[0, 3] = -1  # 墙壁位置
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
        """
        执行动作
        """
        # 定义动作对应的移动
        directions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        # 根据动作计算新的位置
        move = directions[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        # 默认的奖励和终止标志
        reward = -1
        done = False

        # 检查新位置是否在迷宫范围内
        if (0 <= new_pos[0] < self.maze_size[0]) and (0 <= new_pos[1] < self.maze_size[1]):
            # 检查新位置是否是墙壁
            if self.maze[new_pos] == -1:
                # 撞到墙壁
                reward = -5
            else:
                # 合法移动
                self.agent_pos = new_pos
        else:
            # 超出迷宫范围
            reward = -5

        # 检查是否到达终点
        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True

        obs = np.array(self.agent_pos)
        info = {}
        return obs, reward, done, info

    def reset(self):
        """
        重置环境到初始状态
        """
        self.agent_pos = self.start_pos
        return np.array(self.agent_pos)

    def render(self, mode='human'):
        """
        渲染迷宫环境
        """
        maze_render = np.copy(self.maze)
        maze_render[self.agent_pos] = 2  # 智能体的位置
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
