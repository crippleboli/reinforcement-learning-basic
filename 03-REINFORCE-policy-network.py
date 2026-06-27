import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# 1. 策略神经网络
class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128) # 推车的速度 位置 木杆的角度 角速度
        self.l2 = nn.Linear(128, action_size)  # 动作的概率分布

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x
        # x: [batch_size,feature_dim] -> l1 relu: [batch_size, hidden_size] ->
        # l2: [batch_size, action_logits] ->softmax: [batch_size, action_probs]

# 2. 智能体
class Agent:

    def __init__(self):
        self.gamma = 0.98  # 折扣因子
        self.lr = 0.0002  # 学习率
        self.action_size = 2  # 两个动作（向左或向右）

        self.pi = PolicyNet(self.action_size)  # 初始化策略网络
        self.optimizer = optim.Adam(
            self.pi.parameters(), lr=self.lr
        )  # 优化器

    def get_action(self, state):
        """
        :param state:   [feature_dim]
        :return:    action: action_index   probs:[action_probs]
        """
        # 维度转换: [4] -> [1, 4] -> 神经网络 -> [1, 2] -> [2]
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 转为张量 增加 batch_size = 1维度 [1,feature_dims]
        probs = self.pi(state_tensor).squeeze(0)         # 输入神经网络  输入后去掉  batch_size = 1维度 [action_probs]

        # 根据动作的概率分布创建一个分类分布
        m = Categorical(probs)

        # 按照 [action_probs] 概率分布 随机采样动作: action_index
        action = m.sample().item()

        # 返回采样的动作 动作概率分布
        return action, probs

    def collect_trajectory(self, env):
        """
        采集随机轨迹
        :param env:
        :return:
        """
        state = env.reset()
        states, actions, rewards = [], [], []
        done = False

        while not done:
            action, _ = self.get_action(state)
            # 物理引擎模拟采取action后的 下个状态 奖励 是否接受 和 额外信息
            next_state, reward, done, _ = env.step(action)

            states.append(state)    # St
            actions.append(action)  # At
            rewards.append(reward)  # Rt

            state = next_state  # St -> St+1

        return states, actions, rewards

    '''
    def update(self, trajectory):
        """
        根据历史轨迹 奖励 进行更新策略网络
        :param trajectory: tuple(states, actions, rewards)
        :return:
        """
        states, actions, rewards = trajectory

        # 逆序计算 G
        G = 0
        for r in rewards[::-1]:
            G = r + self.gamma * G


        states = torch.tensor(states, dtype=torch.float32)              # [batch_size, feature_dims = 4]
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)   # [action_nums, 1]
                                                                        # 1: 按列 actions: 索引
        log_probs = torch.log(self.pi(states).gather(1, actions))       # [batch_size, action_probs = 2]
        loss = -torch.sum(log_probs) * G

        self.optimizer.zero_grad()
        loss.backward()  # 反向传播求导
        self.optimizer.step()  # 梯度下降更新参数   
    '''

    def update(self, trajectory):
        """
        全局的 G-tau换成了每个时刻实时的 G-t
        :param trajectory: tuple(states, actions, rewards)
        :return:
        """
        states, actions, rewards = trajectory
        G, loss = 0, 0
        for r, s, a in zip(rewards[::-1], states[::-1], actions[::-1]):
            G = r + self.gamma * G  # Gt = Rt + gamma * Gt+1
            probs = self.pi(torch.tensor(s).unsqueeze(0)).squeeze(0)
            log_prob = torch.log(probs)[a]  # log pi_theta(At|St)
            loss += -log_prob * G  # -sum(Gt * log pi_theta(At|St))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 3. 创建环境 多轮训练
env = gym.make("CartPole-v0")
agent = Agent()
return_list = []
episode_list = []

for episode in range(3000):
    trajectory = agent.collect_trajectory(env)
    reward_list = trajectory[2]
    return_list.append(sum(reward_list))
    episode_list.append(episode)

    agent.update(trajectory)

    if episode % 100 == 0:
        print("回合:{}, 总奖励:{:.1f}".format(episode, sum(reward_list)))

# 4. 备份绘图
def plot_loss(episode_list, return_list, filename):
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    plt.show()
    f.savefig(filename, bbox_inches="tight")

plot_loss(episode_list, return_list, "reinforce-loss.pdf")