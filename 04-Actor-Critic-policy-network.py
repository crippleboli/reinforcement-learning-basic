import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


# 1. 策略神经网络 (Actor / 演员)
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



# 2. 状态价值神经网络 (Critic / 评论家)
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)  # 输出当前状态的价值期望标量 vw st

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


# 3. 智能体 (Actor-Critic)
class Agent:
    def __init__(self):
        self.gamma = 0.98  # 折扣因子
        self.lr_pi = 0.0002  # 策略网络学习率
        self.lr_v = 0.0005  # 价值网络学习率
        self.action_size = 2    # 两个动作（向左或向右）

        # 初始化演员 和 评论家
        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
        # 各自优化器
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)


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

    def update(self, state, action_prob, reward, next_state, done):
        """
        单步更新：每走一步，直接代入当前步骤和下一步的状态与奖励进行训练
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)            # [4] -> [1, 4]
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # [4] -> [1, 4]

        # TD目标 = Rt + gamma * V(St+1)
        target = reward + self.gamma * self.v(next_state_tensor) * (1 - done)           # 最后一步 done = 1时 V(St+1) = 0
        v = self.v(state_tensor)  # V(St)

        loss_fn = nn.MSELoss()
        loss_v = loss_fn(v, target.detach())  # 目标 target 需要剥离计算图

        # 计算 PolicyNet (Actor) 的策略梯度损失
        delta = target - v                    # 1步TD误差
        loss_pi = -torch.log(action_prob) * delta.detach().item()

        # 清空梯度、反向传播、更新参数
        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()

        loss_v.backward()
        loss_pi.backward()

        self.optimizer_v.step()
        self.optimizer_pi.step()


# 4. 创建环境与主训练循环
env = gym.make("CartPole-v0")
agent = Agent()
return_list = []
episode_list = []

for episode in range(2000):  # 2000 轮
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, probs = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        # 每走完一步，立刻用单步数据 实时训练
        agent.update(state, probs[action], reward, next_state, done)

        state = next_state
        total_reward += reward

    return_list.append(total_reward)
    episode_list.append(episode)

    if episode % 100 == 0:
        print("回合:{}, 总奖励:{:.1f}".format(episode, total_reward))


# 5. 绘图展示
def plot_loss(episode_list, return_list, filename):
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("Actor-Critic on CartPole-v0")
    plt.show()
    f.savefig(filename, bbox_inches="tight")


plot_loss(episode_list, return_list, "actor-critic-pg-loss.pdf")