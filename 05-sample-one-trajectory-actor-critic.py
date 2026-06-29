import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# 1. 策略神经网络 (Actor / 演员)
class PolicyNet(nn.Module):
    """策略神经网络的结构"""

    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128) # 推车的速度 位置 木杆的角度 角速度
        self.l2 = nn.Linear(128, action_size)  # 动作的概率分布

    def forward(self, x):       # x 为 St
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x
        # x: [batch_size,feature_dim] -> l1 relu: [batch_size, hidden_size] ->
        # l2: [batch_size, action_logits] ->softmax: [batch_size, action_probs]


# 2. 状态价值神经网络 (Critic / 评论家)
class ValueNet(nn.Module):
    """价值函数神经网络V_ω"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1) # 输出当前状态的价值期望标量 V_w(St)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


# 5. 智能体
class Agent:
    def __init__(self):
        self.gamma = 0.98           # 折扣因子
        self.lr_pi = 0.0002         # 两个网络的学习率
        self.lr_v = 0.005
        self.action_size = 2        # 动作数量: 左 右

        self.pi = PolicyNet(self.action_size)   # 初始化策略网络
        self.v = ValueNet()                     # 初始化状态价值神经网络

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)     # 优化器
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

    def collect_trajectory(self, env):
        """采样一条完整的轨迹"""
        state = env.reset()
        states, next_states, actions, rewards, dones = [], [], [], [], []   # St St+1 At Rt
        done = False

        while not done:
            action, _ = self.get_action(state)  # 随机采样动作
            next_state, reward, done, _ = env.step(action)

            states.append(state)  # St
            next_states.append(next_state)  # St+1
            actions.append(action)  # At
            rewards.append(reward)  # Rt
            dones.append(done)  # done

            state = next_state

        return states, next_states, actions, rewards, dones

    def update(self, trajectory):
        """整条轨迹的actor-critic批量更新"""
        states, next_states, actions, rewards, dones = trajectory

        # 转换为 Tensor，并调整形状为网络接收的 Batch 矩阵形式
        states = torch.tensor(states, dtype=torch.float32)  # [steps, 4]
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)  # [steps, 1]
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)  # [steps, 1]
        next_states = torch.tensor(next_states, dtype=torch.float32)  # [steps, 4]
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)  # [steps, 1]

        v = self.v(states)  # 调用价值估计网络估计  V_w(St)   # [steps, 1]

        # 计算批量 TD 目标：Rt + gamma * V(St+1) * (1 - done) -> [T, 1]
        td_target = rewards + self.gamma * self.v(next_states) * (1 - dones)

        # 价值网络损失：均方误差损失，   V_w(st) 尽可能地逼近阶段性真理目标  td_target
        loss_v = F.mse_loss(v, td_target.detach())

        # 策略网络损失：利用 gather 提取真正执行的动作概率
        action_probs = self.pi(states).gather(1, actions)

        # 1步TD误差
        delta = td_target - v

        # 公式求和部分
        loss_pi = -torch.sum(torch.log(action_probs) * delta.detach())

        # 清空梯度、反向传播、批量并行更新
        self.optimizer_pi.zero_grad()
        self.optimizer_v.zero_grad()

        loss_v.backward()
        loss_pi.backward()

        self.optimizer_pi.step()
        self.optimizer_v.step()

# 2. 奖励曲线绘制函数
def plot_loss(episode_list, return_list, filename):
    """绘制奖励图像"""
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0 (Batch Actor-Critic)")
    plt.show()
    f.savefig(filename, bbox_inches="tight")


# 6. 主训练循环
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = Agent()
    return_list = []
    episode_list = []

    for episode in range(3000):
        # 采样一条完整轨迹
        trajectory = agent.collect_trajectory(env)

        # 批量更新策略网络和价值网络
        agent.update(trajectory)

        # 记录每轮总奖励
        total_reward = sum(trajectory[3])   # rewards 列表
        return_list.append(total_reward)
        episode_list.append(episode)

        if episode % 100 == 0:
            print(f"回合：{episode}, 总奖励：{total_reward:.1f}")

    # 训练结束，保存曲线
    plot_loss(episode_list, return_list, "sample-one-trajectory-actor-critic-loss.pdf")