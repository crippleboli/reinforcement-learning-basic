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
        self.lr_pi = 0.001  # 策略网络学习率
        self.lr_v = 0.02  # 价值网络学习率
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
        states, next_states, actions, action_probs, rewards, dones = [], [], [], [], [], []  # St St+1 At Probs Rt
        done = False

        while not done:
            action, probs = self.get_action(state)  # 随机采样动作
            next_state, reward, done, _ = env.step(action)

            states.append(state)  # St
            next_states.append(next_state)  # St+1
            actions.append(action)  # At
            action_probs.append(probs[action])  # Probs 在St情况下执行At的概率
            rewards.append(reward)  # Rt
            dones.append(done)  # done

            state = next_state

        return states, next_states, actions, action_probs, rewards, dones

    def update(self, trajectory):
        """整条轨迹的 PPO 批量更新"""
        states, next_states, actions, action_probs, rewards, dones = trajectory

        # 转换为 Tensor 并调整为 Batch 矩阵形状
        states = torch.tensor(states, dtype=torch.float32)  # [s0, s1, ..., sT-1]
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)  # [a0, a1, ..., aT-1]
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)  # [R0, R1, ..., RT-1]
        next_states = torch.tensor(next_states, dtype=torch.float32)  # [s1, s2, ..., sT]
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)  # [False1, False2, ..., TrueT]

        # 评估当前策略的心理预期值 V_w(St)，使用价值神经网络进行估计，并阻断梯度
        V_w = self.v(states).detach()  # [V_w(s0), V_w(s1), ..., V_w(sT-1)]

        # 计算 1步时序差分目标：TD-target_t = Rt + gamma * V(s_t+1) * (1 - done)
        td_target = rewards + self.gamma * self.v(next_states) * (1 - dones)

        # 计算 1步TD误差：delta_t = TD-target_t - V(s_t)
        td_delta = td_target - V_w

        # 计算每个时刻 t 的广义优势估计GAE
        gae = compute_gae(self.gamma, td_delta.cpu()).view(-1, 1)  # .cpu() 函数内部使用到 numpy

        # 冻结一份旧策略采取动作的对数概率 [probs0, probs1, ..., probsT-1]
        old_probs = torch.tensor(action_probs, dtype=torch.float32).view(-1, 1)
        old_log_probs = torch.log(old_probs).detach()

        # 每条轨迹内层循环复习 10 次
        for _ in range(10):
            # 新策略采取动作的对数概率
            log_probs = torch.log(self.pi(states).gather(1, actions))

            # 指数运算计算新旧概率比值：ratio = exp(log_new - log_old) = new / old
            ratio = torch.exp(log_probs - old_log_probs)

            # 未裁剪的目标值：ratio * A
            surr1 = ratio * gae

            # 裁剪后的目标值：clip(ratio, 1 - epsilon, 1 + epsilon) * A
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * gae

            # 计算策略网络损失
            pi_loss = torch.mean(-torch.min(surr1, surr2))

            # 计算价值网络损失
            v_loss = torch.mean(F.mse_loss(self.v(states), gae + V_w))

            # 梯度清零、反向传播、参数更新
            self.optimizer_pi.zero_grad()
            self.optimizer_v.zero_grad()
            pi_loss.backward()
            v_loss.backward()
            self.optimizer_pi.step()
            self.optimizer_v.step()




# 3. 计算广义优势估计
def compute_gae(gamma, td_delta):
    """计算每个时刻 t 的广义优势估计（GAE）"""
    # 阻断梯度并转为 numpy 数组，方便进行一维切片和反向遍历
    device = td_delta.device  # 记录原始设备
    td_delta = td_delta.detach().numpy()
    gae_list = []
    last_gae = 0.0
    lmbda = 0.95  # 广义优势估计的平滑调节系数 lambda

    # 公式：A_t = delta_t + gamma * lambda * A_{t+1}
    for delta in td_delta[::-1]:    # [::-1] 代表将数组倒序，
        last_gae = gamma * lmbda * last_gae + delta
        gae_list.append(last_gae)

    # [A_0, ..., A_T-1]
    gae_list.reverse()

    return torch.tensor(gae_list, dtype=torch.float32).to(device)



# 4. 奖励曲线绘制函数
def plot_loss(episode_list, return_list, filename):
    """绘制奖励图像"""
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0 (Batch Actor-Critic)")
    plt.show()
    f.savefig(filename, bbox_inches="tight")


# 5. 训练函数
def train(env, agent):
    """PPO 训练主循环"""
    return_list = []
    episode_list = []

    # 外层循环：控制总共采集多少局（Episodes）游戏数据
    for episode in range(500):
        # 1. 采集一条完整的轨迹数据
        trajectory = agent.collect_trajectory(env)

        # 2. 拿着这条轨迹去内层循环里复习 10 次并更新网络
        agent.update(trajectory)

        # 3. 统计信息：trajectory[4] 对应 rewards 列表
        episode_reward = sum(trajectory[4])
        return_list.append(episode_reward)
        episode_list.append(episode)

        # 每 10 个回合打印一次日志
        if (episode + 1) % 10 == 0:
            print(f"回合：{episode + 1}, 回报：{episode_reward}")

    return return_list, episode_list


def main():
    # 初始化经典控制环境：平衡车
    env = gym.make("CartPole-v0")
    env.seed(0)
    torch.manual_seed(0)

    # 初始化 PPO 智能体
    agent = Agent()

    # 开始训练
    return_list, episode_list = train(env, agent)

    # 绘制并保存奖励曲线图
    plot_loss(episode_list, return_list, "ppo-loss.pdf")



if __name__ == "__main__":
    main()