import gym
import matplotlib.pyplot as plt
import numpy as np
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

# 2. 智能体
class Agent:

    def __init__(self):
        # 折扣因子 = 1  方便后续计算 G_tau
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
        """采样一条轨迹"""
        state = env.reset()
        states, log_probs, actions = [], [], []
        episode_reward = 0
        done = False

        while not done:
            action, probs = self.get_action(state)              # 获取当前St下的 At 和 概率分布
            next_state, reward, done, _ = env.step(action)      # 模拟采取At后的St+1 Rt

            states.append(state)                    # St
            actions.append(action)                  # At
            log_prob = torch.log(probs)[action]     # 动作对数概率
            log_probs.append(log_prob.item())

            state = next_state                      # St+1
            episode_reward += reward                # 轨迹奖励求和

        normalized_reward = episode_reward / 200.0   # 归一化奖励
        return states, log_probs, actions, normalized_reward

    def calc_advantages_with_grpo(self, trajectories):
        """
        使用一组轨迹计算某条轨迹的组内优势
        :param trajectories:
        :return:组内所有轨迹的相对优势得分列表 list[float]
        """
        # 提取这一组内所有轨迹的全局归一化奖励
        rewards = [r for o, l, a, r in trajectories]

        # 计算当前组的平均奖励
        mean_reward = sum(rewards) / len(rewards)

        # 计算当前组奖励的标准差
        std_reward = np.std(rewards) + 1e-8

        # 对每个奖励做 Z-Score 标准化，得出每条轨迹相对于该组的优势
        advantages = [(r - mean_reward) / std_reward for r in rewards]

        return advantages


    def update(self, trajectories):
        """

        :param trajectories:    list[tuple]=list[(states, log_probs, actions, normalized_reward)]
        :return:
        """
        # 计算当前这组轨迹中，每条轨迹对应的组内相对优势
        advantages = self.calc_advantages_with_grpo(trajectories)

        # 策略网络参数重复更新 20 次（Epoch）
        for step in range(20):
            loss = 0.0
            # 遍历组里面的每一条轨迹和对应的组内优势
            for traj, advantage in zip(trajectories, advantages):
                states, log_probs, actions, _ = traj
                states = torch.tensor(states)
                log_probs = torch.tensor(log_probs).view(-1, 1)
                actions = torch.tensor(actions).view(-1, 1)

                # 用最新的策略网络重新计算这路线中所有动作的对数概率
                new_log_probs = torch.log(self.pi(states).gather(1, actions))

                # 计算新旧策略的概率比值（Importance Sampling Weight）
                ratio = torch.exp(new_log_probs - log_probs)

                # PPO 核心：对重要性采样比值进行裁剪，防止策略更新步长过大
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)

                # 计算单条轨迹的近端策略优化损失值
                traj_loss = torch.mean(
                    -torch.min(ratio * advantage, clipped_ratio * advantage))

                # 累加组内每条轨迹的损失
                loss += traj_loss

            # 对方差规范化后的整组总 Loss 取平均
            loss = loss / len(trajectories)

            # 反向传播并更新策略网络权重
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return None



def train(agent, env):
    G = 5  # 组大小：规定同一个起点（或环境）并行采样 5 条轨迹作为一个组
    trial_num = 0

    # 在这里初始化用于绘图的数据记录列表
    return_list = []
    episode_list = []

    while True:
        # 每个大训练回合内循环执行 20 次组采样与更新
        for episode in range(20):
            trajectories, episode_rewards = [], []

            # 使用旧策略网络，连续采样 G 条完整的完整轨迹凑成一组
            for _ in range(G):
                states, log_probs, actions, normalized_reward = agent.collect_trajectory(env)
                trajectories.append((states, log_probs, actions, normalized_reward))
                episode_rewards.append(normalized_reward * 200)

            agent.update(trajectories)

        # 计算当前这组内所有轨迹的真实平均总奖励
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        trial_num += 1

        # 记录每轮数据
        return_list.append(avg_reward)
        episode_list.append(trial_num)


        print(f"训练回合数：{trial_num}，平均奖励：{avg_reward:.2f}")

        if avg_reward > 199:
            print("满足收敛条件，训练结束！")
            return return_list, episode_list


def plot_loss(episode_list, return_list, filename):
    """绘制奖励图像"""
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0 (GRPO)")
    plt.show()
    f.savefig(filename, bbox_inches="tight")

def main():
    # 采用经典控制环境 CartPole-v1
    env = gym.make("CartPole-v1")

    # 设定随机种子以复现结果
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # 初始化无 Critic 网络的 GRPO 智能体
    grpo_agent = Agent()

    # 开始训练并接收绘图统计数据
    return_list, episode_list = train(grpo_agent, env)

    # 绘制并保存曲线图
    plot_loss(episode_list, return_list, "grpo-performance.pdf")
    env.close()


if __name__ == "__main__":
    main()