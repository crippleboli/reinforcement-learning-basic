# ==================== FILE: 01-cartpole.py ====================
# 完整的倒立摆模拟 + 动画播放 (修复版)

import gym
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

print("=" * 60)
print("倒立摆模拟程序")
print("=" * 60)

# ==================== 第一部分：环境测试 ====================
print("\n[1] 环境测试...")
print(f"Gym 版本: {gym.__version__}")

# 创建倒立摆环境（使用 rgb_array 模式捕获帧）
env = gym.make("CartPole-v1", render_mode="rgb_array")

# 打印初始状态和动作空间
state = env.reset()
print(f"初始状态: {state}")
print(f"动作空间: {env.action_space}")

# 测试执行一步动作
action = 0
next_state, reward, done, info = env.step(action)
print(f"下一步状态: {next_state}")
print("-" * 60)

# ==================== 第二部分：运行随机智能体（捕获帧） ====================
print("\n[2] 运行随机智能体...")

state = env.reset()
done = False
episode_rewards = []
frames = []  # 存储每一帧画面
gamma = 0.95
step_count = 0
total_reward_discounted = 0

while not done and step_count < 200:
    # 捕获当前帧
    frame = env.render()

    # 修复：如果帧数据多了一个维度，去掉它
    # 当 frame.shape = (1, height, width, 3) 时，取 frame[0]
    if hasattr(frame, 'shape') and len(frame.shape) == 4 and frame.shape[0] == 1:
        frame = frame[0]
    # 如果 frame 是列表且第一个元素是数组，也处理
    elif isinstance(frame, list) and len(frame) > 0 and hasattr(frame[0], 'shape'):
        if len(frame[0].shape) == 3:
            frame = frame[0]

    frames.append(frame)

    # 随机选择动作 (0: 左推, 1: 右推)
    action = random.choice([0, 1])

    # 执行动作
    next_state, reward, done, _ = env.step(action)
    episode_rewards.append(reward)
    state = next_state
    step_count += 1

    # 实时显示进度
    if step_count % 10 == 0:
        print(f"  步数: {step_count}, 当前奖励: {reward}")

# 计算折扣回报
for r in episode_rewards[::-1]:
    total_reward_discounted = r + gamma * total_reward_discounted

print(f"\n[3] 模拟结果:")
print(f"  总步数: {step_count}")
print(f"  每步奖励: {episode_rewards}")
print(f"  折扣回报 (γ={gamma}): {total_reward_discounted:.4f}")
print(f"  未折扣总奖励: {sum(episode_rewards)}")
print(f"  捕获帧数: {len(frames)}")

env.close()
print("-" * 60)

# ==================== 第三部分：播放动画（matplotlib窗口） ====================
print("\n[4] 开始播放动画...")

if frames:
    try:
        # 检查第一帧的形状
        first_frame = frames[0]
        print(f"  帧形状: {first_frame.shape}")

        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis('off')  # 隐藏坐标轴

        # 显示第一帧
        im = ax.imshow(first_frame)


        # 更新函数：每一帧更新图像
        def animate(i):
            im.set_array(frames[i])
            return [im]


        # 创建动画
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(frames),
            interval=50,  # 每帧间隔50毫秒 (20fps)
            blit=True,
            repeat=True  # 循环播放
        )

        # 设置标题
        plt.title(f"CartPole 随机策略 - 维持了 {step_count} 步", fontsize=14)
        plt.tight_layout()

        print("  动画窗口已打开，关闭窗口继续...")

        # 显示动画（会阻塞程序，直到关闭窗口）
        plt.show()

        print("  动画播放完成")

        # ==================== 第四部分：保存动画为GIF（可选） ====================
        print("\n[5] 保存动画（可选）...")
        save_gif = input("  是否保存为 GIF？(y/n): ").strip().lower()

        if save_gif == 'y':
            try:
                print("  正在保存动画为 GIF...")
                anim.save('cartpole_random.gif', writer='pillow', fps=20)
                print("  ✅ 已保存为 cartpole_random.gif")
            except Exception as e:
                print(f"  ❌ 保存失败: {e}")
                print("  提示: 请安装 pillow: pip install pillow")
        else:
            print("  跳过保存")

    except Exception as e:
        print(f"  ❌ 播放动画失败: {e}")
        print("  尝试使用备用播放方式...")

        # 备用方案：逐帧显示
        print("  使用逐帧显示模式（按任意键切换）...")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis('off')

        for i, frame in enumerate(frames):
            ax.imshow(frame)
            ax.set_title(f"帧 {i + 1}/{len(frames)}")
            plt.pause(0.05)
            plt.clf()
        plt.close()

else:
    print("❌ 没有捕获到帧，无法播放动画")

print("\n" + "=" * 60)
print("程序运行完成！")
print("=" * 60)