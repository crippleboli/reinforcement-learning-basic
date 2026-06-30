import matplotlib.pyplot as plt
import gym
import random
import math


class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            data=self.data + other.data,
            children=(self, other),
            local_grads=(1, 1)
        )

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            data=self.data * other.data,
            children=(self, other),
            local_grads=(other.data, self.data)
        )

    def __pow__(self, other):
        return Value(
            data=self.data ** other,
            children=(self,),
            local_grads=(other * self.data ** (other - 1),)
        )

    def log(self):
        return Value(
            data=math.log(self.data),
            children=(self,),
            local_grads=(1/self.data,)
        )

    def exp(self):
        return Value(
            data=math.exp(self.data),
            children=(self,),
            local_grads=(math.exp(self.data),)
        )

    def relu(self):
        return Value(
            data=max(0, self.data),
            children=(self,),
            local_grads=(float(self.data > 0),)
        )

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        topo = []  # 使用拓扑排序遍历有向无环图（DAG）
        visited = set()  # 记录已经访问过的节点

        def build_topo(v):
            """从当前节点self开始构建计算图"""
            if v not in visited:
                visited.add(v)  # 如果节点v没有被访问过，那么添加到visited集合中
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1  # dL/dL = 1
        for v in reversed(topo):
            # 链式求导+梯度累加
            for child, local_grad in zip(v._children, v._local_grads):
                # `+=`是梯度累加
                # `*`链式求导法则
                child.grad += local_grad * v.grad


def matrix(nout, nin, std=0.08):
    # 构造一个(nout, nin)的矩阵
    return [
        [Value(random.gauss(0, std)) for _ in range(nin)]
        for _ in range(nout)
    ]


def linear(x, w):
    # Y = XW = x_1 * w_1 + x_2 * w_2 + ...
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


# mlp的参数
state_dict = {
    "w1": matrix(128, 4),
    "w2": matrix(2, 128),
}


def policy(state):
    state = state.tolist()
    state = [Value(s) for s in state]

    x = linear(state, state_dict["w1"])
    x = [xi.relu() for xi in x]
    x = linear(x, state_dict["w2"])
    probs = softmax(x)

    return probs


env = gym.make("CartPole-v0")


def rollout(env):
    """采样一条轨迹"""
    state = env.reset()
    done = False
    states, actions, rewards = [], [], []

    while not done:
        probs = policy(state)
        action = random.choices([0, 1], weights=[p.data for p in probs])[0]
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    return states, actions, rewards


# 将所有参数展平成一个列表
params = [p for mat in state_dict.values() for row in mat for p in row]

episodes = []
returns = []

gamma = 0.98


def train():
    # Adam优化器用到的
    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m = [0.0] * len(params)  # 一阶矩
    v = [0.0] * len(params)  # 二阶矩

    num_steps = 3000
    for step in range(1, num_steps + 1):
        # 采样轨迹
        states, actions, rewards = rollout(env)
        # G(τ)
        G = 0.0
        for r in rewards[::-1]:
            G = r + gamma * G

        loss = 0.0
        for s, a in zip(states, actions):
            probs = policy(s)
            log_action_prob = probs[a].log()
            loss += (- G * log_action_prob)

        loss.backward()

        # Adam优化器+学习率的线性衰减
        # optimizer.step()
        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2

            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            # 更新神经网络的参数
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            # .zero_grad()
            p.grad = 0

        # 统计信息
        episodes.append(step)
        returns.append(sum(rewards))
        if step % 100 == 0:
            print(f"Step: {step}, Return: {sum(rewards)}")


train()


def plot_loss(episodes, returns, filename):
    f = plt.figure()
    plt.plot(episodes, returns)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("Policy Gradient Method")
    plt.show()
    f.savefig(filename, bbox_inches="tight")


plot_loss(episodes, returns, "micropg-loss.pdf")