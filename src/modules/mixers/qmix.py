import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args # 包含所有的参数
        self.n_agents = args.n_agents # 智能体的数量
        self.state_dim = int(np.prod(args.state_shape)) # 智能体状态的维度（通过状态形状的乘积计算得到）

        self.embed_dim = args.mixing_embed_dim # 嵌入维度
        self.abs = getattr(self.args, 'abs', True) # 是否使用绝对值

        # 根据 hypernetworks 的层数设置不同的网络结构，注意是 hypernetworks 的层数
        if getattr(args, "hypernet_layers", 1) == 1:
            # 如果 hypernetworks 层数只有1层，就用一个线性层
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            # 如果 hypernetworks 层数有两层，使用两层线性层和 RELU 激活函数
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        # 如果 hypernetworks 层数超过2层或者是其他情况，报错
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer 第一层隐藏层的状态依赖偏置
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers 最后一层的状态依赖偏置的V(s)函数
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))
        

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0) # 返回张量在第一个维度（即批量维度，batch dimension）上的大小
        states = states.reshape(-1, self.state_dim) # 假设原始 states 张量的形状为 [batch_size, ...]，将 states 张量展平为 [batch_size, self.state_dim] 形状的二维张量
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents) # 假设原始 agent_qs 张量的形状为 [batch_size, n_agents]，将 agent_qs 张量展平为 [batch_size, 1, n_agents] 形状的三维张量
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        # th.bmm 是 PyTorch 中的批次矩阵乘法 (batch matrix multiplication) 函数。
        # agent_qs 的形状是 [batch_size, 1, n_agents]。
        # w1 的形状是 [batch_size, n_agents, embed_dim]。
        # th.bmm(agent_qs, w1) 的结果是 [batch_size, 1, embed_dim]。
        # 这是因为对于每个批次样本，我们计算形状为 [1, n_agents] 的 agent_qs 和形状为 [n_agents, embed_dim] 的 w1 之间的矩阵乘法，结果是形状为 [1, embed_dim] 的矩阵。
        hidden = F.elu(th.bmm(agent_qs, w1) + b1) # F.elu 是 PyTorch 中的 ELU (Exponential Linear Unit) 激活函数。
        # ELU(x)= x when x>0 and \alpha *(\exp (x)-1) when x<=0
        
        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        
        return q_tot

    # Q_tot = w2( w1Q + b1) + v = w2w1Q + w2b1 + v
    # k = w2w1
    def k(self, states):
        # 通过将两个权重矩阵 w1 和 w_final 相乘，并进行归一化操作，
        # k 函数生成了一个表示各个智能体 Q 值与最终组合 Q 值之间关系的权重矩阵。
        # 这个权重矩阵的每一行和为1，表示在组合最终 Q 值时，各个智能体 Q 值的相对重要性。
        # 通过动态调整权重矩阵，可以更加灵活地结合各个智能体的 Q 值，从而提高策略的表现。
        bs = states.size(0) # 批量维度，batch dimension
        w1 = th.abs(self.hyper_w_1(states))
        w_final = th.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        k = th.bmm(w1,w_final).view(bs, -1, self.n_agents)
        k = k / th.sum(k, dim=2, keepdim=True)
        return k

    # b = w2b1 + v
    def b(self, states):
        # 这个函数的目的是生成一个与状态 states 相关的偏置项，
        # 经过计算后与权重矩阵相加，最后得到调整后的输出。
        bs = states.size(0)
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(states).view(-1, 1, 1)
        b = th.bmm(b1, w_final) + v
        return b
