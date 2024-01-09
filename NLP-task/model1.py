import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import random
from sklearn import metrics

class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data = nn.init.constant_(self.bias.data, 0.0)
            # init.xavier_uniform_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


class GCN(nn.Module):
    def __init__(self, in_dim=48, out_dim=48, neg_penalty=0.2):
        super(GCN, self).__init__()
        self.in_dim = in_dim  # 输入的维度
        self.out_dim = out_dim  # 输出的维度
        self.neg_penalty = neg_penalty  # 负值
        self.kernel = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.kaiming_normal_(self.kernel)
        # init.uniform_(weight, -stdv, stdv)
        # nn.init.zeros_(layer.bias)
        self.c = 0.85
        self.losses = []

    def forward(self, x, adj):
        # GCN-node
        feature_dim = int(adj.shape[-1])
        eye = torch.eye(feature_dim).cuda()  # 生成对角矩阵 feature_dim * feature_dim
        if x is None:  # 如果没有初始特征
            AXW = torch.tensordot(adj, self.kernel, [[-1], [0]])  # batch_size * num_node * feature_dim
        else:
            XW = torch.tensordot(x, self.kernel, [[-1], [0]])  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim
        # I_cAXW = eye+self.c*AXW
        I_cAXW = self.c * AXW
        # y_relu = torch.nn.functional.relu(I_cAXW)
        # temp = torch.mean(input=y_relu, dim=-2, keepdim=True) + 1e-6
        # col_mean = temp.repeat([1, feature_dim, 1])
        # y_norm = torch.divide(y_relu, col_mean)  # 正则化后的值
        # output = torch.nn.functional.softplus(y_norm)
        # print(output)
        # output = y_relu
        # 做个尝试
        if self.neg_penalty != 0:
            neg_loss = torch.multiply(torch.tensor(self.neg_penalty),
                                      torch.sum(torch.nn.functional.relu(1e-6 - self.kernel)))
            self.losses.append(neg_loss)
        # print(I_cAXW)
        return I_cAXW


class E2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))
        self.nodes = 78

    def forward(self, A):
        #         print(A.shape)
        A = A.view(-1, self.in_channel, self.nodes, self.nodes)

        a = self.conv1xd(A)
        b = self.convdx1(A)

        concat1 = torch.cat([a] * self.d, 2)
        concat2 = torch.cat([b] * self.d, 3)

        # A = torch.mean(concat1+concat2, 1)
        # print('e2e', (concat1+concat2).shape)
        return concat1 + concat2

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nums_train = np.ones(116) # 制作mask模板
# nums_train[:78] = 0 # 根据设置的nodes number 决定多少是mask 即mask比例 # 写错了 应该是[:nodes]
Mask_train = nums_train.reshape(nums_train.shape[0], 1) * nums_train # 116 116
for i in range(78):
    Mask_train[i][:78] = 0
# np.repeat(Mask_train, X_train.shape[0], 0)
Mask_train_tensor = torch.from_numpy(Mask_train).float().to(device)
# Mask_train_tensor = tf.cast(Mask_train_tensor, tf.float32)

class Model(nn.Module):
    def __init__(self, dropout=0.5, num_class=2, nodes=78):
        super().__init__()

        self.e2e = nn.Sequential(
            E2E(1, 8, (nodes, nodes)),
            nn.LeakyReLU(0.33),
            E2E(8, 8, (nodes, nodes)),  # 0.642
            nn.LeakyReLU(0.33),
        )

        self.e2n = nn.Sequential(
            nn.Conv2d(8, 48, (1, nodes)),  # 32 652
            nn.LeakyReLU(0.33),
        )

        self.n2g = nn.Sequential(
            nn.Conv2d(48, nodes, (nodes, 1)),
            nn.LeakyReLU(0.33),
        )

        self.linear = nn.Sequential(
            nn.Linear(nodes, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(64, 10),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(10, num_class)
        )

        self.GC = DenseGCNConv(48, 48)

        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.GCN = GCN()

    def MaskAutoEncoder(self, e2n, A, masked_x):  # masked_x 32 50 48
        e2n_encoder = torch.squeeze(e2n)
        # print('e2n_encoder ', e2n_encoder.shape) # 16 116
        # print('masked_x ', masked_x.shape)
        masked_x = masked_x.permute(0, 2, 1)  # 32 48 50
        e2n_encoder = torch.cat((e2n_encoder, masked_x), -1)  # 补上了masked
        # print('e2n_encoder ', e2n_encoder.shape) # 32 48 116
        e2n_encoder_T = e2n_encoder.permute(0, 2, 1)  # batch 116 48
        # print(temp.shape)
        # print('A ', A.shape) # 116 116
        # print(A[0])
        # e2n_encoder_T = self.GCN(e2n_encoder_T, A)
        e2n_encoder_T = self.GC(e2n_encoder_T, A)
        e2n_encoder = e2n_encoder_T.permute(0, 2, 1)
        Z = torch.matmul(e2n_encoder_T, e2n_encoder)  # batch 116 116
        # print('Z ', Z.shape)
        # Z = nn.sigmoid(Z) # 正相关 负相关分离
        # 哈达姆乘
        Z = Z * Mask_train_tensor
        # print(Mask_train_tensor)
        # print(Z[0][199])

        return Z
        # Z = K.expand_dims(Z, axis=-1)

    def forward(self, x, A, masked_x):
        # print('input', x.shape)
        x = self.e2e(x)
        x = self.e2n(x)
        # print('e2n', x.shape) # batch 16 116 1
        z = self.MaskAutoEncoder(x, A, masked_x)
        x = self.n2g(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        #print('output', x.shape)

        return x, z

    def get_A(self, x):
        x = self.e2e(x)
        #         print(x.shape)
        x = torch.mean(x, dim=1)
        return x