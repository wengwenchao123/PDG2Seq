import torch
import torch.nn as nn
from model.PDG2Seq_DGCN import PDG2Seq_GCN
from collections import OrderedDict
import torch.nn.functional as F
class FC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FC, self).__init__()
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.mlp=nn.Sequential( #疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
                OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.middle_dim, dim_out))]))

    def forward(self, x):

        ho = self.mlp(x)

        return ho

class PDG2SeqCell(nn.Module):  #这个模块只进行GRU内部的更新，所以需要修改的是AGCN里面的东西
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim):
        super(PDG2SeqCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = PDG2Seq_GCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim, time_dim)
        self.update = PDG2Seq_GCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim, time_dim)
        self.fc1 = FC(dim_in + self.hidden_dim, time_dim)
        self.fc2 = FC(dim_in + self.hidden_dim, time_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        filter1 = self.fc1(input_and_state)
        filter2 = self.fc2(input_and_state)

        nodevec1 = torch.tanh(torch.einsum('bd,bnd->bnd', node_embeddings[0], filter1)) #[B,N,dim_in]
        nodevec2 = torch.tanh(torch.einsum('bd,bnd->bnd', node_embeddings[1], filter2))  # [B,N,dim_in]


        adj = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(
            nodevec2, nodevec1.transpose(2, 1))

        adj1 = PDG2SeqCell.preprocessing(F.relu(adj))
        adj2 = PDG2SeqCell.preprocessing(F.relu(-adj.transpose(-2, -1)))


        adj = [adj1, adj2]


        z_r = torch.sigmoid(self.gate(input_and_state, adj, node_embeddings[2]))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, adj, node_embeddings[2]))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

    def preprocessing(adj):               #处理动态矩阵可能不含有对角线元素的问题
        num_nodes= adj.shape[-1]
        adj = adj +  torch.eye(num_nodes).to(adj.device)
        x= torch.unsqueeze(adj.sum(-1), -1)
        adj = adj / x   # D = torch.diag_embed(torch.sum(adj, dim=-1) ** (-1))  adj =torch.matmul(D, adj)
        return adj