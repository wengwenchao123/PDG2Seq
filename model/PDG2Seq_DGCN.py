import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
import time
from collections import OrderedDict

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




class PDG2Seq_GCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, time_dim):
        super(PDG2Seq_GCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k*2+1, dim_in, dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k*2+1,dim_in, dim_out))
        # self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        # self.weights = nn.Parameter(torch.FloatTensor(cheb_k,dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.time_dim = time_dim
        self.gcn = gcn(cheb_k)
        self.fc1 = FC(dim_in, time_dim)
        self.fc2 = FC(dim_in, time_dim)

    def forward(self, x, adj, node_embedding):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]


        x_g = self.gcn(x, adj)

        weights = torch.einsum('nd,dkio->nkio', node_embedding, self.weights_pool)    #[B,N,embed_dim]*[embed_dim,chen_k,dim_in,dim_out] =[B,N,cheb_k,dim_in,dim_out]
                                                                                  #[N, cheb_k, dim_in, dim_out]=[nodes,cheb_k,hidden_size,output_dim]
        bias = torch.matmul(node_embedding, self.bias_pool) #N, dim_out                 #[che_k,nodes,nodes]* [batch,nodes,dim_in]=[B, cheb_k, N, dim_in]

        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias  #b, N, dim_out
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  #b, N, dim_out
        # x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias    #[B,N,cheb_k,dim_in] *[N,cheb_k,dim_in,dim_out] =[B,N,dim_out]

        return x_gconv


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self, x, A):
        # x = torch.einsum("bnm,bmc->bnc", A, x)#[batch_size, D, num_nodes, num_steps]  [N,N]  [batch_size, num_steps, num_nodes, D]
        x = torch.einsum("bnm,bmc->bnc", A,x)  # [batch_size, D, num_nodes, num_steps]  [N,N]  [batch_size, num_steps, num_nodes, D]
        return x.contiguous()

class gcn(nn.Module):
    def __init__(self,k=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        self.k = k

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)                   #先做一次图扩散卷积
            out.append(x1)                         #放入输出列表中
            for k in range(2, self.k + 1):     #在对经过卷积的X1进行多级运算，得到一系列扩散卷积结果，都存入out中
                x2 = self.nconv(x1,a)      #这里的order应该就是进行多少次扩散卷积运算，默认是2，那么range(2, self.order + 1)就是(2,3)也就是算两次就结束了
                out.append(x2)
                x1 = x2
        h = torch.stack(out, dim=1)
        #h = torch.cat(out,dim=1)                   #拼接结果
        return h

