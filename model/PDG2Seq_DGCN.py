'''
这个例子对应最后的x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
简单来说就是多项式每项与权重相乘后相加
x=np.random.randint(0, high=100, size=[4,6,2,8])
x=torch.from_numpy(x)
y=np.random.randint(0, high=100, size=[6,2,8,16])
#y=torch.randn(6,2,8,16)
y=torch.from_numpy(y)
out=torch.einsum('bnki,nkio->bno', x, y)
out1=torch.einsum('bni,nio->bno', x[:,:,0,:],y[:,0,:,:])+torch.einsum('bni,nio->bno', x[:,:,1,:],y[:,1,:,:])
print(np.allclose(out.numpy(), out1.numpy()))
'''

'''
就是d这个维度的元素相乘相加
weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)

x=np.random.randint(0, high=100, size=[4,5])
x=torch.from_numpy(x)
y=np.random.randint(0, high=100, size=[5,2,6,8])
#y=torch.randn(6,2,8,16)
y=torch.from_numpy(y)
weights1 = torch.einsum('nd,dkio->nkio', x, y)[:,:,1,1]
weights2 = torch.einsum('nd,dk->nk', x, y[:,:,1,1])
print(np.allclose(weights1.numpy(), weights2.numpy()))
'''

'''
x=np.random.randint(0, high=100, size=[2,5,5])
x=torch.from_numpy(x)
y=np.random.randint(0, high=100, size=[6,5,8])
y=torch.from_numpy(y)
x_g = torch.einsum("knm,bmc->bknc", x, y)
x1= torch.einsum("nm,bmc->bnc", x[0], y)+torch.einsum("nm,bmc->bnc", x[1], y)
x2=x_g[:,0,:,:]+x_g[:,1,:,:]
print(np.allclose(x1.numpy(), x2.numpy()))
'''

'''
def get_laplacian(graph, I, normalize=True):
    """
    return the laplacian of the graph.

    :param graph: the graph structure without self loop, [N, N].
    :param normalize: whether to used the normalized laplacian.
    :return: graph laplacian.
    """
    if normalize:
        D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
        L = I - torch.matmul(torch.matmul(D, graph), D)
    else:
        D = torch.diag(torch.sum(graph, dim=-1))
        L = D - graph
    return L

supports1 = torch.eye(10).cuda()

D1 = torch.ones([10,1])
D2 = torch.ones([1,10])
x = F.relu(torch.randn([10,10,10]).cuda())
x1 = torch.sum(x, dim=-1) ** (-1 / 2)
x2 = torch.diag_embed(x1)

y = get_laplacian(x, supports1)

#D = torch.einsum('nm,mc->nc', D1, D2)
print(y)

'''
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

        # x_g = torch.einsum("knm,bmc->bknc", supports, x)

        # weights = torch.einsum('bnd,dkio->bnkio', nodevec, self.weights_pool)

        weights = torch.einsum('nd,dkio->nkio', node_embedding, self.weights_pool)    #[B,N,embed_dim]*[embed_dim,chen_k,dim_in,dim_out] =[B,N,cheb_k,dim_in,dim_out]
                                                                                  #[N, cheb_k, dim_in, dim_out]=[nodes,cheb_k,hidden_size,output_dim]
        bias = torch.matmul(node_embedding, self.bias_pool) #N, dim_out                 #[che_k,nodes,nodes]* [batch,nodes,dim_in]=[B, cheb_k, N, dim_in]

        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias  #b, N, dim_out
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  #b, N, dim_out
        # x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias    #[B,N,cheb_k,dim_in] *[N,cheb_k,dim_in,dim_out] =[B,N,dim_out]

        return x_gconv

    # def preprocessing(graph):               #处理动态矩阵可能不含有对角线元素的问题
    #     # graph = graph + torch.eye(self.num_nodes).to(self.device)
    #     D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1))
    #     # x= torch.unsqueeze(adj.sum(-1), -1)
    #     # adj = adj / x
    #     L = torch.matmul(D, graph)
    #     return L
    @staticmethod
    def graph_directed2(graph, k):
        num_nodes = graph.shape[-1]
        graph = graph + torch.eye(num_nodes).to(graph.device)
        s1, t1 = graph.topk(k, dim=2)  # 注意这里的topk(k,dim=n)和下面的mask.scatter_(dim=k,t1,s1.fill_(1))中k要注意，不然出现nan值
        adj = PDG2Seq_GCN.preprocessing2(s1)
        return [adj,t1]

    @staticmethod
    def preprocessing2(adj):               #处理动态矩阵可能不含有对角线元素的问题
        x= torch.unsqueeze(adj.sum(-1), -1)
        adj = adj / x   # D = torch.diag_embed(torch.sum(adj, dim=-1) ** (-1))  adj =torch.matmul(D, adj)
        return adj
    '稀疏归一化'
    @staticmethod
    def graph_directed(graph, k):
        num_nodes = graph.shape[-1]
        graph = graph + torch.eye(num_nodes).to(graph.device)
        mask = torch.zeros_like(graph)
        s1, t1 = graph.topk(k, dim=2)  # 注意这里的topk(k,dim=n)和下面的mask.scatter_(dim=k,t1,s1.fill_(1))中k要注意，不然出现nan值
        mask.scatter_(2, t1, s1.fill_(1))
        zeros = torch.zeros_like(graph)
        adj = torch.where(mask > 0, graph, zeros)
        adj = PDG2Seq_GCN.preprocessing1(adj)
        return adj

    @staticmethod
    def preprocessing1(adj):               #处理动态矩阵可能不含有对角线元素的问题
        x= torch.unsqueeze(adj.sum(-1), -1)  #按行归一化
        adj = adj / x   # D = torch.diag_embed(torch.sum(adj, dim=-1) ** (-1))  adj =torch.matmul(D, adj)
        return adj


    '正常的归一化'
    @staticmethod
    def preprocessing(adj):               #处理动态矩阵可能不含有对角线元素的问题
        num_nodes= adj.shape[-1]
        adj = adj +  torch.eye(num_nodes).to(adj.device)
        x= torch.unsqueeze(adj.sum(-1), -1)
        adj = adj / x   # D = torch.diag_embed(torch.sum(adj, dim=-1) ** (-1))  adj =torch.matmul(D, adj)
        return adj
    @staticmethod
    def get_laplacian(graph, I, normalize=True):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            #L = I - torch.matmul(torch.matmul(D, graph), D)
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self, x, A):
        # x = torch.einsum("bnm,bmc->bnc", A, x)#[batch_size, D, num_nodes, num_steps]  [N,N]  [batch_size, num_steps, num_nodes, D]
        x = torch.einsum("bnm,bmc->bnc", A,x)  # [batch_size, D, num_nodes, num_steps]  [N,N]  [batch_size, num_steps, num_nodes, D]
        return x.contiguous()

class nconv1(nn.Module):
    def __init__(self):
        super(nconv1,self).__init__()

    def forward(self, x, A):
        x = torch.einsum("bnk,bnkc->bnc", A, x)#[batch_size, num_nodes, topk] [batch_size, num_steps,topk, D]
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

class gcn1(nn.Module):
    def __init__(self,k=2):
        super(gcn1,self).__init__()
        self.nconv = nconv1()
        self.k = k

    def forward(self,x,support):
        out = [x]
        B, N, D = x.shape
        for a in support:
            s1, t1 = a
            k1 = t1.shape[-1]
            index = torch.arange(B).repeat_interleave(N*k1,0)
            t = t1.reshape(-1)
            x1 = x[index,t].reshape(B,N,k1,D)
            x1 = self.nconv(x1,s1)                   #先做一次图扩散卷积
            out.append(x1)                         #放入输出列表中
            for k in range(2, self.k + 1):     #在对经过卷积的X1进行多级运算，得到一系列扩散卷积结果，都存入out中
                x1 = x1[index,t].reshape(B,N,k1,D)
                x2 = self.nconv(x1,s1)      #这里的order应该就是进行多少次扩散卷积运算，默认是2，那么range(2, self.order + 1)就是(2,3)也就是算两次就结束了
                out.append(x2)
                x1 = x2
        h = torch.stack(out, dim=1)
        #h = torch.cat(out,dim=1)                   #拼接结果
        return h
