import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import accumulate

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphAttention(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=1, nheads=4, dropout=0.1, alpha=0.1):
        super(GraphAttention, self).__init__()

        self.nheads = nheads

        self.layers = [
            [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        ]
        self.layers += [
            [GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
            for _ in range(nlayers-1)
        ]

        self.dropout = dropout

        for j in range(nlayers):
            for i, attention in enumerate(self.layers[j]):
                self.add_module('attention_{} @ layer {}'.format(i, j), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, x, adj):

        x = [x for _ in range(self.nheads)]
        for layer in self.layers:
            #print(self.dropout)
            x = [F.dropout(s, self.dropout, training=self.training) for s in x]
            x = [att(s, adj) for s, att in zip(x, layer)]
            x = [F.dropout(s, self.dropout, training=self.training) for s in x]
            x = [F.elu(s) for s in x]
            
        x = torch.cat(x, dim=1)
        x = F.elu(self.out_att(x, adj))
        #return F.log_softmax(x, dim=1)
        return x


class GraphBatchAttention(nn.Module):
    def __init__(self, nvertex, nhid, nclass, nlayers=1, nheads=4, dropout=0.1, alpha=0.1):
        super(GraphBatchAttention, self).__init__()

        self.v_embedding = nn.Embedding(nvertex, nhid)
        self.gat = GraphAttention(nfeat=nhid, nhid=nhid, nclass=nclass, 
                                nlayers=nlayers, nheads=nheads, dropout=dropout, 
                                alpha=alpha)

    
    def forward(self, vertices, adjs):
        """[summary]
        
        Arguments:
            vertices: a list of vertices. 
                Each element is a 1-dimensional torch.LongTensor, not necessarily of the same length. 
            adjs -- list of adjacency matrices.
                Each element is a 2-dimensional torch.FloatTensor.
        """

        lens = list(map(len, vertices)) 

        # pack vertices and adjacency matrices
        vtx_packed = torch.cat(vertices, dim=0)   
        n_vtx = vtx_packed.shape[0]
        adj_packed = torch.zeros((n_vtx, n_vtx))
        cur = 0
        for adj, n in zip(adjs, lens):
            adj_packed[cur:cur+n, cur:cur+n] = adj
            cur = cur + n

        x = self.v_embedding(vtx_packed)
        
        out = self.gat(x, adj_packed)
        segs = [0] + list(accumulate(lens))
        return [out[segs[i]: segs[i+1]] for i in range(len(segs)-1)]
        
        