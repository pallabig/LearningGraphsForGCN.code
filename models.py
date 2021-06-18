import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, adj, A):
        x = F.relu(self.gc1(x, adj))
        x1 = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, A)
        return x1, F.log_softmax(x, dim=1)
