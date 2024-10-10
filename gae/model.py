import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class GCNModelAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, dropout):
        super(GCNModelAE, self).__init__()
        self.hidden = GraphConvolution(input_dim, hidden_dim, dropout, act=F.relu)
        self.embeddings = GraphConvolution(hidden_dim, z_dim, dropout, act=lambda x: x)
        self.reconstructions = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self, x, adj):
        hidden = self.hidden(x, adj)
        embeddings = self.embeddings(hidden, adj)
        z_mean = embeddings
        return self.reconstructions(embeddings), z_mean


class GCNModelVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, dropout):
        super(GCNModelVAE, self).__init__()
        self.hidden = GraphConvolution(input_dim, hidden_dim, dropout, act=F.relu)
        self.z_mean = GraphConvolution(hidden_dim, z_dim, dropout, act=lambda x: x)
        self.z_log_std = GraphConvolution(hidden_dim, z_dim, dropout, act=lambda x: x)
        self.reconstructions = InnerProductDecoder(dropout, act=lambda x: x)

    def reparameterize(self, z_mean, z_log_std):
        if self.training:
            std = torch.exp(z_log_std)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(z_mean)
        else:
            return z_mean

    def forward(self, x, adj):
        hidden = self.hidden(x, adj)
        z_mean = self.z_mean(hidden, adj)
        z_log_std = self.z_log_std(hidden, adj)
        z = self.reparameterize(z_mean, z_log_std)
        return self.reconstructions(z), z_mean, z_log_std


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
