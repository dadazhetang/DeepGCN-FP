import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.dataloading import GraphDataLoader
from dgllife.model.gnn.gat import  GATLayer
# from dgllife.model.gnn.gcn import GCNLayer
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
from layers import GCNLayer, GCNLayerWithEdge


class EmbeddingLayerConcat(nn.Module):
    def __init__(self, node_in_dim, node_emb_dim, edge_in_dim=None, edge_emb_dim=None):
        super(EmbeddingLayerConcat, self).__init__()
        self.node_in_dim = node_in_dim
        self.node_emb_dim= node_emb_dim
        self.edge_in_dim = edge_emb_dim
        self.edge_emb_dim=edge_emb_dim

        self.atom_encoder = nn.Linear(node_in_dim, node_emb_dim)
        if edge_emb_dim is not None:
            self.bond_encoder = nn.Linear(edge_in_dim, edge_emb_dim)

    def forward(self, g):
        node_feats, edge_feats= g.ndata["node_feat"], g.edata["edge_feat"]
        node_feats = self.atom_encoder(node_feats)

        if self.edge_emb_dim is None:
            return node_feats
        else:
            edge_feats = self.bond_encoder(edge_feats)
            return  node_feats, edge_feats


'''GCN model with edge, attention and GRU readout'''
class GCNModelWithEdgeAFPreadout(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_feats=None, activation=F.relu,
                 residual=True, output_norm="none", dropout=0.1, gru_out_layer=2, update_func="no_relu"):
        super(GCNModelWithEdgeAFPreadout, self).__init__()
        # self.readout_layer_norm = readout_layer_norm

        if hidden_feats is None:
            hidden_feats = [200]*5

        in_feats = hidden_feats[0]
        n_layers = len(hidden_feats)

        # gnn_norm = [gnn_norm for _ in range(n_layers)]
        activation = [activation for _ in range(n_layers)]
        residual = [residual for _ in range(n_layers)]
        output_norm = [output_norm for _ in range(n_layers)]
        dropout = [dropout for _ in range(n_layers)]

        lengths = [len(hidden_feats), len(activation),
                   len(residual), len(output_norm), len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, gnn_norm, ' \
                                       'activation, residual, batchnorm and dropout to ' \
                                       'be the same, got {}'.format(lengths)

        self.embed_layer = EmbeddingLayerConcat(node_in_dim, hidden_feats[0], edge_in_dim, hidden_feats[0])
        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GCNLayerWithEdge(in_feats, hidden_feats[i], activation[i], residual[i], output_norm[i], dropout[i], update_func))
            in_feats = hidden_feats[i]

        self.readout = AttentiveFPReadout(
            hidden_feats[-1], num_timesteps=gru_out_layer, dropout=dropout[-1]
        )

        # mlp layers
        self.out = nn.Sequential(
            nn.Linear(hidden_feats[-1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g,ecfp):
        node_feat, edge_feat = self.embed_layer(g)
        for gnn in self.gnn_layers:
            node_feat = gnn(g, node_feat, edge_feat)
        # g.ndata['feats'] = feats
        # feats = dgl.sum_nodes(g, "feats")
        # feats = self.out(feats)
        out_gat = self.readout(g, node_feat)
        out = self.out(out_gat)
        return out,out_gat,ecfp


if __name__ == "__main__":
    pass

