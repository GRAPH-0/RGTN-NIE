import torch
import torch.nn as nn
from .SA_layer import SALayer

class GENI(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 rel_num,
                 pred_dim,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 centrality,
                 scale):
        super(GENI, self).__init__()
        self.g = g
        self.centrality = centrality
        self.scale = scale # True or False

        self.num_layers = num_layers
        self.sa_layers = nn.ModuleList()
        self.activation = activation
        # Scoring Network
        self.scoring_nn = nn.ModuleList()
        self.heads = heads
        for _ in range(heads[0]):
            self.scoring_nn.append(nn.Sequential(
                nn.Dropout(feat_drop),
                nn.Linear(in_dim, int(0.5*in_dim)),
                nn.ReLU(),
                nn.Linear(int(0.5*in_dim), 1)))
        # Relation Embedding
        self.rel_emb = nn.Embedding(rel_num, pred_dim)

        # hidden layers
        for l in range(0, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.sa_layers.append(SALayer(
                1, 1, pred_dim, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))

        if self.scale:
            self.gamma = nn.Parameter(torch.FloatTensor(size=(1, heads[-2])))
            self.beta = nn.Parameter(torch.FloatTensor(size=(1, heads[-2])))
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def forward(self, inputs, edge_types):
        h = [score_nn(inputs) for score_nn in self.scoring_nn]
        h = torch.cat(h, dim=-1)

        edge_feats = self.rel_emb(edge_types)
        for l in range(self.num_layers):
            h = self.sa_layers[l](self.g, h, edge_feats)
            if l != (self.num_layers-1):
                h = h.flatten(1).mean(-1, keepdim=True) # [n_node, 1]
                h = h.repeat(1, self.heads[l])
        # output scale
        logits = h.flatten(1) # [n_nodes, n_heads]

        if self.scale:
            logits = nn.functional.leaky_relu(((self.centrality.unsqueeze(-1) * self.gamma + self.beta) * logits)\
                                        .mean(-1, keepdim=True))
        else:
            logits = logits.mean(-1, keepdim=True)
        return logits