import torch
import torch.nn as nn
from .SA_layer import SALayer
import dgl
import tqdm

class GENIB(nn.Module):
    def __init__(self,
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
                 scale):
        super(GENIB, self).__init__()
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
                nn.Linear(in_dim, int(0.75*in_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(0.75*in_dim), 1)))
        # Relation Embedding
        self.rel_emb = nn.Embedding(rel_num, pred_dim)

        self.layer_output = []
        # hidden layers
        for l in range(0, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.sa_layers.append(SALayer(
                1, 1, pred_dim, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
            self.layer_output.append(heads[l])

        if self.scale:
            self.gamma = nn.Parameter(torch.FloatTensor(size=(1, heads[-2])))
            self.beta = nn.Parameter(torch.FloatTensor(size=(1, heads[-2])))
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def forward(self, blocks, inputs):
        h = [score_nn(inputs) for score_nn in self.scoring_nn]
        h = torch.cat(h, dim=-1)

        for l in range(self.num_layers):
            edge_types = blocks[l].edata['etypes']
            edge_feats = self.rel_emb(edge_types)
            h = self.sa_layers[l](blocks[l], h, edge_feats)

            if l != (self.num_layers-1):
                h = h.flatten(1).mean(-1, keepdim=True) # [n_node, 1]
                h = h.repeat(1, self.heads[l])

        # output scale
        logits = h.flatten(1) # [n_nodes, n_heads]

        if self.scale:
            centrality = blocks[-1].dstdata['centrality']
            logits = nn.functional.relu(((centrality.unsqueeze(-1) * self.gamma + self.beta) * logits)\
                                        .mean(-1, keepdim=True), inplace=True)
        else:
            logits = logits.mean(-1, keepdim=True)
        return logits

    def inference(self, g, x, batch_size, num_workers, device):
        x = g.ndata['features']

        for l, layer in enumerate(self.sa_layers):
            y = torch.zeros(g.number_of_nodes(), self.layer_output[l] if l!=self.num_layers-1 else 1)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.number_of_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]
                block = block.int().to(device)
                edge_types = block.edata['etypes']
                edge_feat = self.rel_emb(edge_types)

                h = x[input_nodes].to(device)
                if l == 0:
                    h = [score_nn(h) for score_nn in self.scoring_nn]
                    h = torch.cat(h, dim=-1)
                h = layer(block, h, edge_feat).flatten(1)

                if l != (self.num_layers-1):
                    # the hidden layers
                    h = h.flatten(1).mean(-1, keepdim=True) # [n_node, 1]
                    h = h.repeat(1, self.heads[l])
                else:
                    # the last output layer
                    h = h.flatten(1) # [n_nodes, n_heads]
                    if self.scale:
                        centrality = block.dstdata['centrality']
                        h = nn.functional.relu(((centrality.unsqueeze(-1) * self.gamma + self.beta) * h)\
                                        .mean(-1, keepdim=True), inplace=True)
                    else:
                        h = h.mean(-1, keepdim=True)

                y[output_nodes] = h.cpu()

            x = y
        return y.to(device)