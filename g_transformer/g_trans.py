import torch
import torch.nn as nn
from g_transformer.graph_transformer import GTLayer
import dgl
import tqdm

class GTRANRel(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 rel_num,
                 pred_dim,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 residual,
                 centrality,
                 scale,
                 batch_norm,
                 edge_mode,
                 ret_feat=False,
                 rel_emb=None):
        super(GTRANRel, self).__init__()
        self.g = g
        self.centrality = centrality
        self.scale = scale # True or False
        self.return_feat = ret_feat

        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        # input projection (no residual)
        self.gat_layers.append(GTLayer(
            in_dim, num_hidden, pred_dim, heads[0],
            feat_drop, attn_drop, residual, batch_norm, edge_mode))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GTLayer(
                num_hidden * heads[l-1], num_hidden, pred_dim, heads[l],
                feat_drop, attn_drop, residual, batch_norm, edge_mode))

        self.output_linear = nn.Linear(num_hidden * heads[-2], 1)

        if self.scale:
            self.gamma = nn.Parameter(torch.FloatTensor(size=(1,)))
            self.beta = nn.Parameter(torch.FloatTensor(size=(1,)))
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

        # relation embedding
        if rel_emb is None:
            self.rel_emb = nn.Embedding(rel_num, pred_dim)
        else:
            self.rel_emb = rel_emb

    def forward(self, inputs, edge_types):
        h = inputs
        edge_feats = self.rel_emb(edge_types)

        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, q=h, k=h, v=h, edge_feat=edge_feats)
        # output projection
        logits = self.output_linear(h)

        if self.scale:
            logits = nn.functional.relu((self.centrality * self.gamma + self.beta).unsqueeze(-1) * logits)

        if self.return_feat:
            return logits, h.clone()
        return logits


class GTRANRel_feat(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 rel_num,
                 pred_dim,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 residual,
                 batch_norm,
                 edge_mode,
                 rel_emb=None):
        super(GTRANRel_feat, self).__init__()
        self.g = g

        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        # input projection (no residual)
        self.gat_layers.append(GTLayer(
            in_dim, num_hidden, pred_dim, heads[0],
            feat_drop, attn_drop, residual, batch_norm, edge_mode))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GTLayer(
                num_hidden * heads[l-1], num_hidden, pred_dim, heads[l],
                feat_drop, attn_drop, residual, batch_norm, edge_mode))

        # relation embedding
        if rel_emb is None:
            self.rel_emb = nn.Embedding(rel_num, pred_dim)
        else:
            self.rel_emb = rel_emb

    def forward(self, inputs, edge_types):
        h = inputs
        edge_feats = self.rel_emb(edge_types)

        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, q=h, k=h, v=h, edge_feat=edge_feats)

        return h


class GTRANBRel(nn.Module):
    def __init__(self,
                 num_layers,
                 rel_num,
                 pred_dim,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 residual,
                 scale,
                 batch_norm,
                 edge_mode,
                 ret_feat=False,
                 rel_emb=None):
        super(GTRANBRel, self).__init__()
        self.scale = scale # True or False
        self.return_feat = ret_feat

        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()

        self.layer_output = []
        # input projection (no residual)
        self.gat_layers.append(GTLayer(
            in_dim, num_hidden, pred_dim, heads[0],
            feat_drop, attn_drop, residual, batch_norm, edge_mode))
        self.layer_output.append(num_hidden * heads[0])
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GTLayer(
                num_hidden * heads[l-1], num_hidden, pred_dim, heads[l],
                feat_drop, attn_drop, residual, batch_norm, edge_mode))
            self.layer_output.append(num_hidden * heads[l])

        self.output_linear = nn.Linear(num_hidden * heads[-2], 1)

        if self.scale:
            self.gamma = nn.Parameter(torch.FloatTensor(size=(1,)))
            self.beta = nn.Parameter(torch.FloatTensor(size=(1,)))
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

        # relation embedding
        if rel_emb is None:
            self.rel_emb = nn.Embedding(rel_num, pred_dim)
        else:
            self.rel_emb = rel_emb

    def forward(self, blocks, inputs):
        h = inputs

        for l in range(self.num_layers):
            edge_types = blocks[l].edata['etypes']
            edge_feats = self.rel_emb(edge_types)
            h = self.gat_layers[l](blocks[l], q=h, k=h, v=h, edge_feat=edge_feats)
        # output projection
        logits = self.output_linear(h)

        if self.scale:
            centrality = blocks[-1].dstdata['centrality']
            logits = nn.functional.relu((centrality * self.gamma + self.beta).unsqueeze(-1) * logits)

        if self.return_feat:
            return logits, h.clone()
        return logits

    def inference(self, g, x, batch_size, num_workers, device):

        for l, layer in enumerate(self.gat_layers):

            y = torch.zeros(g.number_of_nodes(), self.layer_output[l] if l!=len(self.gat_layers)-1 else 1)

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
                h = layer(block, q=h, k=h, v=h, edge_feat=edge_feat)

                if l == (len(self.gat_layers)-1):
                    h = self.output_linear(h)
                    if self.scale:
                        centrality = block.dstdata['centrality']
                        h = nn.functional.relu(((centrality * self.gamma + self.beta)).unsqueeze(-1) * h, inplace=True)

                y[output_nodes] = h.cpu()

            x = y

        return y.to(device)


class GTRANBRel_feat(nn.Module):
    def __init__(self,
                 num_layers,
                 rel_num,
                 pred_dim,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 residual,
                 batch_norm,
                 edge_mode,
                 rel_emb=None):
        super(GTRANBRel_feat, self).__init__()

        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()

        self.layer_output = []
        # input projection (no residual)
        self.gat_layers.append(GTLayer(
            in_dim, num_hidden, pred_dim, heads[0],
            feat_drop, attn_drop, residual, batch_norm, edge_mode))
        self.layer_output.append(num_hidden * heads[0])
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GTLayer(
                num_hidden * heads[l-1], num_hidden, pred_dim, heads[l],
                feat_drop, attn_drop, residual, batch_norm, edge_mode))
            self.layer_output.append(num_hidden * heads[l])

        # relation embedding
        if rel_emb is None:
            self.rel_emb = nn.Embedding(rel_num, pred_dim)
        else:
            self.rel_emb = rel_emb

    def forward(self, blocks, inputs):
        h = inputs

        for l in range(self.num_layers):
            edge_types = blocks[l].edata['etypes']
            edge_feats = self.rel_emb(edge_types)
            h = self.gat_layers[l](blocks[l], q=h, k=h, v=h, edge_feat=edge_feats)

        return h
