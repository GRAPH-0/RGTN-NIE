import torch
from torch import nn
import torch.nn.functional as F
from g_transformer.g_trans import GTRANBRel_feat, GTRANRel_feat
import dgl
import tqdm


def list_loss(y_pred, y_true, list_num=10, eps=1e-10):
    '''
    y_pred: [n_node, 1]
    y_true: [n_node, 1]
    '''
    n_node = y_pred.shape[0]

    ran_num = list_num - 1
    indices = torch.multinomial(torch.ones(n_node), n_node*ran_num, replacement=True).to(y_pred.device)

    list_pred = torch.index_select(y_pred, 0, indices).reshape(n_node, ran_num)
    list_true = torch.index_select(y_true, 0, indices).reshape(n_node, ran_num)

    list_pred = torch.cat([y_pred, list_pred], -1) # [n_node, list_num]
    list_true = torch.cat([y_true, list_true], -1) # [n_node, list_num]

    list_pred = F.softmax(list_pred, -1)
    list_true = F.softmax(list_true, -1)

    list_pred = list_pred + eps
    log_pred = torch.log(list_pred)

    return torch.mean(-torch.sum(list_true * log_pred, dim=1))


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, temperature, in_dim, out_dim, in_drop):
        super(CrossAttention, self).__init__()
        self.temperature = temperature

        self.in_dim = in_dim
        self.out_dim = out_dim

        # q, k, v
        self.w_q = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.w_k = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.w_v = nn.Linear(self.in_dim, self.out_dim, bias=False)

        # FFN
        self.FFN = nn.Sequential(
            nn.Linear(self.out_dim, int(self.out_dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(self.out_dim*0.5), self.out_dim),
            nn.Dropout(0.1)
        )

        self.layer_norm = nn.LayerNorm(self.out_dim, eps=1e-6)


    def forward(self, struct_h, cont_h):
        h = torch.stack([struct_h, cont_h], 1) # [n_node, 2, in_dim]

        q = self.w_q(h)
        k = self.w_k(h)
        v = self.w_v(h)

        attn = torch.matmul(q / self.temperature, k.transpose(1,2))
        attn = F.softmax(attn, dim=-1)
        attn_h = torch.matmul(attn, v)

        attn_o = self.FFN(attn_h) + attn_h
        attn_o = self.layer_norm(attn_o)

        struct_o = attn_o[:, 0, :]
        cont_o = attn_o[:, 1, :]

        return struct_o, cont_o


# full batch version
class rgtn(nn.Module):
    def __init__(self,
                 args,
                 g,
                 rel_num,
                 struct_in_dim,
                 content_in_dim,
                 centrality,
                 loss_function):
        super(rgtn, self).__init__()
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        self.rel_emb = nn.Embedding(rel_num, args.pred_dim)
        self.struct_gtran = GTRANRel_feat(g, args.num_layers, rel_num, args.pred_dim, struct_in_dim,
                                 args.num_hidden, heads, args.in_drop, args.attn_drop,
                                 args.residual, args.norm, args.edge_mode, self.rel_emb)
        self.content_gtran = GTRANRel_feat(g, args.num_layers, rel_num, args.pred_dim, content_in_dim,
                                 args.num_hidden, heads, args.in_drop, args.attn_drop,
                                 args.residual, args.norm, args.edge_mode, self.rel_emb)
        self.loss_fn = loss_function
        self.feat_drop = args.feat_drop
        self.graph = g
        self.scale = args.scale

        self.h_dim = args.num_hidden*heads[-2]

        self.cross_attention = CrossAttention(self.h_dim ** 0.5, self.h_dim, self.h_dim, args.in_drop)

        self.attn_vec = nn.Parameter(torch.FloatTensor(size=(self.h_dim, 1)))
        nn.init.xavier_uniform_(self.attn_vec)

        self.output_layer1 = nn.Sequential(
            nn.Linear(self.h_dim, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.output_layer2 = nn.Sequential(
            nn.Linear(self.h_dim, 1),
            nn.LeakyReLU(inplace=True)
        )

        if self.scale:
            self.centrality = centrality
            self.gamma = nn.Parameter(torch.FloatTensor(size=(1,)))
            self.beta = nn.Parameter(torch.FloatTensor(size=(1,)))
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

        self.loss_lambda = args.loss_lambda
        self.bn_s = nn.BatchNorm1d(self.h_dim)
        self.bn_c = nn.BatchNorm1d(self.h_dim)

        self.loss_alpha = args.loss_alpha
        self.list_num = args.list_num

    def forward(self, struct_input, content_input, edge_types, labels=None, idx=None, ret_feat=False):
        if self.feat_drop > 0:
            struct_input = F.dropout(struct_input, self.feat_drop, self.training)
            content_input = F.dropout(content_input, self.feat_drop, self.training)

        struct_h = self.struct_gtran(struct_input, edge_types)
        content_h = self.content_gtran(content_input, edge_types)

        struct_h1, content_h1 = self.cross_attention(struct_h, content_h)
        # add_norm
        struct_h1 = self.bn_s(struct_h + struct_h1)
        content_h1 = self.bn_c(content_h + content_h1)

        # attention-based aggregation
        q = torch.stack([struct_h1, content_h1], 1)
        attn_last = torch.matmul(q, self.attn_vec) # [N_node, 2, 1]
        attn_last = F.softmax(attn_last, 1)

        logit_struct = self.output_layer1(q[:, 0, :])
        if self.scale:
            logit_struct = nn.functional.relu((self.centrality * self.gamma + self.beta).unsqueeze(-1) * logit_struct)

        logit_content = self.output_layer2(q[:, 1, :])
        logit_all = torch.stack([logit_struct, logit_content], 1) # [N_node, 2, 1]

        logits = torch.sum(logit_all * attn_last, 1) # [N_node, 1]

        if ret_feat:
            node_emb = torch.sum(q * attn_last, 1).cpu()
            return logits, node_emb

        if self.training:
            loss_struct = self.loss_fn(logit_struct[idx], labels[idx].unsqueeze(-1))
            loss_content = self.loss_fn(logit_content[idx], labels[idx].unsqueeze(-1))
            loss_all = self.loss_fn(logits[idx], labels[idx].unsqueeze(-1))
            loss = (1-self.loss_lambda) * loss_all + self.loss_lambda * (loss_struct + loss_content) / 2
            loss = self.loss_alpha * list_loss(logits[idx], labels[idx].unsqueeze(-1), self.list_num) + loss
            return logits, loss
        else:
            return logits


# mini batch version
class rgtn_b(nn.Module):
    def __init__(self,
                 args,
                 rel_num,
                 struct_in_dim,
                 content_in_dim,
                 loss_function):
        super(rgtn_b, self).__init__()

        self.rel_emb = nn.Embedding(rel_num, args.pred_dim)
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        self.struct_gtran = GTRANBRel_feat(args.num_layers, rel_num, args.pred_dim, struct_in_dim,
                                  args.num_hidden, heads, args.in_drop, args.attn_drop, args.residual,
                                  args.norm, args.edge_mode, self.rel_emb)
        self.content_gtran = GTRANBRel_feat(args.num_layers, rel_num, args.pred_dim, content_in_dim,
                                  args.num_hidden, heads, args.in_drop, args.attn_drop, args.residual,
                                  args.norm, args.edge_mode, self.rel_emb)
        self.loss_fn = loss_function
        self.feat_drop = args.feat_drop
        self.attn_struct = nn.Linear(args.num_hidden * heads[-2], 1)
        self.attn_content = nn.Linear(args.num_hidden * heads[-2], 1)
        self.scale = args.scale

        self.h_dim = args.num_hidden*heads[-2]

        self.attention = ScaledDotProductAttention(temperature=self.h_dim ** 0.5)

        self.w_q = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.w_k = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.w_v = nn.Linear(self.h_dim, self.h_dim, bias=False)

        self.out_fc = PositionwiseFeedForward(self.h_dim, int(0.5 * self.h_dim), 0.1)

        self.attn_vec = nn.Parameter(torch.FloatTensor(size=(self.h_dim,1)))
        nn.init.xavier_uniform_(self.attn_vec)

        self.output_layer1 = nn.Sequential(
            nn.Linear(self.h_dim, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.output_layer2 = nn.Sequential(
            nn.Linear(self.h_dim, 1),
            nn.LeakyReLU(inplace=True)
        )

        if self.scale:
            self.gamma = nn.Parameter(torch.FloatTensor(size=(1,)))
            self.beta = nn.Parameter(torch.FloatTensor(size=(1,)))
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

        self.loss_lambda = args.loss_lambda
        self.bn_s = nn.BatchNorm1d(self.h_dim)
        self.bn_c = nn.BatchNorm1d(self.h_dim)

        self.loss_alpha = args.loss_alpha
        self.list_num = args.list_num


    def forward(self, blocks, struct_input, content_input, labels=None):
        if self.feat_drop > 0:
            struct_input = F.dropout(struct_input, self.feat_drop, self.training)
            content_input = F.dropout(content_input, self.feat_drop, self.training)

        struct_h = self.struct_gtran(blocks, struct_input)
        content_h = self.content_gtran(blocks, content_input)

        h = torch.stack([struct_h, content_h], 1) # [N_node, 2, h_dim]

        # interact struct and content feature
        q = self.w_q(h)
        k = self.w_k(h)
        v = self.w_v(h)

        q, _ = self.attention(q, k, v)  # [N_node, 2, h_dim]
        q = self.out_fc(q) + h

        q = torch.stack(
            [self.bn_s(q[:, 0, :]), self.bn_c(q[:, 1, :])], 1
        )

        # attention-based aggregation
        attn_last = torch.matmul(q, self.attn_vec)  # [N_node, 2, 1]
        attn_last = F.softmax(attn_last, 1)

        logit_struct = self.output_layer1(q[:, 0, :])
        if self.scale:
            centrality = blocks[-1].dstdata['centrality']
            logit_struct = nn.functional.relu((centrality * self.gamma + self.beta).unsqueeze(-1) * logit_struct)

        logit_content = self.output_layer2(q[:, 1, :])
        logit_all = torch.stack([logit_struct, logit_content], 1)

        logits = torch.sum(logit_all * attn_last, 1)

        if self.training:
            loss_struct = self.loss_fn(logit_struct, labels)
            loss_content = self.loss_fn(logit_content, labels)
            loss_all = self.loss_fn(logits, labels)
            loss = (1-self.loss_lambda) * loss_all + self.loss_lambda * (loss_struct + loss_content) / 2
            loss = self.loss_alpha * list_loss(logits, labels, self.list_num) + loss
            return logits, loss
        else:
            return logits

    def inference(self, g, x_struct, x_content, batch_size, num_workers, device, ret_feat=False):
        layer_num = len(self.struct_gtran.gat_layers)
        layer_output_s = self.struct_gtran.layer_output
        layer_output_c = self.content_gtran.layer_output

        logits = torch.zeros(g.number_of_nodes(), 1).to(device)

        if ret_feat:
            node_emb = torch.zeros(g.number_of_nodes(), layer_output_c[-1])

        for l, (layer_s, layer_c) in enumerate(zip(self.struct_gtran.gat_layers, self.content_gtran.gat_layers)):

            y_struct = torch.zeros(g.number_of_nodes(), layer_output_s[l] if l!=layer_num-1 else 1)
            y_content = torch.zeros(g.number_of_nodes(), layer_output_c[l] if l!=layer_num-1 else 1)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.number_of_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]
                block = block.int().to(device)
                edge_types = block.edata['etypes']

                # structure branch
                edge_feat_s = self.struct_gtran.rel_emb(edge_types)
                h_struct = x_struct[input_nodes].to(device)
                h_struct = layer_s(block, h_struct, h_struct, h_struct, edge_feat_s)

                # content branch
                edge_feat_c = self.content_gtran.rel_emb(edge_types)
                h_content = x_content[input_nodes].to(device)
                h_content = layer_c(block, h_content, h_content, h_content, edge_feat_c)

                if l == (layer_num-1):
                    h = torch.stack([h_struct, h_content], 1)  # [N_node, 2, h_dim]

                    # interact struct and content feature
                    q = self.w_q(h)
                    k = self.w_k(h)
                    v = self.w_v(h)

                    q, _ = self.attention(q, k, v)  # [N_node, 2, h_dim]
                    q = self.out_fc(q) + h

                    q = torch.stack(
                        [self.bn_s(q[:, 0, :]), self.bn_c(q[:, 1, :])], 1
                    )

                    # attention-based aggregation
                    attn_last = torch.matmul(q, self.attn_vec)  # [N_node, 2, 1]
                    attn_last = F.softmax(attn_last, 1)

                    h_struct = self.output_layer1(q[:, 0, :])
                    if self.scale:
                        centrality = block.dstdata['centrality']
                        h_struct = nn.functional.relu(
                            (centrality * self.gamma + self.beta).unsqueeze(-1) * h_struct)

                    h_content = self.output_layer2(q[:, 1, :])
                    logit_all = torch.stack([h_struct, h_content], 1)

                    logits[output_nodes] = torch.sum(logit_all * attn_last, 1)
                    if ret_feat:
                        node_emb[output_nodes] = torch.sum(q * attn_last, 1).cpu()

                y_struct[output_nodes] = h_struct.cpu()
                y_content[output_nodes] = h_content.cpu()

            x_struct = y_struct
            x_content = y_content
        if ret_feat:
            return logits, node_emb
        return logits

