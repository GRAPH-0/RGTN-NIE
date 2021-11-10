import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
import pdb

class GTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 edge_dim,
                 n_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 residual=True,
                 batch_norm=True,
                 edge_mode=None):
        super(GTLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.residual = residual
        self.batch_norm = batch_norm
        self.edge_mode = edge_mode # {None, 'MUL', 'ADD'}

        self.q_linear = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.k_linear = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.v_linear = nn.Linear(in_dim, out_dim * n_heads, bias=False)

        if self.edge_mode == 'DOT':
            self.e_linear = nn.Linear(edge_dim, out_dim * n_heads, bias=False)
        elif self.edge_mode == 'GATE':
            self.e_linear = nn.Sequential(
                nn.Linear(edge_dim, n_heads),
                nn.BatchNorm1d(n_heads),
                nn.Sigmoid()
            )
        else:
            self.e_linear = nn.Linear(edge_dim, n_heads)

        self.out_channel = out_dim * n_heads
        self.sqrt_dim = out_dim ** 0.5

        self.o_linear = nn.Linear(self.out_channel, self.out_channel, bias=False)

        self.FFN_h = nn.Sequential(
            nn.Linear(self.out_channel, self.out_channel*2),
            nn.ReLU(),
            nn.Linear(self.out_channel*2, self.out_channel),
            nn.Dropout(feat_drop),
        )

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(self.out_channel)
            self.batch_norm2 = nn.BatchNorm1d(self.out_channel)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        if residual:
            if self.in_dim != self.out_channel:
                self.res_fc = nn.Linear(self.in_dim, self.out_channel, bias=False)
            else:
                self.res_fc = Identity()


    def forward(self, graph, q, k, v, edge_feat):
        with graph.local_scope():
            #　q: dst feature;  k,v: src feature
            if graph.is_block:
                q_in1 = q_dst = self.feat_drop(q[:graph.number_of_dst_nodes()])
            else:
                q_in1 = q_dst = self.feat_drop(q)
            k_src = self.feat_drop(k)
            v_src = self.feat_drop(v)

            # scale dot attention
            q_dst = self.q_linear(q_dst).reshape(-1, self.n_heads, self.out_dim)
            k_src = self.k_linear(k_src).reshape(-1, self.n_heads, self.out_dim)
            v_src = self.v_linear(v_src).reshape(-1, self.n_heads, self.out_dim)

            graph.dstdata.update({'q': q_dst})
            graph.srcdata.update(
                {'k': k_src, 'v': v_src}
            )

            # dot product q and k
            graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
            attn_score = graph.edata.pop('t').sum(-1) / self.sqrt_dim # [n_edge, n_head]

            # project edge info
            if self.edge_mode == 'DOT':
                rel_feat = self.e_linear(edge_feat).reshape(-1, self.n_heads, self.out_dim)
                graph.edata.update({'rel': rel_feat})
                graph.apply_edges(lambda edges: {'e': edges.src['q'] * edges.data['rel']})
                rel_attn = graph.edata.pop('e').sum(-1) / self.sqrt_dim
                attn_score = attn_score + rel_attn
            elif self.edge_mode == 'MUL':
                edge_attn = self.e_linear(edge_feat)  # [n_edge, n_head]
                attn_score = attn_score * edge_attn
            elif self.edge_mode == 'ADD':
                edge_attn = self.e_linear(edge_feat)  # [n_edge, n_head]
                attn_score = attn_score + edge_attn
            elif self.edge_mode == 'GATE':
                edge_attn = self.e_linear(edge_feat)
                attn_score = attn_score * edge_attn
            attn_score = self.attn_drop(edge_softmax(graph, attn_score.unsqueeze(-1), norm_by='dst'))

            graph.edata.update({'t': attn_score})

            # message passing
            graph.update_all(fn.u_mul_e('v', 't', 'm'),
                             fn.sum('m', 't'))
            rst = graph.dstdata['t'] # [n_node, n_head, out_dim]
            rst = self.o_linear(rst.flatten(1)) # [n_node, o_channel]

            # add-norm
            if self.residual:
                resval = self.res_fc(q_in1)
                rst = rst + resval

            if self.batch_norm:
                rst = self.batch_norm1(rst)

            # FFN
            q_in2 = rst
            rst = self.FFN_h(rst)

            # add-norm
            if self.residual:
                rst = rst + q_in2

            if self.batch_norm:
                rst = self.batch_norm2(rst)

            return rst
