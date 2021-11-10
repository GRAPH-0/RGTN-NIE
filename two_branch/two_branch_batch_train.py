import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
import os
import sys
import pickle as pk
import tqdm

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from utils.EarlyStopping import EarlyStopping_simple
from utils.utils import set_random_seed, load_data, get_rank_metrics, rank_evaluate, get_centrality
from utils.metric import overlap, ndcg
from two_branch.model import rgtn_b


def main(args):

    set_random_seed(0)

    ndcg_scores = []
    spearmans = []
    rmses = []
    overlaps = []

    # set the save path
    save_root = 'results/' + args.dataset + '_RGTN/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for cross_id in range(args.cross_num):
        g, edge_types, _, rel_num, struct_feat, content_feat, labels, train_idx, val_idx, test_idx = \
            load_data(args.data_path, args.dataset, cross_id)

        # convert train_idx to mask
        with torch.no_grad():
            train_mask = torch.zeros(size=(g.number_of_nodes(),))
            train_mask[train_idx] = 1

        g.ndata['struct_feat'] = struct_feat
        g.ndata['content_feat'] = content_feat
        g.ndata['labels'] = labels.unsqueeze(-1)
        g.ndata['mask'] = train_mask.unsqueeze(-1)

        # add self loop
        g = dgl.add_self_loop(g)
        new_edge_types = torch.tensor([rel_num for _ in range(g.number_of_nodes())])
        edge_types = torch.cat([edge_types, new_edge_types], 0)
        rel_num += 1
        g.edata['etypes'] = edge_types
        g.ndata['centrality'] = get_centrality(g)

        # generate edge norm
        with g.local_scope():
            in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
            norm = 1.0 / in_deg
            norm[np.isinf(norm)] = 0
            node_norm = torch.from_numpy(norm).view(-1, 1)
            g.ndata['norm'] = node_norm
            g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
            edge_norm = g.edata['norm']
        g.edata['enorm'] = edge_norm

        # create pytorch dataloader for constructing block

        ## full sampler
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)

        ## neighbor sampler
        # samples = [ (args.num_sample * i) for i in range(args.num_layers, 0, -1) ]
        # sampler = dgl.dataloading.MultiLayerNeighborSampler(samples)

        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            train_idx,
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=args.num_workers
        )
        total_step = len(train_idx) // args.batch_size + 1

        if args.gpu >= 0:
            device = torch.device('cuda:%d' % args.gpu)
        else:
            device = torch.device('cpu')

        labels = labels.to(device)
        n_edges = g.number_of_edges()

        num_struct_feat = struct_feat.shape[1]
        num_content_feat = content_feat.shape[1]

        print("""----Data statistics------'
          #Edges %d
          #Train samples %d
          #Val samples %d
          #Test samples %d""" %
              (n_edges,
               len(train_idx),
               len(val_idx),
               len(test_idx)))

        # create model
        loss_fcn = torch.nn.MSELoss()
        model = rgtn_b(args, rel_num, num_struct_feat, num_content_feat, loss_fcn)

        print(model)
        model_path = save_root + str(cross_id) + '_' + args.save_path
        if args.early_stop:
            stopper = EarlyStopping_simple(patience=args.patience, save_path=model_path, min_epoch=args.min_epoch)

        model = model.to(device)

        # use optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # initialize graph
        for epoch in range(args.epochs):

            for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                # load the input features and output labels
                blocks = [block.int().to(device) for block in blocks]
                batch_struct = blocks[0].srcdata['struct_feat']
                batch_content = blocks[0].srcdata['content_feat']
                batch_labels = blocks[-1].dstdata['labels']

                # forward
                model.train()
                batch_pred, loss = model(blocks, batch_struct, batch_content, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_ndcg = get_rank_metrics(batch_pred, batch_labels, 100)

                print("Epoch {:05d} | Step {:05d}/{:05d} | Loss {:.4f} | TrainNDCG {:.4f} |".
                      format(epoch, step, total_step, loss.item(), train_ndcg))

            model.eval()
            with torch.no_grad():
                val_logits = model.inference(g, struct_feat, content_feat, args.batch_size, args.num_workers, device)
                val_loss, val_ndcg, val_spm = rank_evaluate(val_logits[val_idx], labels[val_idx].unsqueeze(-1), 100, loss_fcn, spearman=True)
                test_loss, test_ndcg, test_spm = rank_evaluate(val_logits[test_idx], labels[test_idx].unsqueeze(-1), 100, loss_fcn, spearman=True)

            if args.early_stop:
                if args.spm:
                    stop = stopper.step(val_spm, epoch, model)
                else:
                    stop = stopper.step(val_ndcg, epoch, model)
                if stop:
                    print('best epoch :', stopper.best_epoch)
                    break

            print("CROSS {} | Epoch {:05d} | ValLoss {:.4f} | ValNDCG {:.4f} | ValSPM {:.4f} | TestLoss {:.4f} | "
                  "TestNDCG {:.4f} | TestSPM {:.4f}".
                  format(cross_id, epoch, val_loss, val_ndcg, val_spm, test_loss, test_ndcg, test_spm))

        print()
        if args.early_stop:
            model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            test_logits = model.inference(g, struct_feat, content_feat, args.batch_size, args.num_workers, device)
            test_loss, test_ndcg, test_spearman = \
                rank_evaluate(test_logits[test_idx], labels[test_idx].unsqueeze(-1), 100, loss_fcn, spearman=True)
            test_overlap = overlap(labels[test_idx], test_logits[test_idx], 100)
            print("Test NDCG {:.4f} | Test Loss {:.4f} | Test Spearman {:.4f} | Test Overlap {:.4f}".
                  format(test_ndcg, test_loss, test_spearman, test_overlap))

        ndcg_scores.append(test_ndcg)
        spearmans.append(test_spearman)
        rmses.append(torch.sqrt(test_loss).item())
        overlaps.append(test_overlap)

    print()
    ndcg_scores = np.array(ndcg_scores)
    print('ndcg: ', ndcg_scores, ndcg_scores.mean(), np.std(ndcg_scores))

    spearmans = np.array(spearmans)
    print('spearmans: ', spearmans, spearmans.mean(), np.std(spearmans))

    rmses = np.array(rmses)
    print('RMSE: ', rmses, rmses.mean(), np.std(rmses))

    overlaps = np.array(overlaps)
    print(overlaps, overlaps.mean(), np.std(overlaps))

    results = {'ndcg': ndcg_scores,
               'spearman': spearmans,
               'rmse': rmses,
               'overlap': overlaps,
               'args': vars(args)}

    result_path = save_root + args.save_path.replace('checkpoint.pt', '') + 'result.pk'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    pk.dump(results, open(result_path, 'wb'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT-Two-Batch')
    parser.add_argument("--dataset", type=str, default='IMDB_S_rel_two',
                        help="The input dataset. Can be IMDB_S_rel_two")
    parser.add_argument("--data_path", type=str, default='datasets/imdb_s_rel.pk',
                        help="path of dataset")
    parser.add_argument("--cross-num", type=int, default=5,
                        help="number of cross validation")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="number of nodes in a batch")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of sampling processes. Use 0 for no extra process.")
    parser.add_argument('--min-epoch', type=int, default=-1,
                        help='the least epoch for training, avoiding stopping at the start time')
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=4,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=20,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--feat-drop", type=float, default=0.)
    parser.add_argument("--in-drop", type=float, default=.3,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.3,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=50,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--scale', action="store_true", default=False,
                        help="utilize centrality to scale scores")
    parser.add_argument('--pred-dim', type=int, default=10,
                        help="the size of predicate embedding vector")
    parser.add_argument('--save-path', type=str, default='gat-fuse_checkpoint.pt',
                        help='the path to save the best model')

    parser.add_argument('--loss-lambda', type=float, default=0.5,
                        help='the weight to add unsupervised loss')
    parser.add_argument('--norm', action="store_true", default=False)
    parser.add_argument('--edge-mode', type=str, default='MUL')
    parser.add_argument('--spm', action="store_true",default=False,
                        help="Use spearman to early stop")
    parser.add_argument('--loss-alpha', type=float, default=1.,
                        help="the weight to add list loss")
    parser.add_argument('--list-num', type=int, default=10, help="the number of list for the list loss")

    parser.add_argument('--num-sample', type=int, default=5, help='the number of samples for layer sampling')

    args = parser.parse_args()
    print(args)

    main(args)
