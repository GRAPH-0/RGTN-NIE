import torch
import scipy.stats

def dcg(y_true, y_pred, top_k):
    '''
    params:
        y_true: tensor (n_samples, 1)
        y_pred: tensor (n_samples, 1)
        top_k: int
    '''
    with torch.no_grad():
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        _, pred_indices = y_pred.topk(k=top_k)
        gain = y_true.gather(-1, pred_indices)

        return (gain.float() / torch.log2(torch.arange(top_k, device=y_pred.device).float() + 2)).sum(-1)


def ndcg(y_true, y_pred, top_k):
    '''
    params:
        y_true: tensor (n_samples, 1)
        y_pred: tensor (n_samples, 1)
        top_k: int
    '''

    dcg_score = dcg(y_true, y_pred, top_k)
    idcg_score = dcg(y_true, y_true, top_k)

    with torch.no_grad():
        return (dcg_score / idcg_score).item()


def spearman_sci(y_true, y_pred):
    y_true = y_true.reshape(-1).detach().cpu().numpy()
    y_pred = y_pred.reshape(-1).detach().cpu().numpy()
    return scipy.stats.spearmanr(y_true, y_pred)[0]

def overlap(y_true, y_pred, topk):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    assert y_true.shape == y_pred.shape
    _, true_indices = y_true.topk(k=topk)
    _, pred_indices = y_pred.topk(k=topk)

    overlap_num = len(set(true_indices.tolist()) & set(pred_indices.tolist()))
    return overlap_num / topk


if __name__ == '__main__':
    from scipy import stats
    a = [1,2,3,4,5]
    b = [5,6,7,8,7]
    print(stats.spearmanr(a,b))
