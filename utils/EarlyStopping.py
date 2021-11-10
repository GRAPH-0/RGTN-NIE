import os
import numpy as np
import torch


class EarlyStopping(object):
    def __init__(self, patience=50, save_path=None, min_epoch=-1):
        """
        strategy for early stopping
        :param patience: max patience
        :param save_path: save model path
        :param min_epoch: the minimum epochs for training
        """
        self.patience = patience
        self.counter = 0
        self.best_ndcg = None
        self.best_spm = None
        self.early_stop = False
        self.best_epoch = 0
        self.save_path = save_path
        self.min_epoch = min_epoch

    def step(self, ndcg, spm, epoch, model):
        """

        :param ndcg: NDCG@100, the higher, the better
        :param spm: spearman, the higher, the better
        :param model: model
        :param epoch: training epoch
        :return:
        """
        if self.best_ndcg is None or self.best_spm is None or \
                (ndcg > self.best_ndcg and spm > self.best_spm):
            self.best_ndcg = ndcg
            self.best_spm = spm
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        # metric not improved after an epoch
        elif ndcg <= self.best_ndcg or spm <= self.best_spm:
            if epoch < self.min_epoch:
                return self.early_stop
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            raise NotImplementedError('wrong conditions in early stopping.')

        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation metric increases."""
        # filename = os.path.join(self.save_folder, f"epoch_{epoch}.pkl")
        # print(f"save model {filename}")
        torch.save(model.state_dict(), self.save_path)

    # def load_checkpoint(self, model):
    #     """Load the latest checkpoint."""
    #     filename = os.path.join(self.save_folder, f"epoch_{self.best_epoch}.pkl")
    #     print(f"load model {filename}")
    #     model.load_state_dict(torch.load(filename))


class EarlyStopping_simple:
    def __init__(self, patience=50, save_path=None, min_epoch=-1):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.save_path = save_path
        self.min_epoch = min_epoch

    def step(self, acc, epoch, model):
        score = acc
        if epoch < self.min_epoch:
            return self.early_stop
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.save_path)
