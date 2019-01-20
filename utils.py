import os
import numpy as np
from sklearn.metrics import roc_auc_score
import config as cf


def sample_from_probs(cand_probs):
    batch_size, cand_size = np.shape(cand_probs)
    idx = [np.random.choice(cand_size, 1, p=cand_probs[i]) for i in range(batch_size)]
    return np.squeeze(idx)


def get_neg_from_idx(neg_cands, idx):
    batch_size, _ = np.shape(neg_cands)
    neg = [neg_cands[i][idx[i]] for i in range(batch_size)]
    return np.squeeze(neg)


def cal_auc(labels, scores):
    auc = roc_auc_score(y_true=labels, y_score=scores) * 100
    return auc


def write_results(data_list, name):
    with open(os.path.join(cf.result_dir, name + '.txt'), 'a') as f:
        opt_score = data_list[0]
        f.write('%s|%s|' % (cf.note, cf.data_dates))
        for data in data_list:
            f.write('%.3f ' % data)
            opt_score = max(opt_score, data)
        f.write('|%.3f\n' % opt_score)

