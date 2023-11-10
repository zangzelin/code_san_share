import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import logging
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_auc_score,  accuracy_score
# import umap
import matplotlib.pyplot as plt


# def feat_get(G,  dataset_source, dataset_target):
#     G.eval()

#     for batch_idx, data in enumerate(dataset_source):
#         # if batch_idx == 500:
#         #     break
#         with torch.no_grad():
#             img_s = data[0]
#             label_s = data[1]
#             id_s = data[2]
#             img_s0, img_s1, img_s2 = Variable(img_s[0].cuda()), \
#                 Variable(img_s[1].cuda()), Variable(img_s[2].cuda())

#             # img_s, label_s = Variable(img_s.cuda()), \
#             #                  Variable(label_s.cuda())
#             feat_s = G(img_s2)


#             if batch_idx == 0:
#                 feat_all_s = feat_s.data.cpu().numpy()
#                 label_all_s = label_s.data.cpu().numpy()
#                 id_all_s = id_s.data.cpu().numpy()
#             else:
#                 feat_s = feat_s.data.cpu().numpy()
#                 label_s = label_s.data.cpu().numpy()
#                 id_s = id_s.data.cpu().numpy()
#                 feat_all_s = np.r_[feat_all_s, feat_s]
#                 label_all_s = np.r_[label_all_s, label_s]
#                 id_all_s = np.r_[id_all_s, id_s]
#     print('dataset_source->feat_all_s.mean()', feat_all_s.mean())
#     print('max(id_all_s.tolist())', max(id_all_s.tolist()))
#     print('min(id_all_s.tolist())', min(id_all_s.tolist()))
#     # import pdb; pdb.set_trace()

#     for batch_idx, data in enumerate(dataset_target):
#         # if batch_idx == 500:
#         #     break
#         with torch.no_grad():
#             img_t = data[0]
#             label_t = data[1]
#             id_t = data[2]
#             _, _, img_t = Variable(img_s[0].cuda()), \
#                 Variable(img_s[1].cuda()), Variable(img_s[2].cuda())

#             # img_t, label_t = Variable(img_t.cuda()), \
#             #                  Variable(label_t.cuda())
#             feat_t = G(img_t)

#             if batch_idx == 0:
#                 feat_all = feat_t.data.cpu().numpy()
#                 label_all = label_t.data.cpu().numpy()
#                 id_all_t = id_t.data.cpu().numpy()
#                 # unk_all = pred_unk.data.cpu().numpy()
#             else:
#                 feat_t = feat_t.data.cpu().numpy()
#                 label_t = label_t.data.cpu().numpy()
#                 id_t = id_t.data.cpu().numpy()
#                 # pred_unk = pred_unk.data.cpu().numpy()
#                 feat_all = np.r_[feat_all, feat_t]
#                 label_all = np.r_[label_all, label_t]
#                 id_all_t = np.r_[id_all_t, id_t]
#                 # unk_all = np.r_[unk_all, pred_unk]
#     print('dataset_target->feat_all_s.mean()', feat_all.mean())
#     G.train()
#     return feat_all_s, feat_all


from sklearn.metrics import pairwise_distances
import scipy


def DistanceSquared(x, y=None, metric="euclidean"):
    if metric == "euclidean":
        if y is not None:
            m, n = x.size(0), y.size(0)
            xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12)
        else:
            m, n = x.size(0), x.size(0)
            xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
            yy = xx.t()
            dist = xx + yy
            dist = torch.addmm(dist, mat1=x, mat2=x.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12)
            dist[torch.eye(dist.shape[0]) == 1] = 1e-12
    
    if metric == "cossim":
        input_a, input_b = x, x
        normalized_input_a = torch.nn.functional.normalize(input_a)  
        normalized_input_b = torch.nn.functional.normalize(input_b)
        dist = torch.mm(normalized_input_a, normalized_input_b.T)
        dist *= -1 # 1-dist without copy
        dist += 1

        dist[torch.eye(dist.shape[0]) == 1] = 1e-12

    return dist

def Similarity(dist, v=100, h=1, pow=2):

    a = scipy.special.gamma((v + 1) / 2)
    b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
    gamma = a / b
    # dist_rho = dist

    dist_rho = dist

    dist_rho[dist_rho < 0] = 0
    Pij = (
        gamma
        * torch.tensor(2 * 3.14)
        * gamma
        * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))
    )
    return Pij


def LossPatEmb(
    data_t,
    data_t_id,
    data_s,
    latent_data_t,
    latent_data_s,
    mask,
    v_latent,
):  
    def _CalGamma(v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out
    # import pdb; pdb.set_trace()
    # data_1 = input_data[: input_data.shape[0] // 2]
    # data_2 = input_data[input_data.shape[0] // 2 :]
    mask_c = mask[data_t_id]
    dis_P = DistanceSquared(data_t, data_s)

    dis_P_2 = dis_P # + nndistance.reshape(1, -1)
    P_2 = Similarity(dist=dis_P_2, v=100)
    P_2[mask_c] = 1-1e-5
    # latent_data_1 = latent_data[: input_data.shape[0] // 2]
    # latent_data_2 = latent_data[(input_data.shape[0] // 2):]
    dis_Q_2 = DistanceSquared(latent_data_t, latent_data_s)
    Q_2 = Similarity(
        dist=dis_Q_2,
        v=v_latent,
    )
    # loss_ce_2 = self.ITEM_loss(P_=P_2, Q_=Q_2)
    EPS = 1e-5
    losssum1 = P_2 * torch.log(Q_2 + EPS)
    losssum2 = (1 - P_2) * torch.log(1 - Q_2 + EPS)
    loss_ce_2 = -1 * (losssum1 + losssum2).mean()
    return loss_ce_2

def NF(dict_s, dict_t, k=30):
    # print(dict_s.sum(axis=0))
    # print('-----------------')
    # print(dict_s.sum(axis=1))
    dis = pairwise_distances(dict_t, dict_s)
    neighbors_index = dis.argsort(axis=1)[:, 1:k+1]
    # mask = np.zeros((dict_t.shape[0], dict_s.shape[0],))
    
    # for i in range(neighbors_index.shape[0]):
    #     mask[i, neighbors_index[i]] = 1
    # mask_index_0 = np.concatenate([np.array([i]*k) for i in range(dict_t.shape[0])]).reshape((-1,1))
    # mask_index_1 = neighbors_index.reshape((-1)).reshape((-1,1))
    # mask[np.concatenate([mask_index_0, mask_index_1])] = 1
    # import pdb; pdb.set_trace()
    return neighbors_index