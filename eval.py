import numpy as np
import torch
print('-----')
import torch.nn.functional as F
from torch.autograd import Variable
print('-----')
import os
import logging
print('-----')
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_auc_score,  accuracy_score
print('-----')
# from openTSNE import TSNE
import pandas as pd 
import uuid
import wandb
import matplotlib.pyplot as plt
# import umap
print('-----')


def feat_get(step, G, Cs, dataset_source, dataset_target, save_path,
             ova=True):
    G.eval()

    for batch_idx, data in enumerate(dataset_source):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_s = data[0]
            label_s = data[1]
            img_s, label_s = Variable(img_s.cuda()), \
                             Variable(label_s.cuda())
            feat_s = G(img_s)


            if batch_idx == 0:
                feat_all_s = feat_s.data.cpu().numpy()
                label_all_s = label_s.data.cpu().numpy()
            else:
                feat_s = feat_s.data.cpu().numpy()
                label_s = label_s.data.cpu().numpy()
                feat_all_s = np.r_[feat_all_s, feat_s]
                label_all_s = np.r_[label_all_s, label_s]
    for batch_idx, data in enumerate(dataset_target):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_t = data[0]
            label_t = data[1]
            img_t, label_t = Variable(img_t.cuda()), \
                             Variable(label_t.cuda())
            feat_t = G(img_t)

            out_t = Cs[0](feat_t)
            pred = out_t.data.max(1)[1]
            out_t = F.softmax(out_t)
            if ova:
                out_open = Cs[1](feat_t)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1), 1)
                tmp_range = torch.arange(out_t.size(0)).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                weights_open = Cs[1].module.fc.weight.data.cpu().numpy()
            else:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)

            if batch_idx == 0:
                feat_all = feat_t.data.cpu().numpy()
                label_all = label_t.data.cpu().numpy()
                unk_all = pred_unk.data.cpu().numpy()
                pred_all = pred.data.cpu().numpy()
                pred_all_soft = out_t.data.cpu().numpy()
            else:
                feat_t = feat_t.data.cpu().numpy()
                label_t = label_t.data.cpu().numpy()
                pred_unk = pred_unk.data.cpu().numpy()
                feat_all = np.r_[feat_all, feat_t]
                label_all = np.r_[label_all, label_t]
                unk_all = np.r_[unk_all, pred_unk]
                pred_all = np.r_[pred_all, pred.data.cpu().numpy()]
                pred_all_soft = np.r_[pred_all_soft, out_t.data.cpu().numpy()]

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, "save_%s_ova_%s_target_feat.npy" % (step, ova)), feat_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_anom.npy" % (step, ova)), unk_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_pred.npy" % (step, ova)), pred_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_soft.npy" % (step, ova)), pred_all_soft)
    np.save(os.path.join(save_path, "save_%s_ova_%s_source_feat.npy" % (step, ova)), feat_all_s)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_label.npy" % (step, ova)), label_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_source_label.npy" % (step, ova)), label_all_s)
    if ova:
        np.save(os.path.join(save_path, "save_%s_ova_%s_weight.npy" % (step, ova)), weights_open)


def inference(dataset_test, G, Cs, class_list, entropy, thr, per_class_correct, per_class_num, correct_close, correct, size, thr_open=0.5, sourse_loader_bool=False):
    
    feature_list = []
    predict_list = []
    path_list = []
    pred_unk_list = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            if sourse_loader_bool == False:
                img_t, label_t, path_t = data[0].cuda(), data[1].cuda(), data[2]
            else:
                img_t, label_t, path_t = data[0][0].cuda(), data[1].cuda(), str(data[2])

            path_list += path_t
            feat = G(img_t)
            # feat = (feat)
            out_t = Cs[0](feat)
            if batch_idx == 0:
                open_class = int(out_t.size(1))
                class_list.append(open_class)
            pred = out_t.data.max(1)[1]
            feature_list.append(feat)
            # predict_list.append(pred)
            # print('out_t.shape', out_t.shape)
            # print('label_t.shape', label_t.shape)
            # print('pred.shape', pred.shape)
            correct_close += pred.eq(label_t.data).cpu().sum()
            out_t = F.softmax(out_t, dim=1)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                out_open = Cs[1](feat)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                tmp_range = torch.arange(out_t.size(0)).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > thr_open)[0]
            pred_unk_list.append(pred_unk)
            
            pred[ind_unk] = open_class
            predict_list.append(pred)
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]

    return feature_list, predict_list, path_list, pred_unk_list, label_all, pred_open, pred_ent, pred_all, size, correct, correct_close


def inference_amlp(dataset_test, G_mlp, Cs, class_list, entropy, thr, per_class_correct, per_class_num, correct_close, correct, size, thr_open=0.5, sourse_loader_bool=False):
    
    G, mlp = G_mlp
    feature_list = []
    predict_list = []
    path_list = []
    pred_unk_list = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            # if sourse_loader_bool == False:
            img_t, label_t, path_t = data[0].cuda(), data[1].cuda(), data[2]
            # else:
            #     img_t, label_t, path_t = data[0][0].cuda(), data[1].cuda(), str(data[2])

            path_list += path_t
            feat = mlp(G(img_t))
            # feat = (feat)
            out_t = Cs[0](feat)
            if batch_idx == 0:
                open_class = int(out_t.size(1))
                class_list.append(open_class)
            pred = out_t.data.max(1)[1]
            feature_list.append(feat)
            # predict_list.append(pred)
            # print('out_t.shape', out_t.shape)
            # print('label_t.shape', label_t.shape)
            # print('pred.shape', pred.shape)
            correct_close += pred.eq(label_t.data).cpu().sum()
            out_t = F.softmax(out_t, dim=1)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                out_open = Cs[1](feat)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                tmp_range = torch.arange(out_t.size(0)).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > thr_open)[0]
            pred_unk_list.append(pred_unk)
            
            pred[ind_unk] = open_class
            predict_list.append(pred)
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]

    return feature_list, predict_list, path_list, pred_unk_list, label_all, pred_open, pred_ent, pred_all, size, correct, correct_close


def inference_amlp_v2(dataset_test, G_mlp, Cs, class_list, entropy, thr, per_class_correct, per_class_num, correct_close, correct, size, thr_open=0.5, sourse_loader_bool=False):
    
    G, mlp = G_mlp
    feature_list = []
    predict_list = []
    path_list = []
    pred_unk_list = []
    c1_softmax_all_predict_list = []
    c2_softmax_all_predict_list = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            # if sourse_loader_bool == False:
            img_t, label_t, path_t = data[0].cuda(), data[1].cuda(), data[2]
            # else:
            #     img_t, label_t, path_t = data[0][0].cuda(), data[1].cuda(), str(data[2])

            path_list += path_t
            feat = mlp(G(img_t))
            # feat = (feat)
            out_t = Cs[0](feat)
            c1_softmax_all_predict_list.append(torch.softmax(out_t, dim=1))
            # import pdb; pdb.set_trace()
            if batch_idx == 0:
                open_class = int(out_t.size(1))
                class_list.append(open_class)
            pred = out_t.data.max(1)[1]
            feature_list.append(feat)
            # predict_list.append(pred)
            # print('out_t.shape', out_t.shape)
            # print('label_t.shape', label_t.shape)
            # print('pred.shape', pred.shape)
            correct_close += pred.eq(label_t.data).cpu().sum()
            out_t = F.softmax(out_t, dim=1)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                out_open = Cs[1](feat)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                c2_softmax_all_predict_list.append(out_open.reshape(out_t.size(0), -1))
                
                tmp_range = torch.arange(out_t.size(0)).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > thr_open)[0]
            pred_unk_list.append(pred_unk)
            
            pred[ind_unk] = open_class
            predict_list.append(pred)
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]

    return c1_softmax_all_predict_list, c2_softmax_all_predict_list, feature_list, predict_list, path_list, pred_unk_list, label_all, pred_open, pred_ent, pred_all, size, correct, correct_close



def inference_amlp_v2_only_c2(dataset_test, G_mlp, Cs, class_list, entropy, thr, per_class_correct, per_class_num, correct_close, correct, size, thr_open=0.5, sourse_loader_bool=False):
    
    G, mlp = G_mlp
    feature_list = []
    predict_list = []
    path_list = []
    pred_unk_list = []
    c1_softmax_all_predict_list = []
    c2_softmax_all_predict_list = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            # if sourse_loader_bool == False:
            img_t, label_t, path_t = data[0].cuda(), data[1].cuda(), data[2]
            # else:
            #     img_t, label_t, path_t = data[0][0].cuda(), data[1].cuda(), str(data[2])

            path_list += path_t
            feat = mlp(G(img_t))
            # feat = (feat)
            # out_t = Cs[0](feat)
            out_t = Cs[1](feat).view(feat.shape[0], 2, -1)[:,1,:]
            c1_softmax_all_predict_list.append(torch.softmax(out_t, dim=1))
            # import pdb; pdb.set_trace()
            if batch_idx == 0:
                open_class = int(out_t.size(1))
                class_list.append(open_class)
            pred = out_t.data.max(1)[1]
            feature_list.append(feat)
            # predict_list.append(pred)
            # print('out_t.shape', out_t.shape)
            # print('label_t.shape', label_t.shape)
            # print('pred.shape', pred.shape)
            correct_close += pred.eq(label_t.data).cpu().sum()
            out_t = F.softmax(out_t, dim=1)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                out_open = Cs[1](feat)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                c2_softmax_all_predict_list.append(out_open.reshape(out_t.size(0), -1))
                
                tmp_range = torch.arange(out_t.size(0)).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > thr_open)[0]
            pred_unk_list.append(pred_unk)
            
            pred[ind_unk] = open_class
            predict_list.append(pred)
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]

    return c1_softmax_all_predict_list, c2_softmax_all_predict_list, feature_list, predict_list, path_list, pred_unk_list, label_all, pred_open, pred_ent, pred_all, size, correct, correct_close



def inference_amlp_v2_only_c2_maxcompair(dataset_test, G_mlp, Cs, class_list, entropy, thr, per_class_correct, per_class_num, correct_close, correct, size, thr_open=0.5, sourse_loader_bool=False):
    
    G, mlp = G_mlp
    feature_list = []
    predict_list = []
    path_list = []
    pred_unk_list = []
    c1_softmax_all_predict_list = []
    c2_softmax_all_predict_list = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            # if sourse_loader_bool == False:
            img_t, label_t, path_t = data[0].cuda(), data[1].cuda(), data[2]
            # else:
            #     img_t, label_t, path_t = data[0][0].cuda(), data[1].cuda(), str(data[2])

            path_list += path_t
            feat = mlp(G(img_t))
            # feat = (feat)
            # out_t = Cs[0](feat)
            out_t = Cs[1](feat).view(feat.shape[0], 2, -1)[:,1,:]
            c1_softmax_all_predict_list.append(torch.softmax(out_t, dim=1))
            # import pdb; pdb.set_trace()
            if batch_idx == 0:
                open_class = int(out_t.size(1))
                class_list.append(open_class)
            pred = out_t.data.max(1)[1]
            feature_list.append(feat)
            # predict_list.append(pred)
            # print('out_t.shape', out_t.shape)
            # print('label_t.shape', label_t.shape)
            # print('pred.shape', pred.shape)
            correct_close += pred.eq(label_t.data).cpu().sum()
            out_t = F.softmax(out_t, dim=1)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                # out_open = Cs[1](feat)
                # out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                # c2_softmax_all_predict_list.append(out_open.reshape(out_t.size(0), -1))               
                # tmp_range = torch.arange(out_t.size(0)).long().cuda()
                # pred_unk = out_open[tmp_range, 0, pred]
                out_open = Cs[1](feat)
                out_open= F.softmax(out_open, 1)
                out_open = out_open.view(out_t.size(0), 2, -1)                
                c2_softmax_all_predict_list.append(out_open.reshape(out_t.size(0), -1))               
                tmp_range = torch.arange(out_t.size(0)).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                # ind_unk = np.where(pred_unk.data.cpu().numpy() > thr_open)[0]
                ind_unk = torch.tensor(np.where(
                    np.max(out_open.data.cpu().numpy()[:, 0, :], axis=1) >= \
                    np.max(out_open.data.cpu().numpy()[:, 1, :], axis=1)
                    )[0])

                # unknown_cont = (out_open.data.cpu().numpy()[:, 0, :] > out_open.data.cpu().numpy()[:, 1, :]).sum(axis=1)
                # print(unknown_cont)
            pred_unk_list.append(pred_unk)
            
            pred[ind_unk] = open_class
            predict_list.append(pred)
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]

    return c1_softmax_all_predict_list, c2_softmax_all_predict_list, feature_list, predict_list, path_list, pred_unk_list, label_all, pred_open, pred_ent, pred_all, size, correct, correct_close



def inference_amlp_v2_only_c2_maxcompair_eachclass(dataset_test, G_mlp, Cs, class_list, entropy, thr, per_class_correct, per_class_num, correct_close, correct, size, thr_open=0.5, sourse_loader_bool=False):
    
    G, mlp = G_mlp
    feature_list = []
    predict_list = []
    path_list = []
    pred_unk_list = []
    c1_softmax_all_predict_list = []
    c2_softmax_all_predict_list = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            # if sourse_loader_bool == False:
            img_t, label_t, path_t = data[0].cuda(), data[1].cuda(), data[2]
            # else:
            #     img_t, label_t, path_t = data[0][0].cuda(), data[1].cuda(), str(data[2])

            path_list += path_t
            feat = mlp(G(img_t))
            # feat = (feat)
            # out_t = Cs[0](feat)
            out_t = Cs[1](feat).view(feat.shape[0], 2, -1)[:,1,:]
            c1_softmax_all_predict_list.append(torch.softmax(out_t, dim=1))
            # import pdb; pdb.set_trace()
            if batch_idx == 0:
                open_class = int(out_t.size(1))
                class_list.append(open_class)
            pred = out_t.data.max(1)[1]
            feature_list.append(feat)
            # predict_list.append(pred)
            # print('out_t.shape', out_t.shape)
            # print('label_t.shape', label_t.shape)
            # print('pred.shape', pred.shape)
            correct_close += pred.eq(label_t.data).cpu().sum()
            out_t = F.softmax(out_t, dim=1)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                # out_open = Cs[1](feat)
                # out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                # c2_softmax_all_predict_list.append(out_open.reshape(out_t.size(0), -1))               
                # tmp_range = torch.arange(out_t.size(0)).long().cuda()
                # pred_unk = out_open[tmp_range, 0, pred]
                out_open = Cs[1](feat)
                out_open= F.softmax(out_open, 1)
                out_open = out_open.view(out_t.size(0), 2, -1)                
                c2_softmax_all_predict_list.append(out_open.reshape(out_t.size(0), -1))               
                tmp_range = torch.arange(out_t.size(0)).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                # ind_unk = np.where(pred_unk.data.cpu().numpy() > thr_open)[0]
                # ind_unk = torch.tensor(np.where(
                #     np.max(out_open.data.cpu().numpy()[:, 0, :], axis=1) >= \
                #     np.max(out_open.data.cpu().numpy()[:, 1, :], axis=1)
                #     )[0])

                unknown_cont = (out_open.data.cpu().numpy()[:, 0, :] > out_open.data.cpu().numpy()[:, 1, :]).sum(axis=1)
                ind_unk = (unknown_cont==15)
                # print(unknown_cont)
            pred_unk_list.append(pred_unk)
            
            pred[ind_unk] = open_class
            predict_list.append(pred)
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]

    return c1_softmax_all_predict_list, c2_softmax_all_predict_list, feature_list, predict_list, path_list, pred_unk_list, label_all, pred_open, pred_ent, pred_all, size, correct, correct_close


def inference_only_c2(dataset_test, G, Cs, class_list, entropy, thr, per_class_correct, per_class_num, correct_close, correct, size, thr_open=0.5, sourse_loader_bool=False):
    
    feature_list = []
    predict_list = []
    path_list = []
    pred_unk_list = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            if sourse_loader_bool == False:
                img_t, label_t, path_t = data[0].cuda(), data[1].cuda(), data[2]
            else:
                img_t, label_t, path_t = data[0][0].cuda(), data[1].cuda(), str(data[2])

            path_list += path_t
            feat = G(img_t)
            # feat = (feat)
            out_t = Cs[1](feat).view(feat.shape[0], 2, -1)[:,1,:]
            # print(out_t)
            # print('out_t.shape', out_t.shape)
            # [:, 1]
            if batch_idx == 0:
                open_class = int(out_t.size(1))
                class_list.append(open_class)
            pred = out_t.data.max(1)[1]
            feature_list.append(feat)
            # predict_list.append(pred)
            # print('out_t.shape', out_t.shape)
            # print('label_t.shape', label_t.shape)
            # print('pred.shape', pred.shape)
            correct_close += pred.eq(label_t.data).cpu().sum()
            out_t = F.softmax(out_t, dim=1)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                out_open = Cs[1](feat)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                tmp_range = torch.arange(out_t.size(0)).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > thr_open)[0]
            pred_unk_list.append(pred_unk)
            
            pred[ind_unk] = open_class
            predict_list.append(pred)
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]
    # import pdb; pdb.set_trace()

    return feature_list, predict_list, path_list, pred_unk_list, label_all, pred_open, pred_ent, pred_all, size, correct, correct_close



def inference_half(dataset_test, G, Cs, class_list, entropy, thr, per_class_correct, per_class_num, correct_close, correct, size, thr_open=0.5, sourse_loader_bool=False):
    
    feature_list = []
    predict_list = []
    path_list = []
    pred_unk_list = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            if sourse_loader_bool == False:
                img_t, label_t, path_t = data[0].type(torch.HalfTensor).cuda(), data[1].cuda(), data[2]
            else:
                # import pdb; pdb.set_trace()
                img_t, label_t, path_t = data[0][0].type(torch.HalfTensor).cuda(), data[1].cuda(), str(data[2])
                # print('img_t.shape', img_t.shape)
                # print('label_t.shape', label_t.shape)

            path_list += path_t
            feat = G(img_t)
            # feat = (feat)
            out_t = Cs[0](feat)
            if batch_idx == 0:
                open_class = int(out_t.size(1))
                class_list.append(open_class)
            pred = out_t.data.max(1)[1]
            feature_list.append(feat)
            # predict_list.append(pred)
            # print('out_t.shape', out_t.shape)
            # print('label_t.shape', label_t.shape)
            # print('pred.shape', pred.shape)
            correct_close += pred.eq(label_t.data).cpu().sum()
            out_t = F.softmax(out_t, dim=1)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                # print(feat)
                # print(Cs[1])
                # import pdb; pdb.set_trace()
                # print(Cs[1].fc1.weight.data)
                
                out_open =Cs[1](feat)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                tmp_range = torch.arange(out_t.size(0)).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > thr_open)[0]
            pred_unk_list.append(pred_unk)
            
            pred[ind_unk] = open_class
            predict_list.append(pred)
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]

    return feature_list, predict_list, path_list, pred_unk_list, label_all, pred_open, pred_ent, pred_all, size, correct, correct_close



def test(step, source_loader, dataset_test, name, n_share, G, Cs, mlp,
         open_class = None, open=False, entropy=False, thr=0.5, argsSName=''):
    G.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    print('-->')
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    
    (
        feature_list, predict_list, path_list,
        pred_unk_list, label_all, pred_open,
        pred_ent, pred_all, size,
        correct, correct_close
    ) = inference(
        dataset_test, G, Cs, class_list, entropy, thr,
        per_class_correct, per_class_num,
        correct_close, correct, size, thr_open=0.5,
    )

    print('-->')
    correct_s = 0
    correct_close_s = 0
    size_s = 0
    per_class_num_s = np.zeros((n_share + 1))
    per_class_correct_s = np.zeros((n_share + 1)).astype(np.float32)
    class_list_s = [i for i in range(n_share)]
    (
        feature_list_s, predict_list_s, path_list_s,
        pred_unk_list_s, label_all_s, pred_open_s,
        pred_ent_s, pred_all_s, size_s,
        correct_s, correct_close_s
    ) = inference(
        source_loader, G, Cs, class_list_s, entropy, thr,
        per_class_correct_s, per_class_num_s,
        correct_close_s, correct_s, size_s, sourse_loader_bool=True
    )

    print('-->')
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        # roc = roc_auc_score(Y_test[:, -1], pred_open)
        # roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        # roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_acc, mean_score = select_threshold(
            pred_all, pred_open, label_all, class_list)

    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    
    # import pdb; pdb.set_trace()
    print('-->')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    output = ["step %s"%step,
              "closed perclass", list(per_class_acc),
              "acc per class %s" % (float(per_class_acc.mean())),
              "acc %s" % float(acc_all),
              "acc close all %s" % float(acc_close_all),
              "h score %s" % float(h_score),
            #   "roc %s"% float(roc),
            #   "roc ent %s"% float(roc_ent),
            #   "roc softmax %s"% float(roc_softmax),
              "best hscore %s"% float(best_acc),
              "best thr %s"% float(best_th)]
    logger.info(output)

    feature_test = torch.cat(feature_list).detach().cpu().numpy()
    pred_unk = torch.cat(pred_unk_list).detach().cpu().numpy()
    pred_all = torch.cat(predict_list).detach().cpu().numpy()
    # print(feature_test)

    feature_test_s = torch.cat(feature_list_s).detach().cpu().numpy()
    pred_unk_s = torch.cat(pred_unk_list_s).detach().cpu().numpy()
    pred_all_s = torch.cat(predict_list_s).detach().cpu().numpy()

    feature_test_t_s = np.concatenate([feature_test, feature_test_s])
    num_t_sample = feature_test.shape[0]

    print('feature_test_t_s.shape', feature_test_t_s.shape)

    if feature_test_t_s.shape[0] < 50000:

        feature_test_2d_ts = TSNE().fit(feature_test_t_s)

        feature_test_2d = feature_test_2d_ts[:num_t_sample]
        plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=label_all, s=1)
        # import pdb; pdb.set_trace()
        path_uuidstr_t = str(uuid.uuid4())
        plt.colorbar()
        plt.savefig('img/t_label{}.png'.format(path_uuidstr_t))
        plt.close()
        plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=pred_all, s=1)
        plt.colorbar()
        plt.savefig('img/t_pred{}.png'.format(path_uuidstr_t))
        plt.close()

        feature_test_2d_s = feature_test_2d_ts[num_t_sample:]
        plt.scatter(x=feature_test_2d_s[:,0], y=feature_test_2d_s[:,1], c=label_all_s, s=1)
        # import pdb; pdb.set_trace()
        path_uuidstr_s = str(uuid.uuid4())
        plt.colorbar()
        plt.savefig('img/s_label{}.png'.format(path_uuidstr_s))
        plt.close()
        plt.scatter(x=feature_test_2d_s[:,0], y=feature_test_2d_s[:,1], c=pred_all_s, s=1)
        plt.colorbar()
        plt.savefig('img/s_pred{}.png'.format(path_uuidstr_s))
        plt.close()

        data_path = pd.DataFrame()
        data_path['path'] = path_list
        data_path['pre'] = pred_all
        data_path['lab'] = label_all
        data_path['tsne_1'] = feature_test_2d[:,0]
        data_path['tsne_2'] = feature_test_2d[:,1]
        data_path['pred_unk'] = pred_unk

        np.save('file_save/feature_test{}'.format(path_uuidstr_t), feature_test)
        np.save('file_save/feature_test_s{}'.format(path_uuidstr_t), feature_test_s)
        np.save('file_save/pred_all_s{}'.format(path_uuidstr_t), pred_all_s)
        np.save('file_save/pred_all{}'.format(path_uuidstr_t), pred_all)

        wandb.log({
            'epoch': step,
            'entropy'+str(entropy)+'_best best_th':float(best_th),
            'entropy'+str(entropy)+'_best hscore':float(best_acc),
            'entropy'+str(entropy)+'_t_label':wandb.Image('img/t_label{}.png'.format(path_uuidstr_t)),
            'entropy'+str(entropy)+'_t_pred':wandb.Image( 'img/t_pred{}.png'.format(path_uuidstr_t)),
            'entropy'+str(entropy)+'_s_label':wandb.Image('img/s_label{}.png'.format(path_uuidstr_s)),
            'entropy'+str(entropy)+'_s_pred':wandb.Image( 'img/s_pred{}.png'.format(path_uuidstr_s)),
            'feature_test':'file_save/feature_test{}'.format(path_uuidstr_t),
            'feature_test_s':'file_save/feature_test_s{}'.format(path_uuidstr_t),
            'pred_all_s':'file_save/pred_all_s{}'.format(path_uuidstr_t),
            'pred_all':'file_save/pred_all{}'.format(path_uuidstr_t),
            'entropy'+str(entropy)+'_acc_all':acc_all,
            'entropy'+str(entropy)+'_h_score':h_score,
            'entropy'+str(entropy)+'_pre_table': wandb.Table(data=data_path, columns=['path', 'pre', 'label', 'pred_unk'])
        })
    else:
        wandb.log({
            'epoch': step,
            'entropy'+str(entropy)+'_best best_th':float(best_th),
            'entropy'+str(entropy)+'_best hscore':float(best_acc),
            'entropy'+str(entropy)+'_acc_all':acc_all,
            'entropy'+str(entropy)+'_h_score':h_score,
        })

    # wandb.save('img/t_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/t_pred{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_pred{}.png'.format(path_uuidstr_s))

    return acc_all, h_score


def test_amlp(step, test_loader_s, dataset_test, name, n_share, G_mlp, Cs, mlp,
         open_class = None, open=False, entropy=False, thr=0.5, argsSName='', save_wandb=0):
    G, mlp = G_mlp
    G.eval()
    mlp.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    print('-->')
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    
    (
        feature_list, predict_list, path_list,
        pred_unk_list, label_all, pred_open,
        pred_ent, pred_all, size,
        correct, correct_close
    ) = inference_amlp(
        dataset_test, [G, mlp], Cs, class_list, entropy, thr,
        per_class_correct, per_class_num,
        correct_close, correct, size, thr_open=0.5,
    )

    print('-->')
    correct_s = 0
    correct_close_s = 0
    size_s = 0
    per_class_num_s = np.zeros((n_share + 1))
    per_class_correct_s = np.zeros((n_share + 1)).astype(np.float32)
    class_list_s = [i for i in range(n_share)]
    (
        feature_list_s, predict_list_s, path_list_s,
        pred_unk_list_s, label_all_s, pred_open_s,
        pred_ent_s, pred_all_s, size_s,
        correct_s, correct_close_s
    ) = inference_amlp(
        test_loader_s, [G, mlp], Cs, class_list_s, entropy, thr,
        per_class_correct_s, per_class_num_s,
        correct_close_s, correct_s, size_s, sourse_loader_bool=True
    )

    print('-->')
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        # roc = roc_auc_score(Y_test[:, -1], pred_open)
        # roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        # roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_acc, mean_score = select_threshold(
            pred_all, pred_open, label_all, class_list)

    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    
    # import pdb; pdb.set_trace()
    print('-->')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    output = ["step %s"%step,
              "closed perclass", list(per_class_acc),
              "acc per class %s" % (float(per_class_acc.mean())),
              "acc %s" % float(acc_all),
              "acc close all %s" % float(acc_close_all),
              "h score %s" % float(h_score),
            #   "roc %s"% float(roc),
            #   "roc ent %s"% float(roc_ent),
            #   "roc softmax %s"% float(roc_softmax),
              "best hscore %s"% float(best_acc),
              "best thr %s"% float(best_th)]
    logger.info(output)

    feature_test = torch.cat(feature_list).detach().cpu().numpy()
    pred_unk = torch.cat(pred_unk_list).detach().cpu().numpy()
    pred_all = torch.cat(predict_list).detach().cpu().numpy()
    # print(feature_test)

    feature_test_s = torch.cat(feature_list_s).detach().cpu().numpy()
    pred_unk_s = torch.cat(pred_unk_list_s).detach().cpu().numpy()
    pred_all_s = torch.cat(predict_list_s).detach().cpu().numpy()

    feature_test_t_s = np.concatenate([feature_test, feature_test_s])
    num_t_sample = feature_test.shape[0]

    print('feature_test_t_s.shape', feature_test_t_s.shape)

    if feature_test_t_s.shape[0] < 50000:

        feature_test_2d_ts = TSNE().fit(feature_test_t_s)

        feature_test_2d = feature_test_2d_ts[:num_t_sample]
        plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=label_all, s=1)
        # import pdb; pdb.set_trace()
        path_uuidstr_t = str(uuid.uuid4())
        plt.colorbar()
        plt.savefig('img/t_label{}.png'.format(path_uuidstr_t))
        plt.close()
        plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=pred_all, s=1)
        plt.colorbar()
        plt.savefig('img/t_pred{}.png'.format(path_uuidstr_t))
        plt.close()

        feature_test_2d_s = feature_test_2d_ts[num_t_sample:]
        plt.scatter(x=feature_test_2d_s[:,0], y=feature_test_2d_s[:,1], c=label_all_s, s=1)
        # import pdb; pdb.set_trace()
        path_uuidstr_s = str(uuid.uuid4())
        plt.colorbar()
        plt.savefig('img/s_label{}.png'.format(path_uuidstr_s))
        plt.close()
        plt.scatter(x=feature_test_2d_s[:,0], y=feature_test_2d_s[:,1], c=pred_all_s, s=1)
        plt.colorbar()
        plt.savefig('img/s_pred{}.png'.format(path_uuidstr_s))
        plt.close()

        # data_path = pd.DataFrame()
        # data_path['path'] = path_list
        # data_path['pre'] = pred_all
        # data_path['lab'] = label_all
        # data_path['tsne_1'] = feature_test_2d[:,0]
        # data_path['tsne_2'] = feature_test_2d[:,1]
        # data_path['pred_unk'] = pred_unk

        # import pdb; pdb.set_trace()
        data_path = pd.DataFrame()
        data_path['path'] = path_list + path_list_s
        data_path['pre'] = np.concatenate([pred_all, pred_all_s], axis=0)
        data_path['lab'] =  np.concatenate([label_all, label_all_s], axis=0) # label_all
        data_path['tsne_1'] = feature_test_2d_ts[:,0]
        data_path['tsne_2'] = feature_test_2d_ts[:,1]
        data_path['pred_unk'] = np.concatenate([pred_unk, pred_unk_s], axis=0) #pred_unk
        s_or_t = np.zeros(data_path.shape[0])
        s_or_t[num_t_sample:] = 1
        data_path['s_or_t'] = s_or_t

        np.save('file_save/feature_test{}'.format(path_uuidstr_t), feature_test)
        np.save('file_save/feature_test_s{}'.format(path_uuidstr_t), feature_test_s)
        np.save('file_save/pred_all_s{}'.format(path_uuidstr_t), pred_all_s)
        np.save('file_save/pred_all{}'.format(path_uuidstr_t), pred_all)
        
        wandb.log({
            'epoch': step,
            'entropy'+str(entropy)+'_best best_th':float(best_th),
            'entropy'+str(entropy)+'_best hscore':float(best_acc),
            'entropy'+str(entropy)+'_t_label':wandb.Image('img/t_label{}.png'.format(path_uuidstr_t)),
            'entropy'+str(entropy)+'_t_pred':wandb.Image( 'img/t_pred{}.png'.format(path_uuidstr_t)),
            'entropy'+str(entropy)+'_s_label':wandb.Image('img/s_label{}.png'.format(path_uuidstr_s)),
            'entropy'+str(entropy)+'_s_pred':wandb.Image( 'img/s_pred{}.png'.format(path_uuidstr_s)),
            'feature_test':'file_save/feature_test{}'.format(path_uuidstr_t),
            'feature_test_s':'file_save/feature_test_s{}'.format(path_uuidstr_t),
            'pred_all_s':'file_save/pred_all_s{}'.format(path_uuidstr_t),
            'pred_all':'file_save/pred_all{}'.format(path_uuidstr_t),
            'entropy'+str(entropy)+'_acc_all':acc_all,
            'entropy'+str(entropy)+'_h_score':h_score,
            'entropy'+str(entropy)+'_pre_table': wandb.Table(data=data_path, columns=['path', 'pre', 'label', 'pred_unk'])
        })
    else:
        wandb.log({
            'epoch': step,
            'entropy'+str(entropy)+'_best best_th':float(best_th),
            'entropy'+str(entropy)+'_best hscore':float(best_acc),
            'entropy'+str(entropy)+'_acc_all':acc_all,
            'entropy'+str(entropy)+'_h_score':h_score,
        })

    # wandb.save('img/t_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/t_pred{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_pred{}.png'.format(path_uuidstr_s))

    return acc_all, h_score



def test_amlp_q(step, test_loader_s, dataset_test, name, n_share, G_mlp, Cs, mlp,
         open_class = None, open=False, entropy=False, thr=0.5, argsSName='',):
    G, mlp = G_mlp
    G.eval()
    mlp.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    print('-->')
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    
    (
        feature_list, predict_list, path_list,
        pred_unk_list, label_all, pred_open,
        pred_ent, pred_all, size,
        correct, correct_close
    ) = inference_amlp(
        dataset_test, [G, mlp], Cs, class_list, entropy, thr,
        per_class_correct, per_class_num,
        correct_close, correct, size, thr_open=0.5,
    )

    print('-->')
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        # roc = roc_auc_score(Y_test[:, -1], pred_open)
        # roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        # roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_acc, mean_score = select_threshold(
            pred_all, pred_open, label_all, class_list)

    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    
    # import pdb; pdb.set_trace()
    print('-->')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    output = ["step %s"%step,
              "closed perclass", list(per_class_acc),
              "acc per class %s" % (float(per_class_acc.mean())),
              "acc %s" % float(acc_all),
              "acc close all %s" % float(acc_close_all),
              "h score %s" % float(h_score),
            #   "roc %s"% float(roc),
            #   "roc ent %s"% float(roc_ent),
            #   "roc softmax %s"% float(roc_softmax),
              "best hscore %s"% float(best_acc),
              "best thr %s"% float(best_th)]
    logger.info(output)

    # feature_test_2d_ts = TSNE().fit(feature_test_t_s)

    # feature_test_2d = feature_test_2d_ts[:num_t_sample]
    # plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=label_all, s=1)
    # # import pdb; pdb.set_trace()
    # path_uuidstr_t = str(uuid.uuid4())
    # plt.colorbar()
    # plt.savefig('img/t_label{}.png'.format(path_uuidstr_t))
    # plt.close()
    # plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=pred_all, s=1)
    # plt.colorbar()
    # plt.savefig('img/t_pred{}.png'.format(path_uuidstr_t))
    # plt.close()


    wandb.log({
        'epoch': step,
        'closeacc': acc_close_all,
        'best best_th':float(best_th),
        'best hscore':float(best_acc),
        'acc_all':acc_all,
        'h_score':h_score,
    })

    # wandb.save('img/t_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/t_pred{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_pred{}.png'.format(path_uuidstr_s))

    return acc_all, h_score


def test_amlp_q_oda(step, test_loader_s, dataset_test, name, n_share, G_mlp, Cs, mlp,
         open_class = None, open=False, entropy=False, thr=0.5, argsSName='',):
    G, mlp = G_mlp
    G.eval()
    mlp.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    print('-->')
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    
    (
        feature_list, predict_list, path_list,
        pred_unk_list, label_all, pred_open,
        pred_ent, pred_all, size,
        correct, correct_close
    ) = inference_amlp(
        dataset_test, [G, mlp], Cs, class_list, entropy, thr,
        per_class_correct, per_class_num,
        correct_close, correct, size, thr_open=0.5,
    )

    print('-->')
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        # roc = roc_auc_score(Y_test[:, -1], pred_open)
        # roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        # roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_acc, mean_score = select_threshold(
            pred_all, pred_open, label_all, class_list)

    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.0
        best_acc = 0.0
        roc_softmax = 0.0
    
    # import pdb; pdb.set_trace()
    print('-->')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    output = ["step %s"%step,
              "closed perclass", list(per_class_acc),
              "acc per class %s" % (float(per_class_acc.mean())),
              "acc %s" % float(acc_all),
              "acc close all %s" % float(acc_close_all),
              "h score %s" % float(h_score),
            #   "roc %s"% float(roc),
            #   "roc ent %s"% float(roc_ent),
            #   "roc softmax %s"% float(roc_softmax),
              "best hscore %s"% float(best_acc),
              "best thr %s"% float(best_th)]
    logger.info(output)

    # import pdb; pdb.set_trace()
    pred_all = torch.cat(predict_list).detach().cpu().numpy()
    emb_all = torch.cat(feature_list).detach().cpu().numpy()
    label_all = label_all

    r_index = np.random.permutation(pred_all.shape[0])[:10000]
    pred_all = pred_all[r_index]
    emb_all = emb_all[r_index]
    label_all = label_all[r_index]

    feature_test_2d = TSNE().fit(emb_all)
    # feature_test_2d = umap.UMAP().fit(torch.cat(feature_list).detach().cpu().numpy())

    plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=label_all, s=1)
    path_uuidstr_t = str(uuid.uuid4())
    plt.colorbar()
    plt.savefig('img/t_label{}.png'.format(path_uuidstr_t))
    plt.close()
    plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=pred_all, s=1)
    plt.colorbar()
    plt.savefig('img/t_pred{}.png'.format(path_uuidstr_t))
    plt.close()


    wandb.log({
        'epoch': step,
        'closeacc': acc_close_all,
        'best best_th':float(best_th),
        'best hscore':float(best_acc),
        'acc_all':acc_all,
        'h_score':h_score,
        'fig_t_pre':wandb.Image('img/t_pred{}.png'.format(path_uuidstr_t)),
        'fig_t_lab':wandb.Image('img/t_label{}.png'.format(path_uuidstr_t)),
    })

    # wandb.save('img/t_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/t_pred{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_pred{}.png'.format(path_uuidstr_s))

    return acc_all, h_score


def test_amlp_v2(step, test_loader_s, dataset_test, name, n_share, G_mlp, Cs, mlp,
         open_class = None, open=False, entropy=False, thr=0.5, argsSName=''):
    G, mlp = G_mlp
    G.eval()
    mlp.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    print('-->')
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    
    (
        softmax_all_predict_list, c2_softmax_all_predict_list, feature_list, predict_list, path_list,
        pred_unk_list, label_all, pred_open,
        pred_ent, pred_all, size,
        correct, correct_close
    ) = inference_amlp_v2(
        dataset_test, [G, mlp], Cs, class_list, entropy, thr,
        per_class_correct, per_class_num,
        correct_close, correct, size, thr_open=0.5,
    )

    print('-->')
    correct_s = 0
    correct_close_s = 0
    size_s = 0
    per_class_num_s = np.zeros((n_share + 1))
    per_class_correct_s = np.zeros((n_share + 1)).astype(np.float32)
    class_list_s = [i for i in range(n_share)]
    (
        softmax_all_predict_list_s, c2_softmax_all_predict_list_s, feature_list_s, predict_list_s, path_list_s,
        pred_unk_list_s, label_all_s, pred_open_s,
        pred_ent_s, pred_all_s, size_s,
        correct_s, correct_close_s
    ) = inference_amlp_v2(
        test_loader_s, [G, mlp], Cs, class_list_s, entropy, thr,
        per_class_correct_s, per_class_num_s,
        correct_close_s, correct_s, size_s, sourse_loader_bool=True
    )

    print('-->')
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        # roc = roc_auc_score(Y_test[:, -1], pred_open)
        # roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        # roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_acc, mean_score = select_threshold(
            pred_all, pred_open, label_all, class_list)
    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    
    # import pdb; pdb.set_trace()
    print('-->')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    output = ["step %s"%step,
              "closed perclass", list(per_class_acc),
              "acc per class %s" % (float(per_class_acc.mean())),
              "acc %s" % float(acc_all),
              "acc close all %s" % float(acc_close_all),
              "h score %s" % float(h_score),
            #   "roc %s"% float(roc),
            #   "roc ent %s"% float(roc_ent),
            #   "roc softmax %s"% float(roc_softmax),
              "best hscore %s"% float(best_acc),
              "best thr %s"% float(best_th)]
    logger.info(output)

    softmax_all_predict = torch.cat(softmax_all_predict_list).detach().cpu().numpy()
    c2_softmax_all_predict = torch.cat(c2_softmax_all_predict_list).detach().cpu().numpy()
    feature_test = torch.cat(feature_list).detach().cpu().numpy()
    pred_unk = torch.cat(pred_unk_list).detach().cpu().numpy()
    pred_all = torch.cat(predict_list).detach().cpu().numpy()
    # print(feature_test)

    softmax_all_predict_s = torch.cat(softmax_all_predict_list_s).detach().cpu().numpy()
    c2_softmax_all_predict_s = torch.cat(c2_softmax_all_predict_list_s).detach().cpu().numpy()
    feature_test_s = torch.cat(feature_list_s).detach().cpu().numpy()
    pred_unk_s = torch.cat(pred_unk_list_s).detach().cpu().numpy()
    pred_all_s = torch.cat(predict_list_s).detach().cpu().numpy()

    feature_test_t_s = np.concatenate([feature_test, feature_test_s])
    softmax_all = np.concatenate([softmax_all_predict, softmax_all_predict_s])
    c2_softmax_all = np.concatenate([c2_softmax_all_predict, c2_softmax_all_predict_s])
    num_t_sample = feature_test.shape[0]

    print('feature_test_t_s.shape', feature_test_t_s.shape)

    if feature_test_t_s.shape[0] < 50000:

        feature_test_2d_ts = TSNE().fit(feature_test_t_s)

        feature_test_2d = feature_test_2d_ts[:num_t_sample]
        plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=label_all, s=1)
        # import pdb; pdb.set_trace()
        path_uuidstr_t = str(uuid.uuid4())
        plt.colorbar()
        plt.savefig('img/t_label{}.png'.format(path_uuidstr_t))
        plt.close()
        plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=pred_all, s=1)
        plt.colorbar()
        plt.savefig('img/t_pred{}.png'.format(path_uuidstr_t))
        plt.close()

        feature_test_2d_s = feature_test_2d_ts[num_t_sample:]
        plt.scatter(x=feature_test_2d_s[:,0], y=feature_test_2d_s[:,1], c=label_all_s, s=1)
        # import pdb; pdb.set_trace()
        path_uuidstr_s = str(uuid.uuid4())
        plt.colorbar()
        plt.savefig('img/s_label{}.png'.format(path_uuidstr_s))
        plt.close()
        plt.scatter(x=feature_test_2d_s[:,0], y=feature_test_2d_s[:,1], c=pred_all_s, s=1)
        plt.colorbar()
        plt.savefig('img/s_pred{}.png'.format(path_uuidstr_s))
        plt.close()

        # import pdb; pdb.set_trace()
        data_path = pd.DataFrame()
        data_path['path'] = path_list + path_list_s
        data_path['pre'] = np.concatenate([pred_all, pred_all_s], axis=0)
        data_path['lab'] =  np.concatenate([label_all, label_all_s], axis=0) # label_all
        data_path['tsne_1'] = feature_test_2d_ts[:,0]
        data_path['tsne_2'] = feature_test_2d_ts[:,1]
        data_path['pred_unk'] = np.concatenate([pred_unk, pred_unk_s,], axis=0) #pred_unk
        data_path = pd.concat([
            data_path, 
            pd.DataFrame(softmax_all, columns=['soft_{}'.format(i) for i in range(softmax_all.shape[1])]),
            pd.DataFrame(c2_softmax_all, columns=['c2_soft_{}'.format(i) for i in range(c2_softmax_all.shape[1])])
            ],
            axis=1)
        
        s_or_t = np.zeros(data_path.shape[0])
        s_or_t[num_t_sample:] = 1
        data_path['s_or_t'] = s_or_t

        np.save('file_save/feature_test{}'.format(path_uuidstr_t), feature_test)
        np.save('file_save/feature_test_s{}'.format(path_uuidstr_t), feature_test_s)
        np.save('file_save/pred_all_s{}'.format(path_uuidstr_t), pred_all_s)
        np.save('file_save/pred_all{}'.format(path_uuidstr_t), pred_all)
        
        wandb.log({
            'epoch': step,
            'entropy'+str(entropy)+'_best best_th':float(best_th),
            'entropy'+str(entropy)+'_best hscore':float(best_acc),
            'entropy'+str(entropy)+'_t_label':wandb.Image('img/t_label{}.png'.format(path_uuidstr_t)),
            'entropy'+str(entropy)+'_t_pred':wandb.Image( 'img/t_pred{}.png'.format(path_uuidstr_t)),
            'entropy'+str(entropy)+'_s_label':wandb.Image('img/s_label{}.png'.format(path_uuidstr_s)),
            'entropy'+str(entropy)+'_s_pred':wandb.Image( 'img/s_pred{}.png'.format(path_uuidstr_s)),
            'feature_test':'file_save/feature_test{}'.format(path_uuidstr_t),
            'feature_test_s':'file_save/feature_test_s{}'.format(path_uuidstr_t),
            'pred_all_s':'file_save/pred_all_s{}'.format(path_uuidstr_t),
            'pred_all':'file_save/pred_all{}'.format(path_uuidstr_t),
            'entropy'+str(entropy)+'_acc_all':acc_all,
            'entropy'+str(entropy)+'_h_score':h_score,
            'entropy'+str(entropy)+'_pre_table': wandb.Table(data=data_path, columns=['path', 'pre', 'label', 'pred_unk'])
        })
    else:
        wandb.log({
            'epoch': step,
            'entropy'+str(entropy)+'_best best_th':float(best_th),
            'entropy'+str(entropy)+'_best hscore':float(best_acc),
            'entropy'+str(entropy)+'_acc_all':acc_all,
            'entropy'+str(entropy)+'_h_score':h_score,
        })

    # wandb.save('img/t_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/t_pred{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_pred{}.png'.format(path_uuidstr_s))

    return acc_all, h_score



def test_amlp_v2_only_c2(step, test_loader_s, dataset_test, name, n_share, G_mlp, Cs, mlp,
         open_class = None, open=False, entropy=False, thr=0.5, argsSName=''):
    G, mlp = G_mlp
    G.eval()
    mlp.eval()

    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    print('-->')
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    
    (
        softmax_all_predict_list, c2_softmax_all_predict_list, feature_list, predict_list, path_list,
        pred_unk_list, label_all, pred_open,
        pred_ent, pred_all, size,
        correct, correct_close
    ) = inference_amlp_v2_only_c2(
        dataset_test, [G, mlp], Cs, class_list, entropy, thr,
        per_class_correct, per_class_num,
        correct_close, correct, size, thr_open=0.5,
    )

    print('-->')
    correct_s = 0
    correct_close_s = 0
    size_s = 0
    per_class_num_s = np.zeros((n_share + 1))
    per_class_correct_s = np.zeros((n_share + 1)).astype(np.float32)
    class_list_s = [i for i in range(n_share)]
    (
        softmax_all_predict_list_s, c2_softmax_all_predict_list_s, feature_list_s, predict_list_s, path_list_s,
        pred_unk_list_s, label_all_s, pred_open_s,
        pred_ent_s, pred_all_s, size_s,
        correct_s, correct_close_s
    ) = inference_amlp_v2(
        test_loader_s, [G, mlp], Cs, class_list_s, entropy, thr,
        per_class_correct_s, per_class_num_s,
        correct_close_s, correct_s, size_s, sourse_loader_bool=True
    )

    print('-->')
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        # roc = roc_auc_score(Y_test[:, -1], pred_open)
        # roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        # roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_acc, mean_score = select_threshold(
            pred_all, pred_open, label_all, class_list)
    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    
    # import pdb; pdb.set_trace()
    print('-->')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    output = ["step %s"%step,
              "closed perclass", list(per_class_acc),
              "acc per class %s" % (float(per_class_acc.mean())),
              "acc %s" % float(acc_all),
              "acc close all %s" % float(acc_close_all),
              "h score %s" % float(h_score),
            #   "roc %s"% float(roc),
            #   "roc ent %s"% float(roc_ent),
            #   "roc softmax %s"% float(roc_softmax),
              "best hscore %s"% float(best_acc),
              "best thr %s"% float(best_th)]
    logger.info(output)

    softmax_all_predict = torch.cat(softmax_all_predict_list).detach().cpu().numpy()
    c2_softmax_all_predict = torch.cat(c2_softmax_all_predict_list).detach().cpu().numpy()
    feature_test = torch.cat(feature_list).detach().cpu().numpy()
    pred_unk = torch.cat(pred_unk_list).detach().cpu().numpy()
    pred_all = torch.cat(predict_list).detach().cpu().numpy()
    # print(feature_test)

    softmax_all_predict_s = torch.cat(softmax_all_predict_list_s).detach().cpu().numpy()
    c2_softmax_all_predict_s = torch.cat(c2_softmax_all_predict_list_s).detach().cpu().numpy()
    feature_test_s = torch.cat(feature_list_s).detach().cpu().numpy()
    pred_unk_s = torch.cat(pred_unk_list_s).detach().cpu().numpy()
    pred_all_s = torch.cat(predict_list_s).detach().cpu().numpy()

    feature_test_t_s = np.concatenate([feature_test, feature_test_s])
    softmax_all = np.concatenate([softmax_all_predict, softmax_all_predict_s])
    c2_softmax_all = np.concatenate([c2_softmax_all_predict, c2_softmax_all_predict_s])
    num_t_sample = feature_test.shape[0]

    print('feature_test_t_s.shape', feature_test_t_s.shape)

    if feature_test_t_s.shape[0] < 50000:

        feature_test_2d_ts = TSNE().fit(feature_test_t_s)

        feature_test_2d = feature_test_2d_ts[:num_t_sample]
        plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=label_all, s=1)
        # import pdb; pdb.set_trace()
        path_uuidstr_t = str(uuid.uuid4())
        plt.colorbar()
        plt.savefig('img/t_label{}.png'.format(path_uuidstr_t))
        plt.close()
        plt.scatter(x=feature_test_2d[:,0], y=feature_test_2d[:,1], c=pred_all, s=1)
        plt.colorbar()
        plt.savefig('img/t_pred{}.png'.format(path_uuidstr_t))
        plt.close()

        feature_test_2d_s = feature_test_2d_ts[num_t_sample:]
        plt.scatter(x=feature_test_2d_s[:,0], y=feature_test_2d_s[:,1], c=label_all_s, s=1)
        # import pdb; pdb.set_trace()
        path_uuidstr_s = str(uuid.uuid4())
        plt.colorbar()
        plt.savefig('img/s_label{}.png'.format(path_uuidstr_s))
        plt.close()
        plt.scatter(x=feature_test_2d_s[:,0], y=feature_test_2d_s[:,1], c=pred_all_s, s=1)
        plt.colorbar()
        plt.savefig('img/s_pred{}.png'.format(path_uuidstr_s))
        plt.close()

        # import pdb; pdb.set_trace()
        data_path = pd.DataFrame()
        data_path['path'] = path_list + path_list_s
        data_path['pre'] = np.concatenate([pred_all, pred_all_s], axis=0)
        data_path['lab'] =  np.concatenate([label_all, label_all_s], axis=0) # label_all
        data_path['tsne_1'] = feature_test_2d_ts[:,0]
        data_path['tsne_2'] = feature_test_2d_ts[:,1]
        data_path['pred_unk'] = np.concatenate([pred_unk, pred_unk_s,], axis=0) #pred_unk
        data_path = pd.concat([
            data_path, 
            pd.DataFrame(softmax_all, columns=['soft_{}'.format(i) for i in range(softmax_all.shape[1])]),
            pd.DataFrame(c2_softmax_all, columns=['c2_soft_{}'.format(i) for i in range(c2_softmax_all.shape[1])])
            ],
            axis=1)
        
        s_or_t = np.zeros(data_path.shape[0])
        s_or_t[num_t_sample:] = 1
        data_path['s_or_t'] = s_or_t

        np.save('file_save/feature_test{}'.format(path_uuidstr_t), feature_test)
        np.save('file_save/feature_test_s{}'.format(path_uuidstr_t), feature_test_s)
        np.save('file_save/pred_all_s{}'.format(path_uuidstr_t), pred_all_s)
        np.save('file_save/pred_all{}'.format(path_uuidstr_t), pred_all)
        
        wandb.log({
            'epoch': step,
            'entropy'+str(entropy)+'_best best_th':float(best_th),
            'entropy'+str(entropy)+'_best hscore':float(best_acc),
            'entropy'+str(entropy)+'_t_label':wandb.Image('img/t_label{}.png'.format(path_uuidstr_t)),
            'entropy'+str(entropy)+'_t_pred':wandb.Image( 'img/t_pred{}.png'.format(path_uuidstr_t)),
            'entropy'+str(entropy)+'_s_label':wandb.Image('img/s_label{}.png'.format(path_uuidstr_s)),
            'entropy'+str(entropy)+'_s_pred':wandb.Image( 'img/s_pred{}.png'.format(path_uuidstr_s)),
            'feature_test':'file_save/feature_test{}'.format(path_uuidstr_t),
            'feature_test_s':'file_save/feature_test_s{}'.format(path_uuidstr_t),
            'pred_all_s':'file_save/pred_all_s{}'.format(path_uuidstr_t),
            'pred_all':'file_save/pred_all{}'.format(path_uuidstr_t),
            'entropy'+str(entropy)+'_acc_all':acc_all,
            'entropy'+str(entropy)+'_h_score':h_score,
            'entropy'+str(entropy)+'_pre_table': wandb.Table(data=data_path, columns=['path', 'pre', 'label', 'pred_unk'])
        })
    else:
        wandb.log({
            'epoch': step,
            'entropy'+str(entropy)+'_best best_th':float(best_th),
            'entropy'+str(entropy)+'_best hscore':float(best_acc),
            'entropy'+str(entropy)+'_acc_all':acc_all,
            'entropy'+str(entropy)+'_h_score':h_score,
        })

    # wandb.save('img/t_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/t_pred{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_pred{}.png'.format(path_uuidstr_s))

    return acc_all, h_score

def balanced_hscore_compute(class_correct, class_num, class_list):
    right_score = (class_correct[:len(class_list)-1].sum() * class_correct[-1])/\
        (class_correct[:len(class_list)-1].sum() + class_correct[-1])
    number_score = (class_num[:len(class_list)-1].sum() + class_num[-1])/\
        (class_num[:len(class_list)-1].sum() * class_num[-1])
    balanced_hscore = right_score * number_score
    return balanced_hscore


def test_amlp_v2_only_c2_maxcompair(step, test_loader_s, dataset_test, name, n_share, G_mlp, Cs, mlp,
         open_class = None, open=False, entropy=False, thr=0.5, argsSName=''):
    G, mlp = G_mlp
    G.eval()
    mlp.eval()

    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    print('-->')
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    
    (
        softmax_all_predict_list, c2_softmax_all_predict_list, feature_list, predict_list, path_list,
        pred_unk_list, label_all, pred_open,
        pred_ent, pred_all, size,
        correct, correct_close
    ) = inference_amlp_v2_only_c2_maxcompair(
        dataset_test, [G, mlp], Cs, class_list, entropy, thr,
        per_class_correct, per_class_num,
        correct_close, correct, size, thr_open=0.5,
    )

    print('-->')
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        # roc = roc_auc_score(Y_test[:, -1], pred_open)
        # roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        # roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_acc, mean_score = select_threshold(
            pred_all, pred_open, label_all, class_list)
    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    
    # import pdb; pdb.set_trace()
    print('-->')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    
    # b_score = balanced_hscore_compute(class_correct, class_num, class_list)
    balanced_hscore = (per_class_correct[:len(class_list)-1].sum() * per_class_correct[-1])/(per_class_correct[:len(class_list)-1].sum() + per_class_correct[-1])*\
                        (per_class_num[:len(class_list)-1].sum() + per_class_num[-1])/(per_class_num[:len(class_list)-1].sum() * per_class_num[-1])

    output = ["step %s"%step,
              "closed perclass", list(per_class_acc),
              "acc per class %s" % (float(per_class_acc.mean())),
              "acc %s" % float(acc_all),
              "acc close all %s" % float(acc_close_all),
              "h score %s" % float(h_score),
            #   "roc %s"% float(roc),
            #   "roc ent %s"% float(roc_ent),
            #   "roc softmax %s"% float(roc_softmax),
              "best hscore %s"% float(best_acc),
              "best thr %s"% float(best_th)]
    logger.info(output)

    wandb.log({
        'epoch': step,
        'entropy'+str(entropy)+'_best best_th': float(best_th),
        'entropy'+str(entropy)+'_best hscore': float(best_acc),
        'entropy'+str(entropy)+'_acc_all': acc_all,
        'entropy'+str(entropy)+'_h_score': h_score,
        'balanced_hscore': balanced_hscore,
    })

    return acc_all, h_score


def test_amlp_v2_only_c2_maxcompair_eachclass(step, test_loader_s, dataset_test, name, n_share, G_mlp, Cs, mlp,
         open_class = None, open=False, entropy=False, thr=0.5, argsSName=''):
    G, mlp = G_mlp
    G.eval()
    mlp.eval()

    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    print('-->')
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    
    (
        softmax_all_predict_list, c2_softmax_all_predict_list, feature_list, predict_list, path_list,
        pred_unk_list, label_all, pred_open,
        pred_ent, pred_all, size,
        correct, correct_close
    ) = inference_amlp_v2_only_c2_maxcompair_eachclass(
        dataset_test, [G, mlp], Cs, class_list, entropy, thr,
        per_class_correct, per_class_num,
        correct_close, correct, size, thr_open=0.5,
    )

    print('-->')
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        # roc = roc_auc_score(Y_test[:, -1], pred_open)
        # roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        # roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_acc, mean_score = select_threshold(
            pred_all, pred_open, label_all, class_list)
    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    
    # import pdb; pdb.set_trace()
    print('-->')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    output = ["step %s"%step,
              "closed perclass", list(per_class_acc),
              "acc per class %s" % (float(per_class_acc.mean())),
              "acc %s" % float(acc_all),
              "acc close all %s" % float(acc_close_all),
              "h score %s" % float(h_score),
            #   "roc %s"% float(roc),
            #   "roc ent %s"% float(roc_ent),
            #   "roc softmax %s"% float(roc_softmax),
              "best hscore %s"% float(best_acc),
              "best thr %s"% float(best_th)]
    logger.info(output)

    wandb.log({
        'epoch': step,
        'entropy'+str(entropy)+'_best best_th':float(best_th),
        'entropy'+str(entropy)+'_best hscore':float(best_acc),
        'entropy'+str(entropy)+'_acc_all':acc_all,
        'entropy'+str(entropy)+'_h_score':h_score,
    })

    return acc_all, h_score


def test_only_c2(step, source_loader, dataset_test, name, n_share, G, Cs, mlp,
         open_class = None, open=False, entropy=False, thr=0.5, argsSName=''):
    G.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    print('-->')
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    
    (
        feature_list, predict_list, path_list,
        pred_unk_list, label_all, pred_open,
        pred_ent, pred_all, size,
        correct, correct_close
    ) = inference_only_c2(
        dataset_test, G, Cs, class_list, entropy, thr,
        per_class_correct, per_class_num,
        correct_close, correct, size, thr_open=0.5,
    )

    print('-->')
    correct_s = 0
    correct_close_s = 0
    size_s = 0
    per_class_num_s = np.zeros((n_share + 1))
    per_class_correct_s = np.zeros((n_share + 1)).astype(np.float32)
    class_list_s = [i for i in range(n_share)]
    (
        feature_list_s, predict_list_s, path_list_s,
        pred_unk_list_s, label_all_s, pred_open_s,
        pred_ent_s, pred_all_s, size_s,
        correct_s, correct_close_s
    ) = inference_only_c2(
        source_loader, G, Cs, class_list_s, entropy, thr,
        per_class_correct_s, per_class_num_s,
        correct_close_s, correct_s, size_s, sourse_loader_bool=True
    )

    print('-->')
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        # roc = roc_auc_score(Y_test[:, -1], pred_open)
        # roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        # roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_acc, mean_score = select_threshold(
            pred_all, pred_open, label_all, class_list)

    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    
    # import pdb; pdb.set_trace()
    print('-->')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)

    # wandb.save('img/t_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/t_pred{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_pred{}.png'.format(path_uuidstr_s))

    wandb.log({
        'epoch': step,
        # 'entropy'+str(entropy)+'_best best_th':float(best_th),
        # 'entropy'+str(entropy)+'_best hscore':float(best_acc),
        'c2_acc_all': acc_all,
        'c2_h_score': h_score,
    })

    return acc_all, h_score


def test_half(step, source_loader, dataset_test, name, n_share, G, Cs, mlp,
         open_class = None, open=False, entropy=False, thr=0.5, argsSName=''):
    G.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    print('-->')
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    
    (
        feature_list, predict_list, path_list,
        pred_unk_list, label_all, pred_open,
        pred_ent, pred_all, size,
        correct, correct_close
    ) = inference_half(
        dataset_test, G, Cs, class_list, entropy, thr,
        per_class_correct, per_class_num,
        correct_close, correct, size, thr_open=0.5,
    )

    print('-->')
    correct_s = 0
    correct_close_s = 0
    size_s = 0
    per_class_num_s = np.zeros((n_share + 1))
    per_class_correct_s = np.zeros((n_share + 1)).astype(np.float32)
    class_list_s = [i for i in range(n_share)]
    (
        feature_list_s, predict_list_s, path_list_s,
        pred_unk_list_s, label_all_s, pred_open_s,
        pred_ent_s, pred_all_s, size_s,
        correct_s, correct_close_s
    ) = inference_half(
        source_loader, G, Cs, class_list_s, entropy, thr,
        per_class_correct_s, per_class_num_s,
        correct_close_s, correct_s, size_s, sourse_loader_bool=True
    )

    print('-->')
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        # roc = roc_auc_score(Y_test[:, -1], pred_open)
        # roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        # roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_acc, mean_score = select_threshold(
            pred_all, pred_open, label_all, class_list)

    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    
    # import pdb; pdb.set_trace()
    print('-->')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    output = ["step %s"%step,
              "closed perclass", list(per_class_acc),
              "acc per class %s" % (float(per_class_acc.mean())),
              "acc %s" % float(acc_all),
              "acc close all %s" % float(acc_close_all),
              "h score %s" % float(h_score),
            #   "roc %s"% float(roc),
            #   "roc ent %s"% float(roc_ent),
            #   "roc softmax %s"% float(roc_softmax),
              "best hscore %s"% float(best_acc),
              "best thr %s"% float(best_th)]
    logger.info(output)

    feature_test = torch.cat(feature_list).detach().cpu().numpy()
    pred_unk = torch.cat(pred_unk_list).detach().cpu().numpy()
    pred_all = torch.cat(predict_list).detach().cpu().numpy()
    # print(feature_test)

    feature_test_s = torch.cat(feature_list_s).detach().cpu().numpy()
    pred_unk_s = torch.cat(pred_unk_list_s).detach().cpu().numpy()
    pred_all_s = torch.cat(predict_list_s).detach().cpu().numpy()

    feature_test_t_s = np.concatenate([feature_test, feature_test_s])
    num_t_sample = feature_test.shape[0]



    # wandb.save('img/t_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/t_pred{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_label{}.png'.format(path_uuidstr_s))
    # wandb.save('img/s_pred{}.png'.format(path_uuidstr_s))

    return acc_all, h_score

def test_thr_open(step, source_loader, dataset_test, name, n_share, G, Cs, mlp,
         open_class = None, open=False, entropy=False, thr=0.5, argsSName=''):
    G.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    print('-->')

    h_score_dict={'step':step}

    for thr_open in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        correct = 0
        correct_close = 0
        size = 0
        
        per_class_num = np.zeros((n_share + 1))
        per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
        class_list = [i for i in range(n_share)]
        (
            feature_list, predict_list, path_list,
            pred_unk_list, label_all, pred_open,
            pred_ent, pred_all, size,
            correct, correct_close
        ) = inference(
            dataset_test, G, Cs, class_list, entropy, thr,
            per_class_correct, per_class_num,
            correct_close, correct, size, thr_open=thr_open,
        )
        per_class_acc = per_class_correct / per_class_num
        acc_all = 100. * float(correct) / float(size)
        close_count = float(per_class_num[:len(class_list) - 1].sum())
        acc_close_all = 100. *float(correct_close) / close_count
        known_acc = per_class_acc[:len(class_list)-1].mean()
        unknown = per_class_acc[-1]
        h_score = 2 * known_acc * unknown / (known_acc + unknown)
        h_score_dict['thr_{}'.format(thr_open)] = h_score
    
    wandb.log(h_score_dict)


def select_threshold(pred_all, conf_thr, label_all,
                     class_list, thr=None):
    num_class  = class_list[-1]
    best_th = 0.0
    best_f = 0
    #best_known = 0
    if thr is not None:
        pred_class = pred_all.argmax(axis=1)
        ind_unk = np.where(conf_thr > thr)[0]
        pred_class[ind_unk] = num_class
        return accuracy_score(label_all, pred_class), \
               accuracy_score(label_all, pred_class), \
               accuracy_score(label_all, pred_class)
    ran = np.linspace(0.0, 1.0, num=51)
    conf_thr = conf_thr / conf_thr.max()
    scores = []
    for th in ran:
        pred_class = pred_all.argmax(axis=1)
        ind_unk = np.where(conf_thr > th)[0]
        pred_class[ind_unk] = num_class
        score, known, unknown = h_score_compute(label_all, pred_class,
                                                class_list)
        scores.append(score)
        if score > best_f:
            best_th = th
            best_f = score
            best_known = known
            best_unknown = unknown
    mean_score = np.array(scores).mean()
    # print("best known %s best unknown %s "
    #       "best h-score %s"%(best_known, best_unknown, best_f))
    return best_th, best_f, mean_score


def h_score_compute(label_all, pred_class, class_list):
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros((len(class_list))).astype(np.float32)
    for i, t in enumerate(class_list):
        t_ind = np.where(label_all == t)
        correct_ind = np.where(pred_class[t_ind[0]] == t)
        per_class_correct[i] += float(len(correct_ind[0]))
        per_class_num[i] += float(len(t_ind[0]))
    open_class = len(class_list)
    per_class_acc = per_class_correct / per_class_num
    known_acc = per_class_acc[:open_class - 1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    return h_score, known_acc, unknown
