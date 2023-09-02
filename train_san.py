from __future__ import print_function
from unicodedata import name
import yaml
import easydict
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
# from apex import amp, optimizers
from utils.utils import log_set, save_model
from utils.loss import ova_loss, open_entropy, ova_loss_amlp
from utils.lr_schedule import inv_lr_scheduler
from utils.defaults import get_dataloaders, get_models, get_models_amlp, get_dataloaders_mlp, get_models_amlp_oda

import utils.dmt_aug_loss_source as dmtloss
import utils.dmt_aug_loss_source_mask as dmtloss_mask···
import pytorch_lightning as pl
import plotly.graph_objects as go

from eval import test, test_amlp_v2_only_c2_maxcompair_eachclass, test_thr_open, test_only_c2, test_amlp, test_amlp_v2, test_amlp_v2_only_c2
from eval import test_amlp_v2_only_c2_maxcompair
import argparse
import wandb
import plotly.express as px
# from neighbor import NF, LossPatEmb

parser = argparse.ArgumentParser(description='Pytorch SAN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml',
                    help='./configs/office-train-config_OPDA.yaml')

parser.add_argument('--source_data', type=str,
                    default='./txt/source_amazon_opda_zzl.txt',
                    help='path to source list')
parser.add_argument('--target_data', type=str,
                    default='./txt/target_dslr_opda_zzl.txt',
                    help='path to target list')
parser.add_argument('--log-interval', type=int,
                    default=500,
                    help='how many batches before logging training status')
parser.add_argument('--exp_name', type=str,
                    default='office',
                    help='/path/to/config/file')
parser.add_argument('--network', type=str,
                    default='resnet50',
                    help='network name')
parser.add_argument("--gpu_devices", type=int, nargs='+',
                    default=None, help="")
parser.add_argument("--no_adapt",
                    default=False, action='store_true')
parser.add_argument("--save_model",default=False, action='store_true')
parser.add_argument("--v_input",type=float, default=100)
parser.add_argument("--v_latent",type=float, default=10)
parser.add_argument("--augNearRate",type=float, default=10000)
parser.add_argument("--sigmaP",type=float, default=10)
parser.add_argument("--sigmaQ",type=float, default=1)
parser.add_argument("--lr",type=float, default=0.01)
parser.add_argument("--save_path", type=str, default="record/ova_model", help='/path/to/save/model')
parser.add_argument('--multi', type=float, default=0.2, help='weight factor for adaptation')
parser.add_argument("--gamma", type=float, default=0.1, help="")
parser.add_argument("--alpha", type=float, default=1, help="")
parser.add_argument("--sour_aug", type=int, default=0, help="")
parser.add_argument("--min_step", type=int, default=10000, help="")
parser.add_argument("--test_interval", type=int, default=500, help="")
parser.add_argument("--optim_tool", type=str, default='sgd', choices=['sgd', 'adamw'])
parser.add_argument('--ent_open_scale', type=float, default=0.1, help='weight factor for adaptation')
parser.add_argument('--new_loss', type=float, default=0, help='weight factor for adaptation')
parser.add_argument('--batch_size', type=int, default=36)
parser.add_argument('--data_aug_crop', type=float, default=0.8)
parser.add_argument('--aug_type', type=int, default=0)
parser.add_argument('--beta', type=float, default=1.5)
parser.add_argument('--scheduler_gamma', type=float, default=10)
parser.add_argument('--mlp_weight_decay', type=float, default=0.0002)

parser.add_argument('--sgd_momentum', type=float, default=0.9)
parser.add_argument('--top_k', type=int, default=20)

parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()


pl.utilities.seed.seed_everything(1)

args.S = args.source_data.split('_')[1]
args.T = args.target_data.split('_')[1]

wandb.init(
    name=args.source_data+'_'+args.target_data,
    project="OVANET_DMT",
    entity="zangzelin_hotmail",
    config=args,
    )

print(args.S, args.T)

if (args.S != args.T or args.T == 'visda'):
    config_file = args.config
    conf = yaml.full_load(open(config_file))
    save_config = yaml.full_load(open(config_file))
    conf = easydict.EasyDict(conf)
    conf.test.test_interval = args.test_interval
    conf.test.sgd_momentum = args.sgd_momentum
    # conf.test.test_interval = 50
    # gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    args.cuda = torch.cuda.is_available()
    conf.train.lr = args.lr
    conf.train.min_step = args.min_step

    source_data = args.source_data
    target_data = args.target_data
    evaluation_data = args.target_data
    network = args.network
    use_gpu = torch.cuda.is_available()
    n_share = conf.data.dataset.n_share
    n_source_private = conf.data.dataset.n_source_private
    n_total = conf.data.dataset.n_total
    open = n_total - n_share - n_source_private > 0
    num_class = n_share + n_source_private
    script_name = os.path.basename(__file__)

    inputs = vars(args)
    conf.data.dataloader.batch_size = args.batch_size
    print('conf.data.dataloader.batch_size', conf.data.dataloader.batch_size)
    inputs["evaluation_data"] = evaluation_data
    inputs["conf"] = conf
    inputs["script_name"] = script_name
    inputs["num_class"] = num_class
    inputs["config_file"] = config_file
    # inputs["data_aug_crop"] = config_file

    source_loader, target_loader, \
    test_loader, target_folder, test_loader_s = get_dataloaders_mlp(inputs)

    logname = log_set(inputs)

    G, C1, C2, mlp, opt_g, opt_c, \
        opt_mlp, param_lr_g, param_lr_c, param_lr_m = get_models_amlp_oda(inputs)

    ndata = target_folder.__len__()
    print('--------------------')
    dmt_loss = dmtloss.MyLoss(
        v_input=args.v_input,
        v_latent=args.v_latent,
        SimilarityFunc=dmtloss.Similarity,
        augNearRate=args.augNearRate,
        sigmaP=args.sigmaP,
        sigmaQ=args.sigmaQ
        ).cuda()

    def train():
        mask = None
        criterion = nn.CrossEntropyLoss().cuda()
        print('train start!')
        data_iter_s = iter(source_loader)
        data_iter_t = iter(target_loader)
        len_train_source = len(source_loader)
        len_train_target = len(target_loader)

        # import pdb; pdb.set_trace()
        h_score_epoch = 0
        for step in range(conf.train.min_step + 1):
            G.train()
            C1.train()
            C2.train()
            mlp.train()

            if step % len_train_target == 0:
                data_iter_t = iter(target_loader)
            if step % len_train_source == 0:
                data_iter_s = iter(source_loader)
            data_t = next(data_iter_t)
            data_s = next(data_iter_s)
            
            inv_lr_scheduler(param_lr_g, opt_g, step,
                            init_lr=conf.train.lr,
                            max_iter=conf.train.min_step,
                            gamma=args.scheduler_gamma)
            inv_lr_scheduler(param_lr_c, opt_c, step,
                            init_lr=conf.train.lr,
                            max_iter=conf.train.min_step,
                            gamma=args.scheduler_gamma)
            inv_lr_scheduler(param_lr_m, opt_mlp, step,
                            init_lr=conf.train.lr,
                            max_iter=conf.train.min_step,
                            gamma=args.scheduler_gamma)

            img_s_ls, label_s, id_s = data_s
            img_t_ls, label_t, id_t = data_t

            # img_s0, img_s1, img_s2 = Variable(img_s_ls[0].cuda()), \
            #     Variable(img_s_ls[1].cuda()), Variable(img_s_ls[2].cuda())
            
            img_s2 = Variable(img_s_ls.cuda())
            img_t0, img_t1 = Variable(img_t_ls[0].cuda()), \
                Variable(img_t_ls[1].cuda())

            label_s = Variable(label_s.cuda())

            opt_g.zero_grad()
            opt_c.zero_grad()
            opt_mlp.zero_grad()
            C2.module.weight_norm()

            # feat_s0 = G(img_s0)
            # feat_s1 = G(img_s1)
            feat_s2 = G(img_s2)
            z_s2 = mlp(feat_s2)             # (36, 256)

            fea0 = G(img_t0)             # (36, 2048)
            fea1 = G(img_t1)
            # feat_t2 = G(img_t2)
            z0 = mlp(fea0)             # (36, 256)
            z1 = mlp(fea1)             # (36, 256)
            
            # feat_s = torch.cat([feat_s0, feat_s1, feat_s2])
            # feat_s = torch.cat([z_s2])
            
            # out_s = C1(z_s2)
            out_s = C2(z_s2).reshape(z_s2.shape[0], 2, -1)[:,1,:]
            out_open = C2(z_s2)
            # label_s = torch.cat([label_s, label_s, label_s])
            # label_s = torch.cat([label_s])

            # l_cls
            loss_s = criterion(out_s, label_s)
            # loss_s = torch.tensor(0)

            # l_open
            # import pdb; pdb.set_trace()
            # print('top_k_number', top_k_number)
            if args.top_k < out_open.shape[1]//2:
                top_k_number = min(args.top_k , out_open.shape[1]//2)
                out_open_top_k = torch.zeros((out_open.shape[0], 2, top_k_number), device=out_open.device)
                out_open_withlabel = out_open.clone().detach().view(out_s.size(0), 2, -1)[:,1,:]
                label_mask = F.one_hot(label_s, num_classes=out_open.shape[1]//2)
                out_open_withlabel[label_mask==1] = out_open_withlabel[label_mask==1] + 1000
                top_k_mask = out_open_withlabel.sort()[1].sort()[1] >= (out_open.shape[1]//2 - top_k_number)
                out_open_top_k[:,0,:] = out_open.view(out_s.size(0), 2, -1)[:,0,:][top_k_mask].reshape((-1, top_k_number))
                out_open_top_k[:,1,:] = out_open.view(out_s.size(0), 2, -1)[:,1,:][top_k_mask].reshape((-1, top_k_number))
                label_s_top_k =  F.one_hot(label_s, num_classes=out_open.shape[1]//2)[top_k_mask].reshape((-1, top_k_number)).max(dim=1)[1]
            else:
                out_open_top_k = out_open
                label_s_top_k = label_s
            
            out_open_top_k = F.softmax(out_open_top_k.reshape(out_open_top_k.shape[0], -1), 1)
            out_open_top_k = out_open_top_k.view(out_s.size(0), 2, -1)
            open_loss_pos, open_loss_neg, loss_connection_first = ova_loss_amlp(
                out_open_top_k, label_s_top_k, lambada=args.beta
                )
            loss_open = (open_loss_pos + open_loss_neg + loss_connection_first) / 3

            if args.sour_aug == 0 or args.sour_aug == 1:
                l_dmt = dmt_loss(
                    input_data=fea0,
                    input_data_aug=fea1,
                    latent_data=z0,
                    latent_data_aug=z1,
                    rho=0,
                    sigma=1
                    )

            all = loss_s + args.alpha * loss_open + args.gamma * l_dmt

            log_string = 'Train {}/{} \t ' \
                        'Loss Source: {:.4f} ' \
                        'Loss Open: {:.4f} ' \
                        'Loss l_dmt: {:.4f} ' \
                        'Loss Open Source Positive: {:.4f} ' \
                        'Loss Open Source Negative: {:.4f} '
            log_values = [
                step, conf.train.min_step,
                loss_s.item(), args.alpha * loss_open.item(), args.gamma * l_dmt.item(),
                open_loss_pos.item(), open_loss_neg.item()
                ]
            
            if not args.no_adapt:
                out_open_t = C2(z0)
                out_open_t = out_open_t.view(fea0.size(0), 2, -1)
                ent_open = open_entropy(out_open_t)
                all += args.ent_open_scale * ent_open
                log_values.append(args.ent_open_scale*ent_open.item())
                log_string += "Loss Open Target: {:.6f}"
            # with amp.scale_loss(all, [opt_g, opt_c, opt_mlp]) as scaled_loss:
            #     scaled_loss.backward()
            all.backward()
            
            opt_g.step()
            opt_c.step()
            opt_mlp.step()
            opt_g.zero_grad()
            opt_c.zero_grad()
            opt_mlp.zero_grad()

            if step % conf.train.log_interval == 0:
                print(log_string.format(*log_values))
                
                # dmt_numpy = dmt_loss.loss_save.cpu().detach().numpy()
                # fig_hist = go.Figure(data=[go.Histogram(x=dmt_numpy.reshape((-1)))])
                # print(dmt_numpy.shape)
                # fig_heatmap = px.imshow(dmt_numpy)

            if step > 0 and step % conf.test.test_interval == 0:
                acc_o_c2, h_score_c2 = test_amlp_v2_only_c2_maxcompair(
                    step, test_loader_s, test_loader, logname, n_share, [G, mlp],
                    [C1, C2], mlp, open=open, argsSName=args.S
                    )
                # acc_o_c2_eachclass, h_score_c2_eachclass = test_amlp_v2_only_c2_maxcompair_eachclass(
                #     step, test_loader_s, test_loader, logname, n_share, [G, mlp],
                #     [C1, C2], mlp, open=open, argsSName=args.S
                #     )
                h_score_epoch = max(h_score_epoch, h_score_c2)

                # print("acc all %s h_score %s " % (acc_o, h_score))
                print("c2 acc all %s c2 h_score %s " % (acc_o_c2, h_score_c2))
                # print("c2 acc all eachclass %s c2 h_score eachclass %s " % (acc_o_c2_eachclass, h_score_c2_eachclass))
                wandb.log({
                    'step': step,
                    "epoch": step,
                    "loss_class": loss_s.item(),
                    "loss_dmt": args.gamma * l_dmt.item(),
                    "loss_open": args.alpha * loss_open.item(),
                    "open_loss_pos": args.alpha *open_loss_pos/3,
                    "open_loss_neg": args.alpha *open_loss_neg/3,
                    "loss_connection_first": args.alpha *loss_connection_first/3,
                    # "dmt_hist": fig_hist,
                    # "loss_dmt_heatmap": fig_heatmap,
                    "acc all": acc_o_c2,
                    "h_score_epoch": h_score_epoch,
                    "h_score": h_score_c2
                    })

                G.train()
                C1.train()
                C2.train()
                mlp.train()
                if args.save_model:
                    save_path = "%s_%s.pth"%(args.save_path, step)
                    save_model(G, C1, C2, save_path)

    train()
    wandb.finish()
