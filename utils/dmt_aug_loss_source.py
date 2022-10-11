# a pytorch based lisv2 code

import pdb
from multiprocessing import Pool

import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F
from scipy import optimize
from torch import nn
from torch.autograd import Variable
from torch.functional import split
from torch.nn.modules import loss
from typing import Any
import scipy

class MyLoss(nn.Module):
    def __init__(
        self,
        v_input,
        v_latent,
        SimilarityFunc,
        augNearRate=100000,
        sigmaP=1.0,
        sigmaQ=1.0,
    ):
        super(MyLoss, self).__init__()

        self.v_input = v_input
        self.v_latent = v_latent
        self.gamma_input = self._CalGamma(v_input)
        self.ITEM_loss = self._TwowaydivergenceLoss
        self._Similarity = SimilarityFunc
        self.augNearRate = augNearRate
        self.sigmaP = sigmaP
        self.sigmaQ = sigmaQ
    
    def forward(self, input_data, input_data_aug, latent_data, latent_data_aug, rho, sigma):
        """
        after resnet50:
        input, input_data_aug: [128, 2048]
        after MLP:
        latent_data, latent_data_aug: [128, 256]
        rho:0, sigma:1
        """
        metaDistance = self._DistanceSquared(input_data, input_data)
        metaDistance_ = metaDistance.clone().detach()
        metaDistance_[torch.eye(metaDistance_.shape[0])==1.0] = metaDistance_.max()+1
        nndistance, _ = torch.min(metaDistance_, dim=0)
        nndistance = nndistance / self.augNearRate
        downDistance = metaDistance + nndistance.reshape(-1, 1)
        rightDistance = metaDistance + nndistance.reshape(1, -1)
        rightdownDistance = metaDistance + nndistance.reshape(1, -1) + nndistance.reshape(-1, 1)

        disInput = torch.cat(
            [
                torch.cat([metaDistance, downDistance]),
                torch.cat([rightDistance, rightdownDistance]),
            ],
            dim=1
        )
        latent = torch.cat([latent_data, latent_data_aug])
        distlatent = self._DistanceSquared(latent, latent)

        loss = self._TwowaydivergenceLoss(
                    P=self._Similarity(dist=disInput,rho=0,sigma_array=self.sigmaP,gamma=self._CalGamma(self.v_input),v=self.v_input),
                    Q=self._Similarity(dist=distlatent,rho=0,sigma_array=self.sigmaQ,gamma=self._CalGamma(self.v_latent),v=self.v_latent,)
                )
        # import pdb; pdb.set_trace()
        # loss shape: [256,256]
        loss[torch.eye(loss.shape[0])==1]=0

        self.loss_save = loss
        return loss.mean()



    def f_mask(
        self, input_data, input_data_aug,
        latent_data, 
        latent_data_aug,
        mask,
        rho,
        sigma,
        ):
        """
        after resnet50:
        input, input_data_aug: [128, 2048]
        after MLP:
        latent_data, latent_data_aug: [128, 256]
        rho:0, sigma:1
        """
        metaDistance = self._DistanceSquared(input_data, input_data)
        metaDistance_ = metaDistance.clone().detach()
        metaDistance_[torch.eye(metaDistance_.shape[0])==1.0] = metaDistance_.max()+1
        nndistance, _ = torch.min(metaDistance_, dim=0)
        nndistance = nndistance / self.augNearRate
        downDistance = metaDistance + nndistance.reshape(-1, 1)
        rightDistance = metaDistance + nndistance.reshape(1, -1)
        rightdownDistance = metaDistance + nndistance.reshape(1, -1) + nndistance.reshape(-1, 1)

        disInput = torch.cat(
            [
                torch.cat([metaDistance, downDistance]),
                torch.cat([rightDistance, rightdownDistance]),
            ],
            dim=1
        )
        latent = torch.cat([latent_data, latent_data_aug])
        distlatent = self._DistanceSquared(latent, latent)

        loss = self._TwowaydivergenceLoss(
                    P=self._Similarity(
                        dist=disInput,
                        rho=0,
                        sigma_array=self.sigmaP,
                        gamma=self._CalGamma(self.v_input),
                        v=self.v_input
                    ),
                    Q=self._Similarity(
                        dist=distlatent,
                        rho=0,
                        sigma_array=self.sigmaQ,
                        gamma=self._CalGamma(self.v_latent),
                        v=self.v_latent,
                    )
                )
        # loss shape: [256,256]
        return loss.mean()


    def _TwowaydivergenceLoss(self, P, Q):

        EPS = 1e-12
        losssum1 = (P * torch.log(Q + EPS))
        losssum2 = ((1-P) * torch.log(1-Q + EPS))
        losssum = -1*(losssum1 + losssum2)

        return losssum

    def _L2Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=2)/P.shape[0]
        return losssum
    
    def _L3Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=3)/P.shape[0]
        return losssum

    
    def _DistanceSquared(self, x, y):

        return torch.pow(torch.cdist(x, y, p=2),2)

    def _CalGamma(self, v):
        
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out



class MyLoss_intro(nn.Module):
    def __init__(
        self,
        v_input,
        v_latent,
        SimilarityFunc,
        augNearRate=100000,
        sigmaP=1.0,
        sigmaQ=1.0,
    ):
        super(MyLoss_intro, self).__init__()

        self.v_input = v_input
        self.v_latent = v_latent
        self.gamma_input = self._CalGamma(v_input)
        self.ITEM_loss = self._TwowaydivergenceLoss
        self._Similarity = SimilarityFunc
        self.augNearRate = augNearRate
        self.sigmaP = sigmaP
        self.sigmaQ = sigmaQ
    
    def forward(self, input_data, input_data_aug, latent_data, latent_data_aug, rho, sigma):
        """
        after resnet50:
        input, input_data_aug: [128, 2048]
        after MLP:
        latent_data, latent_data_aug: [128, 256]
        rho:0, sigma:1
        """
        metaDistance = self._DistanceSquared(input_data, input_data)
        metaDistance_ = metaDistance.clone().detach()
        metaDistance_[torch.eye(metaDistance_.shape[0])==1.0] = metaDistance_.max()+1
        nndistance, _ = torch.min(metaDistance_, dim=0)
        nndistance = nndistance / self.augNearRate
        downDistance = metaDistance + nndistance.reshape(-1, 1)
        rightDistance = metaDistance + nndistance.reshape(1, -1)
        rightdownDistance = metaDistance + nndistance.reshape(1, -1) + nndistance.reshape(-1, 1)

        disInput = torch.cat(
            [
                torch.cat([metaDistance, downDistance]),
                torch.cat([rightDistance, rightdownDistance]),
            ],
            dim=1
        )
        latent = torch.cat([latent_data, latent_data_aug])
        distlatent = self._DistanceSquared(latent, latent)

        loss = self._TwowaydivergenceLoss(
                    P=self._Similarity(dist=disInput,rho=0,sigma_array=self.sigmaP,gamma=self._CalGamma(self.v_input),v=self.v_input),
                    Q=self._Similarity(dist=distlatent,rho=0,sigma_array=self.sigmaQ,gamma=self._CalGamma(self.v_latent),v=self.v_latent,)
                )
        # import pdb; pdb.set_trace()
        # loss shape: [256,256]
        loss[torch.eye(loss.shape[0])==1]=0

        import numpy as np
        import wandb

        intro_label = np.loadtxt('lab_intro.csv')

        batch_size = loss.shape[0]//2
        loss_part = loss[batch_size:][:,:batch_size] 
        loss_line_a = loss_part[torch.eye(batch_size)==1][torch.tensor(intro_label)==1].sum()
        loss_line_b = loss_part[torch.eye(batch_size)==1][torch.tensor(intro_label)==0].sum()

        print('rate_posi_neg', loss_line_a/loss_line_b)

        wandb.log({
            'rate_posi_neg': loss_line_a/loss_line_b
        })

        import pdb; pdb.set_trace()
        self.loss_save = loss
        return loss.mean()



    def f_mask(
        self, input_data, input_data_aug,
        latent_data, 
        latent_data_aug,
        mask,
        rho,
        sigma,
        ):
        """
        after resnet50:
        input, input_data_aug: [128, 2048]
        after MLP:
        latent_data, latent_data_aug: [128, 256]
        rho:0, sigma:1
        """
        metaDistance = self._DistanceSquared(input_data, input_data)
        metaDistance_ = metaDistance.clone().detach()
        metaDistance_[torch.eye(metaDistance_.shape[0])==1.0] = metaDistance_.max()+1
        nndistance, _ = torch.min(metaDistance_, dim=0)
        nndistance = nndistance / self.augNearRate
        downDistance = metaDistance + nndistance.reshape(-1, 1)
        rightDistance = metaDistance + nndistance.reshape(1, -1)
        rightdownDistance = metaDistance + nndistance.reshape(1, -1) + nndistance.reshape(-1, 1)

        disInput = torch.cat(
            [
                torch.cat([metaDistance, downDistance]),
                torch.cat([rightDistance, rightdownDistance]),
            ],
            dim=1
        )
        latent = torch.cat([latent_data, latent_data_aug])
        distlatent = self._DistanceSquared(latent, latent)

        loss = self._TwowaydivergenceLoss(
                    P=self._Similarity(
                        dist=disInput,
                        rho=0,
                        sigma_array=self.sigmaP,
                        gamma=self._CalGamma(self.v_input),
                        v=self.v_input
                    ),
                    Q=self._Similarity(
                        dist=distlatent,
                        rho=0,
                        sigma_array=self.sigmaQ,
                        gamma=self._CalGamma(self.v_latent),
                        v=self.v_latent,
                    )
                )
        # loss shape: [256,256]
        return loss.mean()


    def _TwowaydivergenceLoss(self, P, Q):

        EPS = 1e-12
        losssum1 = (P * torch.log(Q + EPS))
        losssum2 = ((1-P) * torch.log(1-Q + EPS))
        losssum = -1*(losssum1 + losssum2)

        return losssum

    def _L2Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=2)/P.shape[0]
        return losssum
    
    def _L3Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=3)/P.shape[0]
        return losssum

    
    def _DistanceSquared(self, x, y):

        return torch.pow(torch.cdist(x, y, p=2),2)

    def _CalGamma(self, v):
        
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out

def Similarity(dist, rho, sigma_array, gamma, v=100):

    dist_rho = (dist - rho) / sigma_array
    dist_rho[dist_rho < 0] = 0

    Pij = gamma*gamma * torch.pow(
            (1 + dist_rho / v),
            -1 * (v + 1)
            ) * 2 * 3.14

    P = Pij + Pij.t() - torch.mul(Pij, Pij.t())

    return P