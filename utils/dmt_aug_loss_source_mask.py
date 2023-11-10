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

class MyLossMask(nn.Module):
    def __init__(
        self,
        v_input,
        v_latent,
        SimilarityFunc,
        augNearRate=100000,
        sigmaP=1.0,
        sigmaQ=1.0,
    ):
        super(MyLossMask, self).__init__()

        self.v_input = v_input
        self.v_latent = v_latent
        self.gamma_input = self._CalGamma(v_input)
        self.ITEM_loss = self._TwowaydivergenceLoss
        self._Similarity = SimilarityFunc
        self.augNearRate = augNearRate
        self.sigmaP = sigmaP
        self.sigmaQ = sigmaQ
    
    def forward(self, 
        data_t,
        data_s,
        latent_data_t,
        latent_data_s,
        # data_t_id,
        mask,
        # data_t=feat_t0,
        # data_s=feat_s0,
        # latent_data_t=z_t0,
        # latent_data_s=z_s0,
        # mask=torch.tensor(mask).cuda(),
        v_latent=10,
        ):
        """
        after resnet50:
        input, input_data_aug: [128, 2048]
        after MLP:
        latent_data, latent_data_aug: [128, 256]
        rho:0, sigma:1
        """
        metaDistance = self._DistanceSquared(data_s, data_t)
        metaDistance[mask] = metaDistance.mean()/100
        metaDistance_ = metaDistance.clone().detach()
        # metaDistance_[torch.eye(metaDistance_.shape[0])==1.0] = metaDistance_.max()+1
        # nndistance, _ = torch.min(metaDistance_, dim=0)
        # nndistance = nndistance / self.augNearRate
        # downDistance = metaDistance #+ nndistance.reshape(-1, 1)
        # rightDistance = metaDistance # + nndistance.reshape(1, -1)
        # rightdownDistance = metaDistance # + nndistance.reshape(1, -1) + nndistance.reshape(-1, 1)
        disInput = metaDistance_
        # disInput = torch.cat(
        #     [
        #         torch.cat([metaDistance, downDistance]),
        #         torch.cat([rightDistance, rightdownDistance]),
        #     ],
        #     dim=1
        # )
        # latent = torch.cat([latent_data_t, latent_data_s])
        distlatent = self._DistanceSquared(latent_data_t, latent_data_s)

        loss = self._TwowaydivergenceLoss(
                    P=self._Similarity(dist=disInput,rho=0,sigma_array=self.sigmaP,gamma=self._CalGamma(self.v_input),v=self.v_input),
                    Q=self._Similarity(dist=distlatent,rho=0,sigma_array=self.sigmaQ,gamma=self._CalGamma(self.v_latent),v=self.v_latent,)
                )
        # import pdb; pdb.set_trace()
        # loss shape: [256,256]
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

    # P = Pij + Pij.t() - torch.mul(Pij, Pij.t())

    return Pij