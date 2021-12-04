#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: cbwces <sknyqbcbw@gmail.com>
# Date  : 20211204

import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class CosFace(Module):
    """Implementation of CosFace
    """
    def __init__(self, feat_dim, num_class, margin_arc=0.35, margin_am=0.0, scale=64):
        super(CosFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_arc = margin_arc
        self.margin_am = margin_am
        self.scale = scale

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm) 
        cos_theta = cos_theta.clamp(-1, 1)
        cos_theta_m = cos_theta - margin_arc
        cos_theta_m = torch.where(cos_theta > self.min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output
