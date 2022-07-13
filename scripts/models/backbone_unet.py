import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math
from scripts.utils.model_init import *
from scripts.models.rasc import *
from scripts.models.unet import *
from scripts.models.unet_LUT import CreateNetFusion_LUT
from scripts.models.vmu import UnetSplit
from scripts.models.ablation_unet import CreateNetFusionwoMask

from scripts.models.rn import RN_B
from scripts.models.fusion_alex import CreateNetFusion_alex, alexNet
from scripts.models.fusion_res import CreateNetFusion_res, resnet18
from scripts.models.fusion_mob import CreateNetFusion_mob, mobilenet
from scripts.models.fusion_vgg import CreateNetFusion_vgg, vgg16
from scripts.models.unet_multistack import CreateNetFusion_re
from scripts.models.unet_multistatck_4 import CreateNetFusion_re4

# our method


def fusionv3(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNetFusionV3(sq)
    return model

def fusionv3s(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNetFusionV3(sq, stack=True)
    return model

def fusion_re(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNetFusion_re(sq, stack=True)
    return model

def fusion_re_re(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNetFusion_re4(sq, stack=True)
    return model

def fusionv3s_alex(**kwargs):
    backbone = alexNet(pretrained=True)
    model = CreateNetFusion_alex(backbone, stack=True)
    return model

def fusionv3s_res(**kwargs):
    backbone = resnet18(pretrained=True)
    model = CreateNetFusion_res(backbone, stack=True)
    return model

def fusionv3s_mob(**kwargs):
    backbone = mobilenet(pretrained=True)
    model = CreateNetFusion_mob(backbone, stack=True)
    return model

def fusionv3s_vgg(**kwargs):
    backbone = vgg16(pretrained=True)
    model = CreateNetFusion_vgg(backbone, stack=True)
    return model

def fusion_lut(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNetFusion_LUT(sq, stack=True)
    return model

def fusionv3s_nomask(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNetFusionwoMask(sq, stack=True)
    return model

def fusionv3sm(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNetFusionV3(sq,stack=True,cr1=MatrixRender)
    return model

def fusion64(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNetFusion(sq, 64)
    return model


def s2am(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=True,attention_model=RASC,basicblock=MinimalUnetV2)
    model.apply(weights_init_kaiming)
    return model

def unet(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=False,attention_model=None,basicblock=MinimalUnetV2)
    model.apply(weights_init_kaiming)
    return model
