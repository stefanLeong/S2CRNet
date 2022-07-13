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
from scripts.models.vmu import UnetSplit
from scripts.models.split_s2am import SplitwithS2AM
from scripts.models.split_capp import SplitwithCAPP, SplitwithCAPPV1
from scripts.models.split_cappV2 import SplitwithCAPPV2
from scripts.models.split_cappV3 import SplitwithCAPPV3
from scripts.models.split_Unet import SplitwithUnet
from scripts.models.split_UnetV2 import SplitwithUNetV2
from scripts.models.splitwithUnetV3 import SplitwithUNetV3
from scripts.models.split_single import SplitwithSingle

from scripts.models.split_rn import XRN
from scripts.models.rn import RN_B

# our method
def split(**kwargs):
    model = UnetSplit(shared_depth=3, blocks=7, long_skip=True)
    model.apply(weights_init_kaiming)
    return model


def split_single(**kwargs):
    model = SplitwithSingle(shared_depth=3, blocks=3, depth=5, long_skip=True)
    model.apply(weights_init_kaiming)
    return model


def sn(**kwargs):
    model = SimpleNet()
    model.apply(weights_init_kaiming)
    return model

def hdr(**kwargs):
    model = HDRPointwiseNN()
    return model

def sqq(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNet(sq)
    return model

def basic(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = BasicNet(sq)
    return model



def fore(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = ForeNet(sq)
    return model

def sqq64(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNet(sq, 64)
    return model

def sqq128(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNet(sq, 128)
    return model

def fusion(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNetFusion(sq)
    return model


def sqq3(**kwargs):
    model = CreateNet3stage()
    return model

def adain(**kwargs):
    sq = squeezenet1_1(pretrained=True)
    model = CreateNetADAIN(sq)
    return model

def split_rn(**kwargs):
    model = XRN(shared_depth=3, blocks=3, depth=5, long_skip=True, norm=RN_B)
    model.apply(weights_init_kaiming)
    return model


def split_with_s2am(**kwargs):
    model = SplitwithS2AM(shared_depth=3, blocks=7, long_skip=True)
    model.apply(weights_init_kaiming)
    return model


def split_with_capp(**kwargs):
    model = SplitwithCAPPV1(shared_depth=3, blocks=3, depth=7, long_skip=True)
    model.apply(weights_init_kaiming)
    return model


def split_with_unet(**kwargs):
    model = SplitwithUnet(shared_depth=3, blocks=2, depth=5, long_skip=True)
    model.apply(weights_init_kaiming)
    return model


def split_with_unetv2(**kwargs):
    model = SplitwithUNetV2(shared_depth=2, blocks=2, depth=5, long_skip=True)
    model.apply(weights_init_kaiming)
    return model


def split_with_unetv3(**kwargs):
    model = SplitwithUNetV3(shared_depth=3, blocks=0, depth=6, long_skip=True)
    model.apply(weights_init_kaiming)
    return model


def split_patch(**kwargs):
    model = SplitwithCAPPV2(shared_depth=3, blocks=7, depth=5, long_skip=True)
    model.apply(weights_init_kaiming)
    return model


def split_patch_mask(**kwargs):
    model = SplitwithCAPPV3(shared_depth=3, blocks=3, depth=5, long_skip=True)
    model.apply(weights_init_kaiming)
    return model


def s2am(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=True,attention_model=RASC,basicblock=MinimalUnetV2)
    model.apply(weights_init_kaiming)
    return model

def unet(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=False,attention_model=None,basicblock=MinimalUnetV2)
    model.apply(weights_init_kaiming)
    return model