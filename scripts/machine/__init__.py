from .BasicMachine import BasicMachine
from .S2AM import S2AM
from .SSPP import SSPP
from .SSPPv2 import SSPPv2
from .SSPPv3 import SSPP_LUT
from .SSPP_re import SSPP_re
from .SSPP_re_4 import SSPP_re_4

def basic(**kwargs):
    return BasicMachine(**kwargs)

def sspp(**kwargs):
    return SSPP(**kwargs)


def ssppv2(**kwargs):
    return SSPPv2(**kwargs)


def sspp_re(**kwargs):
    return SSPP_re(**kwargs)

def sspp_re_re(**kwargs):
    return SSPP_re_4(**kwargs)


def ssppv3(**kwargs):
    return SSPP_LUT(**kwargs)

