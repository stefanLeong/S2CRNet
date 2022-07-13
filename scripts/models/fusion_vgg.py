import torch
import torch.nn as nn
from torch.nn import init
import functools
from scripts.models.blocks import *
from scripts.models.rasc import *
import torch.utils.model_zoo as model_zoo
import trilinear

import torch.nn.functional as F
import math
import re
import random
import torchvision
from torchvision.utils import save_image as si

from torch.utils.model_zoo import load_url

from typing import Type, Any, Callable, Union, List, Optional
from typing import Union, List, Dict, Any, cast

from torch import Tensor

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'alexnet': "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    'mobilenet_v2': "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
    'mobilenet_v3_large': "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    'mobilenet_v3_small': "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
}


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13", "B", False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)





class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False, ins_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x




def CF(img, param, pieces):
    # bs x 3 x 1 x 1 x 64
    # print(param.shape)
    #
    color_curve_sum = torch.sum(param, 4) + 1e-30
    total_image = img * 0

    for i in range(pieces):
        total_image += torch.clamp(img - 1.0 * i / pieces, 0, 1.0 / pieces) * param[:, :, :, :, i]
    total_image *= pieces / color_curve_sum

    # import pdb;
    # pdb.set_trace()
    # total_image = img
    # for i in range(pieces):
    #     print(param[:, :, :, :, i].shape)
    #     total_image = total_image + F.tanh(param[:, :, :, :, i]) * (torch.pow(total_image, 2) - total_image)

    return total_image


def CF_deepCurve(img, param, pieces):
    total_image = img

    total_image = total_image + (param) * (torch.pow(total_image, 2) - total_image)

    return total_image


def CF_LUT(img, param, pieces, LUTS):
    gen_A0 = LUTS[0](F.tanh(img))
    gen_A1 = LUTS[1](F.tanh(img))
    gen_A2 = LUTS[2](F.tanh(img))
    combine_A = img.new(img.size())
    # print(gen_A0)
    # import pdb;
    # pdb.set_trace()
    for b in range(img.size(0)):
        # print(param)
        combine_A[b, :, :, :] = param[b, 0] * gen_A0[b, :, :, :] + param[b, 1] * gen_A1[b, :, :, :] + param[
            b, 2] * gen_A2[b, :, :, :]
    # import pdb;
    # pdb.set_trace()
    return combine_A * img


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)





def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Projection(nn.Module):
    def __init__(self, ic, plane, final_relu=False):
        super(Projection, self).__init__()
        self.fc1 = nn.Linear(ic, plane)  # bs x 3 x 64 #
        # self.fc2 = nn.Linear(plane, plane)  # bs x 3 x 64 #
        self.final_relu = final_relu

    def forward(self, f):

        # x = self.fc2(F.relu(self.fc1(f)))
        x = self.fc1(f)
        if self.final_relu:
            return F.relu(x)
        else:
            return x


class MatrixRender(nn.Module):
    def __init__(self, plane, final_relu=False, p=Projection):
        super(MatrixRender, self).__init__()
        self.plane = plane
        self.fc_f_1 = p(512, 9, True)  # bs x 3 x 64 #
        self.fc_b_1 = p(512, 9, True)  # bs x 3 x 64 #

    def forward(self, x, f_feature, b_feature):
        bs, c, h, w = x.size()

        param = self.fc_b_1(b_feature) + self.fc_f_1(f_feature)

        return torch.bmm(param.view(bs, c, -1), x.view(bs, c, -1)).view(bs, c, h, w)  # bs c 3


class CurveRender(nn.Module):
    def __init__(self, plane=64, final_relu=False, p=Projection):
        super(CurveRender, self).__init__()
        self.plane = plane
        self.fc_f_1 = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        self.fc_b_1 = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        self.fc_label = nn.Linear(5, 3 * plane * 512, bias=False)

    def forward(self, x, f_feature, b_feature, label=None):
        # import pdb;
        # pdb.set_trace()
        if label is not None:
            label_feature = self.fc_label(label)  # [b, 3 * 64 * 512]
            f_feature = f_feature.view(x.size(0), 1, 512)
            param_f = torch.bmm(f_feature, label_feature.view(label_feature.size(0), 512, -1)).view(
                label.size(0), -1)
        else:
            param_f = self.fc_f_1(f_feature)
        param = self.fc_b_1(b_feature) + param_f

        # return CF(x, param.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane)
        # print(param.shape)
        enhance_image = self.deep_curve(x, param.view(x.size(0), x.size(1), self.plane))
        return enhance_image


class CurveRenderwithConcat(nn.Module):
    def __init__(self, plane=128, final_relu=False, p=Projection):
        super(CurveRenderwithConcat, self).__init__()
        self.plane = plane
        self.fc_f_1 = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        self.fc_b_1 = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        print(self.plane)

        self.fc_label_1 = nn.Linear(5, 32)
        self.fc_label_2 = nn.Linear(32, 64)
        self.fc_f_label = nn.Linear(512 + 64, 3 * self.plane, bias=True)

    def forward(self, x, f_feature, b_feature, label=None):
        # import pdb;
        # pdb.set_trace()
        # print(f_feature.shape)
        if label is not None:
            label_feature = self.fc_label_2(F.relu(self.fc_label_1(label)))

            f_feature = torch.cat((f_feature, label_feature), dim=1)
            param_f = self.fc_f_label(f_feature)
        else:
            param_f = self.fc_f_1(f_feature)
        param = self.fc_b_1(b_feature) + param_f
        # print(param.shape)

        return CF(x, param.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane), param


class CurveRenderwithDeepCurve(nn.Module):
    def __init__(self, plane=128, final_relu=False, p=Projection):
        super(CurveRenderwithDeepCurve, self).__init__()
        self.plane = plane
        self.fc_f_1 = p(512, 3, final_relu)  # bs x 3 x 64 #
        self.fc_b_1 = p(512, 3, final_relu)  # bs x 3 x 64 #
        print(self.plane)

        self.fc_label_1 = nn.Linear(5, 32)
        self.fc_label_2 = nn.Linear(32, 64)
        self.fc_f_label = nn.Linear(512 + 64, 3, bias=True)

    def forward(self, x, f_feature, b_feature, label=None):
        # import pdb;
        # pdb.set_trace()
        if label is not None:
            label_feature = self.fc_label_2(F.relu(self.fc_label_1(label)))

            f_feature = torch.cat((f_feature, label_feature), dim=1)
            param_f = self.fc_f_label(f_feature)
        else:
            param_f = self.fc_f_1(f_feature)
        param = self.fc_b_1(b_feature) + param_f
        # print(param.shape)

        return CF_deepCurve(x, param.view(x.size(0), x.size(1), 1, 1), self.plane), param


class CurveRenderwithLUT(nn.Module):
    def __init__(self, plane=128, final_relu=False, p=Projection):
        super(CurveRenderwithLUT, self).__init__()
        self.plane = plane
        self.fc_f_1 = p(512, 3, final_relu)  # bs x 3 x 64 #
        self.fc_b_1 = p(512, 3, final_relu)  # bs x 3 x 64 #
        print(self.plane)

        self.fc_label_1 = nn.Linear(5, 32)
        self.fc_label_2 = nn.Linear(32, 64)
        self.fc_f_label = nn.Linear(512 + 64, 3, bias=True)

        self.LUT0 = Generator3DLUT_identity()
        self.LUT1 = Generator3DLUT_zero()
        self.LUT2 = Generator3DLUT_zero()
        trilinear_ = TrilinearInterpolation()

        self.LUT0 = self.LUT0.cuda()
        self.LUT1 = self.LUT1.cuda()
        self.LUT2 = self.LUT2.cuda()
        self.LUTS = [self.LUT0, self.LUT1, self.LUT2]

    def forward(self, x, f_feature, b_feature, label=None):
        # import pdb;
        # pdb.set_trace()
        if label is not None:
            label_feature = self.fc_label_2(F.relu(self.fc_label_1(label)))

            f_feature = torch.cat((f_feature, label_feature), dim=1)
            param_f = self.fc_f_label(f_feature)
        else:
            param_f = self.fc_f_1(f_feature)
        param = self.fc_b_1(b_feature) + param_f
        # print(param.shape)

        return CF_LUT(x, param.view(x.size(0), x.size(1), 1, 1), self.plane, self.LUTS), param


class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        if dim == 33:
            file = open("IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("IdentityLUT64.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)
        # print(lines)
        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k
                    x = lines[n].split()
                    # print(x)
                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        # self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
        return output


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
        self.LUT = nn.Parameter(torch.tensor(self.LUT))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)

        return output


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        assert 1 == trilinear.forward(lut,
                                      x,
                                      output,
                                      dim,
                                      shift,
                                      binsize,
                                      W,
                                      H,
                                      batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        assert 1 == trilinear.backward(x,
                                       x_grad,
                                       lut_grad,
                                       dim,
                                       shift,
                                       binsize,
                                       W,
                                       H,
                                       batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


class Fusion(nn.Module):
    def __init__(self, plane=64, stack=False, p=Projection, final_relu=False, cr1=CurveRenderwithConcat,
                 cr2=CurveRenderwithConcat):
        super(Fusion, self).__init__()
        self.plane = plane
        self.stack = stack
        print('stack:', self.stack)
        self.cr1 = cr1(self.plane, final_relu, p=p)
        self.cr2 = cr2(self.plane, final_relu, p=p)

    def forward(self, ori_img, x, f_feature, b_feature, label, withLabel):
        # label_feature [3 * 64 * 512, 1]
        # import pdb;
        # pdb.set_trace()
        img_1, param1 = self.cr1(x, f_feature, b_feature, label)
        if self.stack:
            img_2, param2 = self.cr2(img_1, f_feature, b_feature, label)
            fusion_img = (img_2, img_1)

        else:
            _, param2 = img_1, param1
            fusion_img = (img_1, img_1)
        # import pdb;
        # pdb.set_trace()
        return fusion_img, param1, param2


class CreateNetFusion_vgg(nn.Module):
    def __init__(self, model, plane=64, fusion=Fusion, final_relu=False, stack=False, cr1=CurveRender,
                 cr2=CurveRender):
        super(CreateNetFusion_vgg, self).__init__()

        self.model = nn.Sequential(*list(model.children())[0][:30])
        self.fusion = fusion(plane=plane, final_relu=final_relu, stack=stack, cr1=CurveRenderwithConcat,
                             cr2=CurveRenderwithConcat)
        self.stack = stack
        print(self.model)

    def forward(self, ori_img, x, fore, label, withLabel):
        x, m = x[:, 0:3], x[:, 3:]
        fx, fm = fore[:, 0:3], fore[:, 3:]

        f = torch.cat([fx * fm, x * (1 - m)], dim=0)
        feature = self.model(f)
        # import pdb;
        # pdb.set_trace()
        f_feature, b_feature = torch.split(feature, feature.size(0) // 2, dim=0)
        # print(f_feature.shape)
        f_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1)
        b_feature = F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), -1)

        fusion_img, param1, param2 = self.fusion(ori_img, x, f_feature, b_feature, label, withLabel)

        return fusion_img, param1, param2


class net_D(nn.Module):
    def __init__(self, in_channels=85):
        super(net_D, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 5, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.local = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        y = self.local(img)
        return y
