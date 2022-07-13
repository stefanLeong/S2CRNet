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

from typing import Type, Any, Callable, Union, List, Optional
from torch.utils.model_zoo import load_url
from torch import Tensor

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'alexnet': "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    'mobilenet_v2': "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
    'mobilenet_v3_large': "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    'mobilenet_v3_small': "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


#

class MinimalUnetV2(nn.Module):
    """docstring for MinimalUnet"""

    def __init__(self, down=None, up=None, submodule=None, attention=None, withoutskip=False, **kwags):
        super(MinimalUnetV2, self).__init__()

        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.sub = submodule
        self.attention = attention
        self.withoutskip = withoutskip
        self.is_attention = not self.attention == None
        self.is_sub = not submodule == None

    def forward(self, x, mask=None):
        if self.is_sub:
            x_up, _ = self.sub(self.down(x), mask)
        else:
            x_up = self.down(x)

        if self.withoutskip:  # outer or inner.
            x_out = self.up(x_up)
        else:
            if self.is_attention:
                x_out = (self.attention(torch.cat([x, self.up(x_up)], 1), mask), mask)
            else:
                x_out = (torch.cat([x, self.up(x_up)], 1), mask)

        return x_out


class MinimalUnet(nn.Module):
    """docstring for MinimalUnet"""

    def __init__(self, down=None, up=None, submodule=None, attention=None, withoutskip=False, **kwags):
        super(MinimalUnet, self).__init__()

        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.sub = submodule
        self.attention = attention
        self.withoutskip = withoutskip
        self.is_attention = not self.attention == None
        self.is_sub = not submodule == None

    def forward(self, x, mask=None):
        if self.is_sub:
            x_up, _ = self.sub(self.down(x), mask)
        else:
            x_up = self.down(x)

        if self.is_attention:
            x = self.attention(x, mask)

        if self.withoutskip:  # outer or inner.
            x_out = self.up(x_up)
        else:
            x_out = (torch.cat([x, self.up(x_up)], 1), mask)

        return x_out


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False,
                 attention_model=RASC, basicblock=MinimalUnet, outermostattention=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = basicblock(down, up, submodule, withoutskip=outermost)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = basicblock(down, up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if is_attention_layer:
                if MinimalUnetV2.__qualname__ in basicblock.__qualname__:
                    attention_model = attention_model(input_nc * 2)
                else:
                    attention_model = attention_model(input_nc)
            else:
                attention_model = None

            if use_dropout:
                model = basicblock(down, up.append(nn.Dropout(0.5)), submodule, attention_model,
                                   outermostattention=outermostattention)
            else:
                model = basicblock(down, up, submodule, attention_model, outermostattention=outermostattention)

        self.model = model

    def forward(self, x, mask=None):
        # build the mask for attention use
        return self.model(x, mask)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False, attention_model=RASC, use_inner_attention=False, basicblock=MinimalUnet):
        super(UnetGenerator, self).__init__()

        # 8 for 256x256
        # 9 for 512x512
        # construct unet structure
        self.need_mask = not input_nc == output_nc

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, basicblock=basicblock)  # 1
        for i in range(num_downs - 5):  # 3 times
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout,
                                                 is_attention_layer=use_inner_attention,
                                                 attention_model=attention_model, basicblock=basicblock)  # 8,4,2
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, is_attention_layer=is_attention_layer,
                                             attention_model=attention_model, basicblock=basicblock)  # 16
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, is_attention_layer=is_attention_layer,
                                             attention_model=attention_model, basicblock=basicblock)  # 32
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             is_attention_layer=is_attention_layer, attention_model=attention_model,
                                             basicblock=basicblock, outermostattention=True)  # 64
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             basicblock=basicblock, norm_layer=norm_layer)  # 128

        self.model = unet_block

        self.optimizers = []

    def forward(self, input):
        if self.need_mask:
            return self.model(input, input[:, 3:4, :, :]), input
        else:
            return self.model(input[:, 0:3, :, :], input[:, 3:4, :, :]), input

    def set_optimizers(self, lr):
        self.optimizer_encoder = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizers.append(self.optimizer_encoder)

    def zero_grad_all(self):
        self.model.zero_grad()

    def step_all(self):
        self.optimizer_encoder.step()


class XBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU,
                 batch_norm=False, ins_norm=False):
        super(XBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None

        if batch_norm:
            self.bn = nn.BatchNorm2d(outc)
        elif ins_norm:
            self.bn = nn.InstanceNorm2d(outc)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


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


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)



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


class CreateNetFusion_res(nn.Module):
    def __init__(self, model, plane=64, fusion=Fusion, final_relu=False, stack=False, cr1=CurveRender,
                 cr2=CurveRender):
        super(CreateNetFusion_res, self).__init__()

        self.model = nn.Sequential(*list(model.children())[:8])
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
