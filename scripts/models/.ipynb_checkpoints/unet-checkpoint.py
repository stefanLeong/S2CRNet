import torch
import torch.nn as nn
from torch.nn import init
import functools
from scripts.models.blocks import *
from scripts.models.rasc import *
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import re
import random
import torchvision
from torchvision.utils import save_image as si

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
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
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU, batch_norm=False,ins_norm=False):
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
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False, ins_norm=False):
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


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),# 
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),#
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),#
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def squeezenet1_0(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            XBlock(3, 32, kernel_size=3, stride=2, padding=1), #  128
            XBlock(32,64, kernel_size=3, stride=2, padding=1), # 64
            XBlock(64,128, kernel_size=3, stride=2, padding=1), # 32
            XBlock(128, 256, kernel_size=3, stride=2, padding=1), # 16
            XBlock(256, 512, kernel_size=3, stride=2, padding=1, activation=None) # 8x8
        )
        self.fc_f = nn.Linear(512, 192) # bs x 3 x 64 # 
        self.fc_b = nn.Linear(512, 192)
        self.optimizers = []

    def forward(self,x):
        x,mask = x[:,0:3],x[:,3:]
        # 256 dim features
        param_f = self.fc_f(F.adaptive_avg_pool2d(self.features(x*mask),1).view(x.size(0), -1))
        param_b = self.fc_b(F.adaptive_avg_pool2d(self.features(x*(1-mask)),1).view(x.size(0), -1))
        
        param = param_b + param_f

        return CF(x,param.view(x.size(0),x.size(1),1,1,64),64) # bsx64



def CF(img, param, pieces):
    
    pieces = torch.
    
    # bs x 3 x 1 x 1 x 64
    color_curve_sum = torch.sum(param, 4) + 1e-30
    total_image = img * 0

    for i in range(pieces):
        total_image += torch.clamp(img - 1.0 * i /pieces, 0, 1.0 / pieces) * param[:, :, :, :, i]
    total_image *= pieces/ color_curve_sum
    return total_image

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

    
class BasicNet(nn.Module):
    def __init__(self,model, plane=64):
        super(BasicNet,self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.plane = plane
        # 512 -> 192
        self.fc_f = nn.Linear(512, 3*self.plane) # bs x 3 x 64 # 
        # self.fc_b = nn.Linear(512, 3*self.plane) # bs x 3 x 64 # 
     
        self.norm = nn.Identity()
        self.optimizers = []


    def forward(self,x):
        x,m = x[:,0:3],x[:,3:]

        f_feature = self.squeezenet1_1(x)

        param = self.fc_f(F.adaptive_avg_pool2d(f_feature,1).view(x.size(0), -1))
        

        return CF(x,param.view(x.size(0),x.size(1),1,1,self.plane),self.plane) # bsx64


class ForeNet(nn.Module):
    def __init__(self,model, plane=64):
        super(ForeNet,self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.plane = plane
        # 512 -> 192
        self.fc_f = nn.Linear(512, 3*self.plane) # bs x 3 x 64 # 
        self.fc_b = nn.Linear(512, 3*self.plane) # bs x 3 x 64 # 
        # self.fc = nn.Linear(2*3*self.plane,3*self.plane) # bs x 3 x 64 # 

        # self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        self.norm = nn.Identity()
        self.optimizers = []

        # for param in self.squeezenet1_1.parameters():
        #     param.requires_grad = False

    def forward(self,x):
        x,m = x[:,0:3],x[:,3:]

        f_feature = self.squeezenet1_1(x*m)
        #b_feature = self.squeezenet1_1(x*(1-m))

        param_f = self.fc_f(F.adaptive_avg_pool2d(f_feature,1).view(x.size(0), -1))
        #param_b = self.fc_b(F.adaptive_avg_pool2d(b_feature,1).view(x.size(0), -1))

        param = param_f #+ param_b
        # param = torch.cat([param_f,param_b],dim=1)
        # param = self.fc(F.relu(param))

        return CF(x,param.view(x.size(0),x.size(1),1,1,self.plane),self.plane) # bsx64

class CreateNet(nn.Module):
    def __init__(self,model, plane=64):
        super(CreateNet,self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.plane = plane
        # 512 -> 192
        self.fc_f = nn.Linear(512, 3*self.plane) # bs x 3 x 64 # 
        self.fc_b = nn.Linear(512, 3*self.plane) # bs x 3 x 64 # 
        # self.fc = nn.Linear(2*3*self.plane,3*self.plane) # bs x 3 x 64 # 

        # self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        self.norm = nn.Identity()
        self.optimizers = []

        # for param in self.squeezenet1_1.parameters():
        #     param.requires_grad = False

    def forward(self,x):
        x,m = x[:,0:3],x[:,3:]

        f_feature = self.squeezenet1_1(x*m)
        b_feature = self.squeezenet1_1(x*(1-m))

        param_f = self.fc_f(F.adaptive_avg_pool2d(f_feature,1).view(x.size(0), -1))
        param_b = self.fc_b(F.adaptive_avg_pool2d(b_feature,1).view(x.size(0), -1))

        param = param_f + param_b
        # param = torch.cat([param_f,param_b],dim=1)
        # param = self.fc(F.relu(param))

        return CF(x,param.view(x.size(0),x.size(1),1,1,self.plane),self.plane) # bsx64

class CreateSimpleNet(nn.Module):
    def __init__(self,  plane=64):
        super(CreateSimpleNet,self).__init__()
        self.net = SimpleNet()

        self.plane = plane
        # 512 -> 192
        self.fc_f = nn.Linear(512, 3*self.plane) # bs x 3 x 64 # 
        self.fc_b = nn.Linear(512, 3*self.plane) # bs x 3 x 64 # 
        # self.fc = nn.Linear(2*3*self.plane,3*self.plane) # bs x 3 x 64 # 

        # self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        self.norm = nn.Identity()
        self.optimizers = []

        # for param in self.squeezenet1_1.parameters():
        #     param.requires_grad = False

    def forward(self,x):
        x,m = x[:,0:3],x[:,3:]

        f_feature = self.squeezenet1_1(x*m)
        b_feature = self.squeezenet1_1(x*(1-m))

        param_f = self.fc_f(F.adaptive_avg_pool2d(f_feature,1).view(x.size(0), -1))
        param_b = self.fc_b(F.adaptive_avg_pool2d(b_feature,1).view(x.size(0), -1))

        param = param_f + param_b
        # param = torch.cat([param_f,param_b],dim=1)
        # param = self.fc(F.relu(param))

        return CF(x,param.view(x.size(0),x.size(1),1,1,self.plane),self.plane) # bsx64

class CreateNet3stage(nn.Module):
    def __init__(self):
        super(CreateNet3stage,self).__init__()
        self.s1 = CreateNet(squeezenet1_1(pretrained=True))
        self.s2 = CreateNet(squeezenet1_1(pretrained=True))
        self.s3 = CreateNet(squeezenet1_1(pretrained=True))


    def forward(self,x):
        m = x[:,3:]
        x1 = self.s1(x)
        x2 = self.s2(torch.cat([x1,m],dim=1))
        x3 = self.s3(torch.cat([x2,m],dim=1))
        return x3,x2,x1

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
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU, batch_norm=False):
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
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
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

class CreateNetADAIN(nn.Module):
    def __init__(self,model):
        super(CreateNetADAIN,self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.plane = 64
        # # 512 -> 192
        # self.fc_f = nn.Linear(512, 3*self.plane) # bs x 3 x 64 # 
        
        # self.fc_b = nn.Linear(512, 3*self.plane) # bs x 3 x 64 # 
        self.fc = nn.Linear(512,3*self.plane) # bs x 3 x 64 # 

        # self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        self.norm = nn.Identity()
        self.optimizers = []
        self.alpha = 0.5



    def forward(self,x):
        x,m = x[:,0:3],x[:,3:]

        f_feature = self.squeezenet1_1(x)
        # b_feature = self.squeezenet1_1(x)

        # t = adaptive_instance_normalization(f_feature, b_feature)
        # t = self.alpha * t + (1 - self.alpha) * f_feature

        # param_f = self.fc_f(F.adaptive_avg_pool2d(f_feature,1).view(x.size(0), -1))
        # param_b = self.fc_b(F.adaptive_avg_pool2d(b_feature,1).view(x.size(0), -1))


        # param = param_f + param_b
        # param = torch.cat([param_f,param_b],dim=1)
        param = self.fc(F.adaptive_avg_pool2d(f_feature,1).view(x.size(0), -1))

        return CF(x,param.view(x.size(0),x.size(1),1,1,self.plane),self.plane) # bsx64

    # def set_optimizers(self, lr):
    #     self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
    #     self.optimizers.append(self.optimizer)

    # def zero_grad_all(self):
    #     self.zero_grad()

    # def step_all(self):
    #     self.optimizer.step()



# class PaperNeRFModel(torch.nn.Module):
#     r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
#     arXiv submission (v0). """

#     def __init__(
#         self,
#         num_layers=8,
#         hidden_size=256,
#         skip_connect_every=4,
#         num_encoding_fn_xyz=6,
#         num_encoding_fn_dir=4,
#         include_input_xyz=True,
#         include_input_dir=True,
#         use_viewdirs=True,
#     ):
#         super(PaperNeRFModel, self).__init__()

#         include_input_xyz = 3 if include_input_xyz else 0
#         include_input_dir = 3 if include_input_dir else 0
#         self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
#         self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

#         self.layers_xyz = torch.nn.ModuleList()
#         self.use_viewdirs = use_viewdirs
#         self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, 256))
#         for i in range(1, 8):
#             if i == 4:
#                 self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + 256, 256))
#             else:
#                 self.layers_xyz.append(torch.nn.Linear(256, 256))
#         self.fc_feat = torch.nn.Linear(256, 256)
#         self.fc_alpha = torch.nn.Linear(256, 1)

#         self.layers_dir = torch.nn.ModuleList()
#         self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
#         for i in range(3):
#             self.layers_dir.append(torch.nn.Linear(128, 128))
#         self.fc_rgb = torch.nn.Linear(128, 3)
#         self.relu = torch.nn.functional.relu

#     def forward(self, x):
#         xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
#         for i in range(8):
#             if i == 4:
#                 x = self.layers_xyz[i](torch.cat((xyz, x), -1))
#             else:
#                 x = self.layers_xyz[i](x)
#             x = self.relu(x)
#         feat = self.fc_feat(x)
#         alpha = self.fc_alpha(feat)
#         if self.use_viewdirs:
#             x = self.layers_dir[0](torch.cat((feat, dirs), -1))
#         else:
#             x = self.layers_dir[0](feat)
#         x = self.relu(x)
#         for i in range(1, 3):
#             x = self.layers_dir[i](x)
#             x = self.relu(x)
#         rgb = self.fc_rgb(x)
#         return torch.cat((rgb, alpha), dim=-1)


from collections import OrderedDict


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        # Nx12x8x16x16
        device = bilateral_grid.get_device()
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap ], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, 'bilinear', align_corners=True)
        
        return coeff.squeeze(2)

class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):

        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''
        
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)

class GuideNN(nn.Module):
    def __init__(self):
        super(GuideNN, self).__init__()
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=True)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Sigmoid) #nn.Tanh

    def forward(self, x):
        return self.conv2(self.conv1(x))#.squeeze(1)

class Coeffs(nn.Module):

    def __init__(self, nin=4, nout=3):
        super(Coeffs, self).__init__()
        self.nin = nin 
        self.nout = nout
        
        lb = 8
        cm = 1
        sb = 16
        bn = False
        nsize = 256

        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize/sb))
        self.splat_features = nn.ModuleList()
        prev_ch = 3
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(ConvBlock(prev_ch, cm*(2**i)*lb, 3, stride=2, batch_norm=use_bn))
            prev_ch = splat_ch = cm*(2**i)*lb

        # global features
        n_layers_global = int(np.log2(sb/4))
        print(n_layers_global)
        self.global_fore_features_conv = nn.ModuleList()
        self.global_back_features_conv = nn.ModuleList()

        self.global_fore_features_fc = nn.ModuleList()
        self.global_back_features_fc = nn.ModuleList()

        x_channel = prev_ch
        for i in range(n_layers_global):
            self.global_fore_features_conv.append(ConvBlock(x_channel, cm*8*lb, 3, stride=2, batch_norm=bn))
            x_channel = cm*8*lb

        for i in range(n_layers_global):
            self.global_back_features_conv.append(ConvBlock(prev_ch, cm*8*lb, 3, stride=2, batch_norm=bn))
            prev_ch = cm*8*lb

        n_total = n_layers_splat + n_layers_global
        prev_ch = prev_ch * (nsize/2**n_total)**2
        self.global_fore_features_fc.append(FC(prev_ch, 32*cm*lb, batch_norm=bn))
        self.global_fore_features_fc.append(FC(32*cm*lb, 16*cm*lb, batch_norm=bn))
        self.global_fore_features_fc.append(FC(16*cm*lb, 8*cm*lb, activation=None, batch_norm=bn))

        self.global_back_features_fc.append(FC(prev_ch, 32*cm*lb, batch_norm=bn))
        self.global_back_features_fc.append(FC(32*cm*lb, 16*cm*lb, batch_norm=bn))
        self.global_back_features_fc.append(FC(16*cm*lb, 8*cm*lb, activation=None, batch_norm=bn))

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(splat_ch, 8*cm*lb, 3, batch_norm=bn))
        self.local_features.append(ConvBlock(8*cm*lb, 8*cm*lb, 3, activation=None, use_bias=False))
        
        # predicton
        self.conv_out = ConvBlock(8*cm*lb, lb*nout*nin, 1, padding=0, activation=None)

   
    def forward(self, lowres_input, mask):

        bs = lowres_input.shape[0]
        lb = 8
        cm = 1
        sb = 16

        x = lowres_input * mask
        for layer in self.splat_features:
            x = layer(x)
        fore_features = x

        x = lowres_input * (1 - mask)
        for layer in self.splat_features:
            x = layer(x)
        back_features = x
        
        x = fore_features
        for layer in self.global_fore_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        for layer in self.global_fore_features_fc:
            x = layer(x)
        global_fore_features = x

        x = back_features
        for layer in self.global_back_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        for layer in self.global_back_features_fc:
            x = layer(x)
        global_back_features = x

        x = fore_features + back_features
        for layer in self.local_features:
            x = layer(x)        
        local_features = x

        global_features = global_fore_features + global_back_features

        fusion_grid = local_features
        fusion_global = global_features.view(bs,8*cm*lb,1,1)
        fusion = self.relu( fusion_grid + fusion_global )

        x = self.conv_out(fusion)
        s = x.shape
        y = torch.stack(torch.split(x, self.nin*self.nout, 1),2)
        # y = torch.stack(torch.split(y, self.nin, 1),3)
        # print(y.shape)
        # x = x.view(bs,self.nin*self.nout,lb,sb,sb) # B x Coefs x Luma x Spatial x Spatial
        # print(x.shape)
        return y


class HDRPointwiseNN(nn.Module):

    def __init__(self):
        super(HDRPointwiseNN, self).__init__()
        self.coeffs = Coeffs()
        self.guide = GuideNN()
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, x):
        img,mask = x[:,:3,:,:],x[:,3:,:,:]
        coeffs = self.coeffs(img, mask) 
        guide = self.guide(img)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, img)
        return out


#########################################################################################################


class CreateNetFusion(nn.Module):
    def __init__(self, model, norm=False, use_adaptor=False, fc='basic', use_da=False):
        super(CreateNetFusion, self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.plane = 128
        # 512 -> 192
        self.fc_f_1 = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        self.fc_f_2 = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        self.fc_b_1 = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        self.fc_b_2 = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        # self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        #adaptive weights
        self.w1 = nn.Parameter(torch.zeros([1, 3, 1, 1]).fill_(0.5))
        self.w2 = nn.Parameter(torch.zeros([1, 3, 1, 1]).fill_(0.5))
        self.norm = nn.Identity()
        self.optimizers = []

        # for param in self.squeezenet1_1.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x, m = x[:, 0:3], x[:, 3:]
        
        xx = F.interpolate(x,size=(256,256))
        xm = F.interpolate(m,size=(256,256))
        
        
        f = torch.cat([xx*xm,xx*(1-xm)],dim=0)

        feature = self.squeezenet1_1(f)
        
        f_feature, b_feature = F.adaptive_avg_pool2d(feature[0:1], 1).view(x.size(0), -1),F.adaptive_avg_pool2d(feature[1:], 1).view(x.size(0), -1)

        param_f_1 = self.fc_f_1(f_feature)
        param_f_2 = self.fc_f_2(f_feature)
        param_b_1 = self.fc_b_1(b_feature)
        param_b_2 = self.fc_b_2(b_feature)

        param_1 = param_b_1 + param_f_1
        param_2 = param_b_2 + param_f_2

        img_1 = CF(x, param_1.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane)
        img_2 = CF(x, param_2.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane)
        fusion_img = self.w1 * img_1 + self.w2 * img_2

        return fusion_img