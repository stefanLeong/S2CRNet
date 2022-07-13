import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.models.rasc import *
from scripts.models.unet import UnetGenerator,MinimalUnetV2
from scripts.models.aspp_att import CAPP



def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def up_conv2x2(in_channels, out_channels, transpose=True):
    if transpose:
        return nn.ConvTranspose2d(  #fractionally-strides convolutions
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class UpCoXvD(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, residual=True, norm=nn.BatchNorm2d, act=F.leaky_relu,
                 batch_norm=True, transpose=True, concat=True, use_att=True):
        super(UpCoXvD, self).__init__()
        self.concat = concat
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.conv2 = []
        self.use_att = use_att
        self.up_conv = up_conv2x2(in_channels, out_channels, transpose=transpose)
        self.norm0 = norm(out_channels)

        if self.use_att:
            self.capp = CAPP(2 * out_channels)
        else:
            self.capp = None

        if self.concat:
            self.conv1 = conv3x3(2 * out_channels, out_channels)
            self.norm1 = norm(out_channels, out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
            self.norm1 = norm(out_channels, out_channels)

        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(norm(out_channels))
            self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def forward(self, from_up, from_down, mask=None, se=None):
        from_up = self.act(self.norm0(self.up_conv(from_up)))
        if self.concat:
            x1 = torch.cat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up

        if self.use_att:
            x1 = self.capp(x1, mask)

        x1 = self.act(self.norm1(self.conv1(x1)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)

            if (se is not None) and (idx == len(self.conv2) - 1):  # last
                x2 = se(x2)

            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        return x2


class DownCoXvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, pooling=True, norm=nn.BatchNorm2d, act=F.leaky_relu, residual=True, batch_norm=True):
        super(DownCoXvD, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.pool = None
        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm1 = norm(out_channels)

        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(norm(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x1 = self.act(self.norm1(self.conv1(x)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool


class UnetDecoderD(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, use_att=False):
        super(UnetDecoderD, self).__init__()
        self.conv_final = None
        self.up_convs = []
        outs = in_channels
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            # 512,256
            # 256,128
            # 128,64
            # 64,32
            up_conv = UpCoXvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, use_att=use_att)
            self.up_convs.append(up_conv)
        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpCoXvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, use_att=use_att)
            self.up_convs.append(up_conv)
        self.up_convs = nn.ModuleList(self.up_convs)

    def __call__(self, x, encoder_outs=None, mask=None):
        return self.forward(x, encoder_outs, mask)

    def forward(self, x, encoder_outs=None, mask=None):
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool, mask, se=None)
        if self.conv_final is not None:
            x = self.conv_final(x)
        return x


class UnetEncoderD(nn.Module):
    def __init__(self, in_channels=3, depth=5, blocks=1, start_filters=32, residual=True, batch_norm=True):
        super(UnetEncoderD, self).__init__()
        self.down_convs = []
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters * (2 ** i)
            pooling = True if i < depth - 1 else False
            down_conv = DownCoXvD(ins, outs, blocks, pooling=pooling, residual=residual, batch_norm=batch_norm, norm=nn.BatchNorm2d, act=F.leaky_relu)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)
        return x, encoder_outs

class UnetDecoderSplit(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True):
        super(UnetDecoderSplit, self).__init__()
        self.conv_final = None
        self.up_convs = []
        self.real_atts = []
        self.fake_atts = []

        outs = in_channels
        for i in range(depth - 1): # depth = 5 [0,1,2,3]
            ins = outs
            outs = ins // 2
            # 512,256
            # 256,128
            # 128,64
            # 64,32
            up_conv = UpCoXvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, norm=nn.BatchNorm2d, act=F.leaky_relu, use_att=False)
            self.up_convs.append(up_conv)

        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpCoXvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, norm=nn.BatchNorm2d, act=F.leaky_relu, use_att=False)
            self.up_convs.append(up_conv)

        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, input, encoder_outs=None):
        # im branch
        x = input
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool, se=None)
        x_real = x

        x = input
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool, se=None)
        x_fake = x

        return x_real, x_fake


class PatchSampleF(nn.Module):
    def __init__(self, device= None, use_mlp=False, nc=256):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.device = device

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            mlp.to(self.device)
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        self.mlp_init = True

    def forward(self, feats, num_patches=256, patch_ids=None, mask_outs=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        if mask_outs is not None:
            for feat_id, feat in enumerate(feats):
                B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
                mask_out = mask_outs[feat_id]
                mask_reshape = mask_out.permute(0, 2, 3, 1).flatten(1, 2)
                if num_patches > 0:
                    if patch_ids is not None:
                        patch_id = patch_ids[feat_id]
                    else:
                        patch_id = torch.randperm(mask_reshape.shape[1], device=feats[0].device).nonzero(as_tuple =False).flatten(0, 1)
                        # print(patch_id.shape)
                        # print(mask_out.nonzero(as_tuple =False).flatten(0, 1).shape)
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                    x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
                else:
                    x_sample = feat_reshape
                    patch_id = []
                if self.use_mlp:
                    mlp = getattr(self, 'mlp_%d' % feat_id)
                    x_sample = mlp(x_sample)
                return_ids.append(patch_id)
                x_sample = self.l2norm(x_sample)

                if num_patches == 0:
                    x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
                return_feats.append(x_sample)
        if mask_outs is None:
            for feat_id, feat in enumerate(feats):
                B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
                if num_patches > 0:
                    if patch_ids is not None:
                        patch_id = patch_ids[feat_id]
                    else:
                        patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                    x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
                else:
                    x_sample = feat_reshape
                    patch_id = []
                if self.use_mlp:
                    mlp = getattr(self, 'mlp_%d' % feat_id)
                    x_sample = mlp(x_sample)
                return_ids.append(patch_id)
                x_sample = self.l2norm(x_sample)

                if num_patches == 0:
                    x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
                return_feats.append(x_sample)

        return return_feats, return_ids


class SplitwithCAPPV3(nn.Module):
    def __init__(self, in_channels=3, depth=5, shared_depth=0, blocks=1,
                 out_channels_image=3, start_filters=32, residual=True, batch_norm=nn.InstanceNorm2d,
                 transpose=True, concat=True, transfer_data=True, long_skip=False):
        super(SplitwithCAPPV3, self).__init__()
        self.transfer_data = transfer_data
        self.shared = shared_depth
        self.optimizer_encoder, self.optimizer_real, self.optimizer_fake = None, None, None
        self.optimizer_shared = None
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks)
        if not transfer_data:
            concat = False
        self.encoder = UnetEncoderD(in_channels=in_channels, depth=depth, blocks=blocks[0],
                                    start_filters=start_filters, residual=residual, batch_norm=batch_norm)
        self.fake_region_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                                out_channels=out_channels_image, depth=depth- shared_depth,
                                                blocks=blocks[1], residual=residual, batch_norm=batch_norm,
                                                transpose=transpose, concat=concat, use_att=False)
        self.real_region_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                                out_channels=out_channels_image, depth=depth- shared_depth,
                                                blocks=blocks[2], residual=residual, batch_norm=batch_norm,
                                                transpose=transpose, concat=concat, use_att=True)
        self.share_decoder = None
        self.long_skip = long_skip
        # self._forward = self.unshared_forward
        if self.shared != 0:
            # self._forward = self.shared_forward
            self.shared_decoder = UnetDecoderSplit(in_channels=start_filters * 2 ** (depth - 1),
                                                  out_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                                  depth=shared_depth, blocks=blocks[3], residual=residual,
                                                  batch_norm=batch_norm, transpose=transpose, concat=concat,
                                                  is_final=False)

    def set_optimizers(self):
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        self.optimizer_fake = torch.optim.Adam(self.fake_region_decoder.parameters(), lr = 0.001)
        self.optimizer_real = torch.optim.Adam(self.real_region_decoder.parameters(), lr=0.001)
        if self.shared != 0:
            self.optimizer_shared = torch.optim.Adam(self.shared_decoder.parameters(), lr=0.001)

    def zero_grad_all(self):
        self.optimizer_encoder.zero_grad()
        self.optimizer_fake.zero_grad()
        self.optimizer_real.zero_grad()
        if self.shared != 0:
            self.optimizer_shared.zero_grad()

    def step_all(self):
        self.optimizer_encoder.step()
        self.optimizer_fake.step()
        self.optimizer_real.step()
        if self.shared != 0:
            self.optimizer_shared.step()

    # def __call__(self, synthesized):
    #     return self._forward(synthesized)

    #shared foward
    def forward(self, synthesized):
        synthesized, mask = synthesized[:, 0:3, :, :], synthesized[:, 3:4, :, :]
        image_code, before_pool = self.encoder(synthesized)
        if self.transfer_data:
            shared_before_pool = before_pool[-self.shared - 1:]
            unshared_before_pool = before_pool[:-self.shared]
        else:
            before_pool = None
            shared_before_pool = None
            unshared_before_pool = None
        real, fake = self.shared_decoder(image_code, shared_before_pool)
        reconstructed_fake = torch.tanh(self.fake_region_decoder(fake, unshared_before_pool, mask))
        reconstructed_real = torch.tanh(self.real_region_decoder(real, unshared_before_pool, mask))
        if self.long_skip:
            reconstructed_fake = reconstructed_fake + synthesized
            reconstructed_real = reconstructed_real + synthesized

        _, mask_outs = self.encoder(torch.cat((mask, mask, mask), dim=1))
        for tensor in mask_outs:
            tensor = tensor[:, 0:1, :, :]
        ori_idt = before_pool
        _, real_idt = self.encoder(reconstructed_real)
        _, fake_idt = self.encoder(reconstructed_fake)

        mask_outs= mask_outs[:-2]
        mask_outs.reverse()
        mask_outs.append(mask_outs[-1])

        real_idt = real_idt[1:]
        fake_idt = fake_idt[1:]

        return reconstructed_fake, reconstructed_real, ori_idt, real_idt, fake_idt, mask_outs

    # def unshared_forward(self, synthesized):
    #     image_code, before_pool = self.encoder(synthesized)
    #     if not self.transfer_data:
    #         before_pool = None
    #     reconstructed_fake = torch.tanh(self.fake_region_decoder(image_code, before_pool))
    #     reconstructed_real = torch.tanh(self.real_region_decoder(image_code, before_pool))
    #     return reconstructed_fake, reconstructed_real

    #default
    def shared_forward(self, synthesized):
        synthesized, mask = synthesized[:, 0:3, :, :], synthesized[:, 3:4, :, :]
        image_code, before_pool = self.encoder(synthesized)



        if self.transfer_data:
            shared_before_pool = before_pool[-self.shared - 1:]
            unshared_before_pool = before_pool[:-self.shared]
        else:
            before_pool = None
            shared_before_pool = None
            unshared_before_pool = None
        real, fake = self.shared_decoder(image_code, shared_before_pool)
        reconstructed_fake = torch.tanh(self.fake_region_decoder(fake, unshared_before_pool, mask))
        reconstructed_real = torch.tanh(self.real_region_decoder(real, unshared_before_pool, mask))
        if self.long_skip:
            reconstructed_fake = reconstructed_fake + synthesized
            reconstructed_real = reconstructed_real + synthesized

        ori_idt = before_pool
        _, real_idt = self.encoder(reconstructed_real)
        _, fake_idt = self.encoder(reconstructed_fake)

        return reconstructed_fake, reconstructed_real, ori_idt, real_idt, fake_idt

