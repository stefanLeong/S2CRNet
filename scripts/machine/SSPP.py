import torch
import torch.nn as nn
from progress.bar import Bar
from tqdm import tqdm
import json
import sys, time, os
import torchvision
from math import log10
import numpy as np
from .BasicMachine import BasicMachine
from scripts.utils.evaluation import accuracy, AverageMeter, final_preds, compute_mse
from scripts.utils.misc import resize_to_match
from torch.autograd import Variable
import torch.nn.functional as F
from scripts.utils.parallel import DataParallelModel, DataParallelCriterion
from scripts.utils.losses import VGGLoss, l1_relative, is_dic, MaskWeightedMSE, set_requires_grad, GANLoss
from scripts.models.patchNCE import PatchNCELoss
from scripts.utils.imgUtils import im_to_numpy, tensor2img
from scripts.models.unet import net_D
import skimage.io
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
# import kornia.color as colors

class SSPP(BasicMachine):
    def __init__(self, **kwargs):
        BasicMachine.__init__(self, **kwargs)
        self.optimizers = []
        self.net_D = net_D(in_channels=3).to(self.device)
        self.optimizer_D = torch.optim.RMSprop(self.net_D.parameters(), lr=self.args.lr)
        # self.loss = Losses(self.args, self.device, self.norm, self.denorm)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.optimizers.append(self.optimizer)
        self.optimizers.append(self.optimizer_D)
        self.set_requires_grad = set_requires_grad
        # self.model.set_optimizers(self.args.lr)
        # self.optimizers = self.model.optimizers
        # self.criterion_GAN = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0).to(
            self.device)  # MSEloss
        if self.args.withLabel == 'True':
            self.withLabel = True
        elif self.args.withLabel == 'False':
            self.withLabel = False
        else:
            print("Args--withLabel input error!")
        print(self.withLabel)


    def train(self, epoch):

        self.current_epoch = epoch

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_relative = AverageMeter()
        loss_real = AverageMeter()
        loss_patch = AverageMeter()
        loss_total = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.train_loader))

        for i, batches in enumerate(self.train_loader):

            current_index = len(self.train_loader) * epoch + i

            inputs = batches['composite_images'].to(self.device)
            target = batches['real_images'].to(self.device)
            mask = batches['mask'].to(self.device)
            fore = batches["fore_images"].to(self.device) #: self.trans(thumb_foreground),
            foremask = batches["fore_mask"].to(self.device)  # "fore_mask":self.trans(thumb_mask)
            label = batches['label']
            # print(label.shape)
            label_oh = torch.zeros(self.args.train_batch, 5).scatter_(1, label.view(label.shape[0], 1), 1).to(
                torch.float32).to(self.device)
            self.optimizer.zero_grad()
            if self.withLabel:
                label_oh = label_oh
            else:
                label_oh = None
            feeded = torch.cat([inputs, mask], dim=1).to(self.device)
            fore = torch.cat([fore, foremask], dim=1).to(self.device)

            outputs = self.model(feeded, fore, label_oh, self.withLabel)
            outputs = outputs if type(outputs) == type(()) else [outputs]

            total_loss =  sum([l1_relative(output, target, mask) for output in outputs])
            # compute gradient and do SGD step
            total_loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss

            loss_real.update(total_loss.item(), inputs.size(0))
            loss_total.update(total_loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            suffix = "({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss Relative: {loss_img:.4f}  | loss Real: {loss_real:.4f} | loss Patch: {loss_patch:.5f} " \
                         "| loss total: {loss_TOTAL:.4f}".format(
                    batch=i + 1,
                    size=len(self.train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss_img=loss_relative.avg,
                    loss_real=loss_real.avg,
                    loss_patch=loss_patch.avg,
                    loss_TOTAL=loss_total.avg
            )
            if current_index % 1000 == 0:
                print(suffix)


        self.record('train/loss_relative', loss_relative.avg, epoch)
        self.record('train/loss_real', loss_real.avg, epoch)
        self.record('train/loss_Patch', loss_patch.avg, epoch)
        self.record('train/loss_TOTAL', loss_total.avg, epoch)

    def validate(self, epoch):

        self.current_epoch = epoch
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_relative = AverageMeter()

        psnres = AverageMeter()
        ssimes = AverageMeter()
        mses = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        # bar = Bar('Processing {} '.format(self.args.arch), max=len(self.val_loader))
        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):

                current_index = len(self.val_loader) * epoch + i

                inputs = batches['composite_images'].to(self.device)
                target = batches['real_images'].to(self.device)
                mask = batches['mask'].to(self.device)
                fore = batches["fore_images"].to(self.device) #: self.trans(thumb_foreground),
                foremask = batches["fore_mask"].to(self.device)  # "fore_mask":self.trans(thumb_mask),
                label = batches['label']
                # print(label.shape)
                label_oh = torch.zeros(self.args.test_batch, 5).scatter_(1, label.view(label.shape[0], 1), 1).to(
                    torch.float32).to(self.device)
                if self.withLabel:
                    label_oh = label_oh
                else:
                    label_oh = None

                feeded = torch.cat([inputs, mask], dim=1).to(self.device)
                fore = torch.cat([fore, foremask], dim=1).to(self.device)
                outputs = self.model(feeded, fore, label_oh, self.withLabel)
                outputs = outputs if type(outputs) == type(()) else [outputs]

                imfinal = outputs[0] * mask + inputs * (1 - mask)

                if i % 300 == 0:
                    # save the sample images
                    ims = torch.cat([inputs, target, imfinal, mask.repeat(1, 3, 1, 1)], dim=3)
                    torchvision.utils.save_image(ims, os.path.join(self.args.checkpoint, '%s_%s.jpg' % (i,epoch)))

                # recover the image to 255
                imfinal = im_to_numpy(torch.clamp(imfinal[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                target = im_to_numpy(torch.clamp(target[0] * 255, min=0.0, max=255.0)).astype(np.uint8)


                psnr = compare_psnr(imfinal, target, data_range=imfinal.max() - imfinal.min())
                ssim = compare_ssim(imfinal, target, multichannel=True)
                mse = compare_mse(imfinal, target)

                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim, inputs.size(0))
                mses.update(mse, inputs.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()


        print("Iter:%s,Losses:%s,PSNR:%.4f,SSIM:%.4f,MSE:%.4f" % (
        epoch, loss_relative.avg, psnres.avg, ssimes.avg, mses.avg))
        self.record('val/loss_relative', loss_relative.avg, epoch)

        self.record('val/PSNR', psnres.avg, epoch)
        self.record('val/SSIM', ssimes.avg, epoch)
        self.record('val/MSE', mses.avg, epoch)
        self.metric = psnres.avg

        self.model.train()

    def test(self, ):

        # switch to evaluate mode
        self.model.eval()
        print("==> testing VM model ")
        batch_time = AverageMeter()
        ssimes = AverageMeter()
        psnres = AverageMeter()
        ssimesx = AverageMeter()
        psnresx = AverageMeter()
        mses = AverageMeter()
        end = time.time()
        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):


                inputs = batches['composite_images'].to(self.device)
                target = batches['real_images'].to(self.device)
                mask = batches['mask'].to(self.device)
                fore = batches["fore_images"].to(self.device)  #: self.trans(thumb_foreground),
                foremask = batches["fore_mask"].to(self.device)  # "fore_mask":self.trans(thumb_mask),
                label = batches['label']
                # print(label.shape)
                label_oh = torch.zeros(self.args.test_batch, 5).scatter_(1, label.view(label.shape[0], 1), 1).to(
                    torch.float32).to(self.device)

                feeded = torch.cat([inputs, mask], dim=1).to(self.device)
                fore = torch.cat([fore, foremask], dim=1).to(self.device)
                outputs, _ = self.model(feeded, fore, label_oh, self.withLabel)
                outputs = outputs if type(outputs) == type(()) else [outputs]

                imfinal = outputs[0] * mask + inputs * (1 - mask)

                # if i % 200 == 0:
                #     # save the sample images
                #     ims = torch.cat([inputs, target, imfinal, mask.repeat(1, 3, 1, 1)], dim=3)
                #     torchvision.utils.save_image(ims, os.path.join(self.args.checkpoint, '%s_%s.jpg' % (i, epoch)))

                # recover the image to 255
                imfinal = im_to_numpy(torch.clamp(imfinal[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                target = im_to_numpy(torch.clamp(target[0] * 255, min=0.0, max=255.0)).astype(np.uint8)

                psnr = compare_psnr(imfinal, target, data_range=imfinal.max() - imfinal.min())
                ssim = compare_ssim(imfinal, target, multichannel=True)
                mse = compare_mse(imfinal, target)

                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim, inputs.size(0))
                mses.update(mse, inputs.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        print("%s:PSNR:%.5f(%.5f),MSE:%.5f, SSIM:%.5f(%.5f)" % (
            self.args.checkpoint, psnres.avg, psnresx.avg, mses.avg, ssimes.avg, ssimesx.avg))
        print("DONE.\n")
