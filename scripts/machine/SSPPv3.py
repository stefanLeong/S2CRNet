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
from scripts.utils.losses import VGGLoss, l1_relative, is_dic, MaskWeightedMSE, set_requires_grad, GANLoss, fMSE
from scripts.models.patchNCE import PatchNCELoss
from scripts.utils.imgUtils import im_to_numpy, tensor2img
from scripts.models.unet import net_D, CF
import skimage.io
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from thop import profile, clever_format
import itertools

# import kornia.color as colors

class SSPP_LUT(BasicMachine):
    def __init__(self, **kwargs):
        BasicMachine.__init__(self, **kwargs)
        self.optimizers = []
        self.net_D = net_D(in_channels=3).to(self.device)
        self.optimizer_D = torch.optim.RMSprop(self.net_D.parameters(), lr=self.args.lr)
        # self.loss = Losses(self.args, self.device, self.norm, self.denorm)
        self.optimizer = torch.optim.AdamW(itertools.chain(self.model.parameters(), self.model.LUT0.parameters(),
                                                           self.model.LUT1.parameters(),self.model.LUT2.parameters()), lr=self.args.lr)

        self.optimizers.append(self.optimizer)
        self.optimizers.append(self.optimizer_D)
        self.set_requires_grad = set_requires_grad
        # self.model.set_optimizers(self.args.lr)
        # self.optimizers = self.model.optimizers
        # self.criterion_GAN = torch.nn.MSELoss()
        self.l1Loss = nn.L1Loss()
        self.stack = self.model.stack
        self.criterionGAN = GANLoss(gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0).to(
            self.device)  # MSEloss
        if self.args.withLabel == 'True':
            self.withLabel = True
        elif self.args.withLabel == 'False':
            self.withLabel = False
        else:
            print("Args--withLabel input error!")
        print(self.withLabel)
        # print(self.model)
    def backward_G(self, target, outputs, mask, stack):
        rL1_loss = sum([l1_relative(output, target, mask) for output in outputs])
        # rL1_loss = sum([self.l1Loss(output, target) for output in outputs])

        # discriminator loss
        # if stack:
        #     D_pred_1 = self.net_D(outputs[0])
        #     loss_G_GAN_1 = self.criterionGAN(D_pred_1, True)
        #     D_pred_2 = self.net_D(outputs[1])
        #     loss_G_GAN_2 = self.criterionGAN(D_pred_2, True)
        #
        #     loss_G_GAN = loss_G_GAN_1 + loss_G_GAN_2
        # else:
        D_pred = self.net_D(outputs[0])
        loss_G_GAN = self.criterionGAN(D_pred, True)
        # import pdb;
        # pdb.set_trace()

        loss_G = rL1_loss + 0.01 * loss_G_GAN
        loss_G.backward()
        return rL1_loss, loss_G_GAN

    def backward_D(self, target, pred, mask, stack):
        # discriminator loss
        # D_target = self.net_D(target.detach() * mask)
        D_target = self.net_D(target.detach())
        loss_D_target = self.criterionGAN(D_target, True)

        # if stack:
        #     D_pred_1 = self.net_D(pred[0].detach())
        #     loss_D_pred_1 = self.criterionGAN(D_pred_1, False)
        #     D_pred_2 = self.net_D(pred[1].detach())
        #     loss_D_pred_2 = self.criterionGAN(D_pred_2, False)
        #     loss_D_pred = loss_D_pred_1 + loss_D_pred_2
        # else:
        D_pred = self.net_D(pred[0].detach())
        loss_D_pred = self.criterionGAN(D_pred, False)

        loss_D = (loss_D_pred + loss_D_target) * 1.0
        loss_D.backward()

        return loss_D_pred, loss_D_target

    def train(self, epoch):

        self.current_epoch = epoch

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_relative = AverageMeter()
        loss_G_gan = AverageMeter()
        loss_D_t = AverageMeter()
        loss_D_p = AverageMeter()
        loss_G_total = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.train_loader))

        for i, batches in enumerate(self.train_loader):

            current_index = len(self.train_loader) * epoch + i

            inputs = batches['composite_images'].to(self.device)
            target = batches['real_images'].to(self.device)
            mask = batches['mask'].to(self.device)
            fore = batches["fore_images"].to(self.device)  #: self.trans(thumb_foreground),
            foremask = batches["fore_mask"].to(self.device)  # "fore_mask":self.trans(thumb_mask)
            label = batches['label']

            label_oh = torch.zeros(self.args.train_batch, 5).scatter_(1, label.view(label.shape[0], 1), 1).to(
                torch.float32).to(self.device)

            feeded = torch.cat([inputs, mask], dim=1).to(self.device)
            fore = torch.cat([fore, foremask], dim=1).to(self.device)

            if self.withLabel:
                label_oh = label_oh
            else:
                label_oh = None
            outputs, param1, param2 = self.model(feeded, feeded, fore, label_oh, self.withLabel)

            outputs = outputs if type(outputs) == type(()) else [outputs]
            # print(len(outputs))
            # imfinal = outputs[0] * mask + inputs * (1 - mask)

            # update G
            self.set_requires_grad(self.net_D, False)  # D requires no gradients when optimizing G
            self.optimizer.zero_grad()
            # outputs: reconstructed_real, real_idt, ori_idt, mask_out
            rL1_loss, loss_G_GAN = self.backward_G(target, outputs, mask, self.stack)
            loss_G = rL1_loss + 0.01 * loss_G_GAN
            self.optimizer.step()

            # update D
            self.set_requires_grad(self.net_D, True)  # enable bp for D
            self.optimizer_D.zero_grad()
            loss_D_pred, loss_D_target = self.backward_D(target, outputs, mask, stack=self.stack)
            self.optimizer_D.step()
            # total_loss = sum([l1_relative(output, target, mask) for output in outputs])
            # # compute gradient and do SGD step
            # total_loss.backward()
            # self.optimizer.step()

            # measure accuracy and record loss

            loss_relative.update(rL1_loss.item(), inputs.size(0))
            loss_G_gan.update(loss_G_GAN.item(), inputs.size(0))
            loss_G_total.update(loss_G.item(), inputs.size(0))
            loss_D_p.update(loss_D_pred.item(), inputs.size(0))
            loss_D_t.update(loss_D_target.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            suffix = "({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | loss Real: {loss_real:.5f} " \
                     "| loss G_GAN: {loss_G_gan:.5f} | loss G_TOTAL: {loss_G:.5f} | loss D_pred: {loss_D_pred:.5f} |loss D_targ: {loss_D_target:.5f} ".format(
                batch=i + 1,
                size=len(self.train_loader),
                data=data_time.val,
                bt=batch_time.val,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss_real=loss_relative.avg,
                loss_D_pred=loss_D_p.avg,
                loss_D_target=loss_D_t.avg,
                loss_G_gan=loss_G_gan.avg,
                loss_G=loss_G_total.avg
            )
            if current_index % 1000 == 0:
                print(suffix)

        self.record('train/loss_relative', loss_relative.avg, epoch)
        self.record('train/loss_D_pred', loss_D_p.avg, epoch)
        self.record('train/loss_D_target', loss_D_t.avg, epoch)
        self.record('train/loss_G_TOTAL', loss_G_total.avg, epoch)
        self.record('train/loss_G_GAN', loss_G_gan.avg, epoch)

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
                fore = batches["fore_images"].to(self.device)  #: self.trans(thumb_foreground),
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
                outputs, param1, param2 = self.model(feeded, feeded, fore, label_oh, self.withLabel)
                outputs = outputs if type(outputs) == type(()) else [outputs]

                imfinal = outputs[0] * mask + inputs * (1 - mask)

                if i % 200 == 0:
                    # save the sample images
                    ims = torch.cat([inputs, target, imfinal, mask.repeat(1, 3, 1, 1)], dim=3)
                    torchvision.utils.save_image(ims, os.path.join(self.args.checkpoint, '%s_%s.jpg' % (i, epoch)))

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
                ori_img = batches["ori_img"].to(self.device)
                # print(label.shape)
                label_oh = torch.zeros(self.args.test_batch, 5).scatter_(1, label.view(label.shape[0], 1), 1).to(
                    torch.float32).to(self.device)
                if self.withLabel:
                    label_oh = label_oh
                else:
                    label_oh = None

                start = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                feeded = torch.cat([inputs, mask], dim=1).to(self.device)
                fore = torch.cat([fore, foremask], dim=1).to(self.device)
                start.record()
                outputs, param1, param2 = self.model(ori_img, feeded, fore, label_oh, self.withLabel)
                end_time.record()
                torch.cuda.synchronize()
                macs, params = profile(self.model, inputs=(ori_img, feeded, fore, label_oh, self.withLabel))
                macs, params = clever_format([macs, params], "%.3f")
                print(macs, params, start.elapsed_time(end_time))

                outputs = outputs if type(outputs) == type(()) else [outputs]

                imfinal = outputs[0] * mask + inputs * (1 - mask)
                # print(imfinal.shape)
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

    def test_hr(self, ):

        # switch to evaluate mode
        self.model.eval()
        print("==> testing VM model ")
        batch_time = AverageMeter()
        ssimes = AverageMeter()
        psnres = AverageMeter()
        mse_5 = AverageMeter()
        mse_15 = AverageMeter()
        mse_o15 = AverageMeter()
        fmse_5 = AverageMeter()
        fmse_15 = AverageMeter()
        fmse_o15 = AverageMeter()
        mses = AverageMeter()
        fmses = AverageMeter()
        hr_mses = AverageMeter()
        hr_psnrs = AverageMeter()
        processing_times = AverageMeter()
        end = time.time()

        macses = AverageMeter()
        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):


                inputs = batches['composite_images'].to(self.device)
                target = batches['real_images'].to(self.device)
                mask = batches['mask'].to(self.device)
                fore = batches["fore_images"].to(self.device)  #: self.trans(thumb_foreground),
                foremask = batches["fore_mask"].to(self.device)  # "fore_mask":self.trans(thumb_mask),
                # ori_img = batches["ori_img"].to(self.device)
                # ori_mask = batches["ori_mask"].to(self.device)
                # ori_tar = batches["ori_tar"].to(self.device)
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
                #input high-resolution images
                    # testing processing time
                start = time.time()
                outputs, param1, param2 = self.model(feeded, feeded, fore, label_oh, self.withLabel)
                # print(outputs[0].shape)
                end = time.time()
                # print('%s' % (i), end-start)
                    # testing computational cost

                # macs, params = profile(self.model, inputs=(feeded, feeded, fore, label_oh, self.withLabel))

                outputs = outputs if type(outputs) == type(()) else [outputs]
                #high-resolution
                # print(ori_img, inputs)
                # hr_outputs = CF(ori_img, param1.view(ori_img.size(0), ori_img.size(1), 1, 1, 64), 64)
                # hr_outputs = CF(hr_outputs, param2.view(ori_img.size(0), ori_img.size(1), 1, 1, 64), 64)
                # hr_final = hr_outputs * ori_mask + ori_img * (1-ori_mask)
                # img_show = hr_final.squeeze(0).cpu().permute(1, 2, 0).numpy()

                # imfinal = outputs[0] * ori_mask + ori_tar * (1 - ori_mask)
                imfinal = outputs[0] * mask + target * (1 - mask)

                # if i % 100 == 0:
                #     # save the sample images
                #     ims = torch.cat([inputs, target, imfinal, mask.repeat(1, 3, 1, 1)], dim=3)
                #     torchvision.utils.save_image(ims, os.path.join(self.args.checkpoint, '%s_.jpg' % (i)))

                # recover the image to 255
                imfinal = im_to_numpy(torch.clamp(imfinal[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                # ori_tar = im_to_numpy(torch.clamp(ori_tar[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                target = im_to_numpy(torch.clamp(target[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                # fr_final = im_to_numpy(torch.clamp(hr_final[0] * 255, min=0.0, max=255.0)).astype(np.uint8)
                # fr_target = im_to_numpy(torch.clamp(ori_tar[0] * 255, min=0.0, max=255.0)).astype(np.uint8)

                # psnr = compare_psnr(imfinal, ori_tar, data_range=imfinal.max() - imfinal.min())
                # ssim = compare_ssim(imfinal, ori_tar, multichannel=True)
                # mse = compare_mse(imfinal, ori_tar)
                # print(imfinal.shape, target.shape)
                psnr = compare_psnr(imfinal, target, data_range=imfinal.max() - imfinal.min())
                ssim = compare_ssim(imfinal, target, multichannel=True)
                mse = compare_mse(imfinal, target)
                # hr_psnr = compare_psnr(fr_final, fr_target, data_range=fr_final.max() - fr_final.min())
                # hr_mse = compare_mse(fr_final, fr_target)
                # print(type(imfinal), type(target), type(mask.cpu().numpy()))
                # print(imfinal.shape, mask.shape)
                mask = mask.cpu().squeeze(0).permute(1, 2, 0)

                fmse = compare_mse(imfinal * mask.numpy(), target * mask.numpy()) * 256 * 256 / (torch.count_nonzero(mask))
                # import pdb;
                # pdb.set_trace()

                # processing_times.update(end-start, inputs.size(0))
                # macses.update(macs, inputs.size(0))
                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim, inputs.size(0))
                mses.update(mse, inputs.size(0))
                fmses.update(fmse, inputs.size(0))
                #
                # hr_psnrs.update(hr_psnr, inputs.size(0))
                # hr_mses.update(hr_mse, inputs.size(0))

                ratio = torch.count_nonzero(mask) / torch.numel(mask)
                if ratio <= 0.05:
                    fmse_5.update(fmse, inputs.size(0))
                    mse_5.update(mse, inputs.size(0))
                elif 0.05 < ratio <= 0.15:
                    fmse_15.update(fmse, inputs.size(0))
                    mse_15.update(mse, inputs.size(0))
                else:
                    fmse_o15.update(fmse, inputs.size(0))
                    mse_o15.update(mse, inputs.size(0))
                # # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()
                torch.cuda.Event(enable_timing=False)
                torch.cuda.Event(enable_timing=False)

        print("%s:PSNR:%.5f,MSE:%.5f, SSIM:%.5f, fMSE:%.5f" % (
            self.args.checkpoint, psnres.avg, mses.avg, ssimes.avg, fmses.avg))

        # Avg_macs, params = clever_format([macs, params], "%.3f")
        # print('hr_psnr/mse:', Avg_macs, params )
        print(
            "%s:MSE0_5:%.5f,fMSE0_5:%.5f,MSE5_15:%.5f, fMSE5_15:%.5f,MSE15_100:%.5f,fMSE15_100:%.5f, MSE:%.5f,fMSE:%.5f" % (
                self.args.checkpoint, mse_5.avg, fmse_5.avg, mse_15.avg, fmse_15.avg, mse_o15.avg, fmse_o15.avg,
                mses.avg, fmses.avg))

        print("DONE.\n")
