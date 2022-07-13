import torch
import torch.nn as nn
from progress.bar import Bar
from tqdm import tqdm
import pytorch_ssim
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
from scripts.utils.losses import VGGLoss, l1_relative, is_dic, MaskWeightedMSE
from scripts.models.split_single import SplitwithSingle
from scripts.utils.imgUtils import im_to_numpy, tensor2img
from scripts.models import split_cappV3
import skimage.io
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse


class SplitFinetune(BasicMachine):
    def __init__(self, **kwargs):
        BasicMachine.__init__(self, **kwargs)
        self.optimizers = []
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.optimizers.append(self.optimizer)

        self.xmodel = SplitwithSingle(shared_depth=3, blocks=3, depth=5, long_skip=True).cuda()
        self.xmodel.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])

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
        self.xmodel.eval()
        
        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.train_loader))

        for i, batches in enumerate(self.train_loader):

            current_index = len(self.train_loader) * epoch + i

            inputs = batches['composite_images'].to(self.device)
            target = batches['real_images'].to(self.device)
            mask = batches['mask'].to(self.device)
            feeded = torch.cat([inputs, mask], dim=1)
            feeded = feeded.to(self.device)

            self.optimizer.zero_grad()

            with torch.no_grad():
                result = self.xmodel(feeded)
                imstage1 = result[0] * mask + inputs * (1 - mask)
            
            outputs = self.model(torch.cat([imstage1,mask],dim=1))

            total_loss = l1_relative(outputs, target, mask) # sum([l1_relative(output,target,mask) for output in outputs ]) #F.l1_loss(outputs,target) 

            # compute gradient and do SGD step
            total_loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            if self.args.L1_pixel_loss > 0.0:
                loss_relative.update(pixel_loss.item(), inputs.size(0))
            loss_real.update(total_loss.item(), inputs.size(0))
            loss_total.update(total_loss.item(), inputs.size(0))

            if self.args.lambda_NCE > 0:
                loss_patch.update(patch_loss.item(), inputs.size(0))

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
            if current_index % 500 == 0:
                print(suffix)

            if self.args.freq > 0 and current_index % self.args.freq == 0:
                self.validate(current_index)
                self.flush()
                self.save_checkpoint()

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
        self.xmodel.eval()

        end = time.time()
        # bar = Bar('Processing {} '.format(self.args.arch), max=len(self.val_loader))
        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):

                current_index = len(self.val_loader) * epoch + i

                inputs = batches['composite_images'].to(self.device)
                target = batches['real_images'].to(self.device)
                mask = batches['mask'].to(self.device)

                feeded = torch.cat([inputs, mask], dim=1)
                feeded = feeded.to(self.device)
                
                result = self.xmodel(feeded)
                imstage1 = result[0] * mask + inputs * (1 - mask)
                
                outputs = self.model(torch.cat([imstage1,mask],dim=1))

                imfinal = outputs  #[0]
                
                imfinal = imfinal * mask + inputs * (1 - mask)

                if i % 200 == 0:
                    # save the sample images
                    ims = torch.cat([inputs, target, imfinal, mask.repeat(1, 3, 1, 1)], dim=3)
                    torchvision.utils.save_image(ims, os.path.join(self.args.checkpoint, '%s_%s.jpg' % (epoch,i)))

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
        ssimes = AverageMeter()
        psnres = AverageMeter()
        ssimesx = AverageMeter()
        psnresx = AverageMeter()

        with torch.no_grad():
            for i, batches in enumerate(tqdm(self.val_loader)):
                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)
                mask = batches['mask'].to(self.device)

                feeded = torch.cat([inputs, mask], dim=1)
                feeded = feeded.to(self.device)
                outputs = self.model(feeded)

                # im_fake, im_real = outputs
                reconv_real = outputs
                reconv_real = reconv_real[0] if is_dic(reconv_real) else reconv_real
                imfinal = reconv_real * mask + inputs * (1 - mask)

                psnrx = 10 * log10(1 / F.mse_loss(imfinal, target).item())
                ssimx = pytorch_ssim.ssim(imfinal, target)

                # recover the image to 255
                imfinal = im_to_numpy(torch.clamp(imfinal * 255, min=0.0, max=255.0)).astype(np.uint8)
                target = im_to_numpy(torch.clamp(target * 255, min=0.0, max=255.0)).astype(np.uint8)

                skimage.io.imsave('%s/%s' % (self.args.checkpoint, batches['name'][0]), imfinal)

                psnr = compare_psnr(target, imfinal)
                ssim = compare_ssim(target, imfinal, multichannel=True)

                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim, inputs.size(0))
                psnresx.update(psnrx, inputs.size(0))
                ssimesx.update(ssimx, inputs.size(0))

        print("%s:PSNR:%.5f(%.5f),SSIM:%.5f(%.5f)" % (
            self.args.checkpoint, psnres.avg, psnresx.avg, ssimes.avg, ssimesx.avg))
        print("DONE.\n")

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)