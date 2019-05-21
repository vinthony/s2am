import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from progress.bar import Bar
import json
from tensorboardX import SummaryWriter
from scripts.utils.evaluation import accuracy, AverageMeter, final_preds
from scripts.utils.osutils import mkdir_p, isfile, isdir, join
import scripts.utils.pytorch_ssim as pytorch_ssim
import torch.optim
import time
import scripts.models as archs
from math import log10

from .BasicMachine import BasicMachine


def deNorm(output):
    # [-1,1] -> [0,1]
    return (output + 1)/2


class MMaskedRASCGANplus(BasicMachine):
    def __init__(self,**kwargs):
        BasicMachine.__init__(self,**kwargs)

        self.attentionLoss8s= kwargs['pixelloss']()
        self.attentionLoss4s= kwargs['pixelloss']()
        self.attentionLoss2s= kwargs['pixelloss']()

        self.discriminator = archs.__dict__[self.args.darch]()

        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_L1 = torch.nn.L1Loss()

        if self.args.gpu:    
            self.model.cuda()
            self.discriminator.cuda()
            self.criterion_GAN.cuda()
            self.criterion_L1.cuda()
            self.attentionLoss8s = self.attentionLoss8s.cuda()
            self.attentionLoss4s = self.attentionLoss4s.cuda()
            self.attentionLoss2s = self.attentionLoss2s.cuda()

        self.optimizer_G = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,  betas=(0.5,0.999), weight_decay=self.args.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr,  betas=(0.5,0.999), weight_decay=self.args.weight_decay)

        if 'patchgan' in self.args.darch:
            self.patch = self.args.input_size//2**4
        elif 'pixelgan' in self.args.darch:
            self.patch = self.args.input_size
        elif 'compared' in self.args.darch:
            self.patch = self.args.input_size//2**5
        else:
            self.patch = 1 


    def train(self,epoch):

        self.current_epoch = epoch

        if self.args.freeze and epoch > 10:
                self.model.freeze_weighting_of_rasc()
                self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,  betas=(0.5,0.999), weight_decay=self.args.weight_decay)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        LoggerLossG = AverageMeter()
        LoggerLossGGAN = AverageMeter()
        LoggerLossGL1 = AverageMeter()

        LoggerLossD = AverageMeter()
        LoggerLossDreal = AverageMeter()
        LoggerLossDfake = AverageMeter()

        lossMask8s = AverageMeter()
        lossMask4s = AverageMeter()
        lossMask2s = AverageMeter()

       
        # switch to train mode
        self.model.train()
        self.discriminator.train()

        end = time.time()
        
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.train_loader))

        for i, (inputs, target) in enumerate(self.train_loader):

            input_image,mask,m2s,m4s,m8s = inputs

            current_index = len(self.train_loader) * epoch+i
            valid = torch.ones((input_image.size(0), self.patch, self.patch),requires_grad=False).cuda()
            fake =  torch.zeros((input_image.size(0), self.patch, self.patch),requires_grad=False).cuda()
 
            reverse_mask = 1 - mask

            if self.args.gpu:
                input_image = input_image.cuda()
                mask = mask.cuda()
                m2s = m2s.cuda()
                m4s = m4s.cuda()
                m8s = m8s.cuda()
                reverse_mask = reverse_mask.cuda()
                target = target.cuda()
                valid.cuda()
                fake.cuda()
          
            # ---------------
            # Train model
            # --------------

            self.optimizer_G.zero_grad()
            fake_input, mask8s,mask4s,mask2s = self.model(torch.cat((input_image,mask),1))

            pred_fake = self.discriminator(fake_input, input_image)
            loss_GAN = self.criterion_GAN(pred_fake, valid)
            loss_pixel = self.criterion_L1(fake_input, target) # fake in
            # here two choice: mseLoss or NLLLoss
            masked_loss8s  =  self.attentionLoss8s(mask8s,m8s)
            masked_loss4s =  self.attentionLoss4s(mask4s,m4s)
            masked_loss2s  = self.attentionLoss2s(mask2s,m2s)
            loss_G = loss_GAN + 100 * loss_pixel + 90 * masked_loss8s + 90 * masked_loss4s + 90 * masked_loss2s

            loss_G.backward()
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            pred_real = self.discriminator(target, input_image)
            loss_real = self.criterion_GAN(pred_real, valid)
            pred_fake = self.discriminator(fake_input.detach(),input_image)
            loss_fake = self.criterion_GAN(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            self.optimizer_D.step()

            # ---------------------
            #        Logger
            # ---------------------

            LoggerLossGGAN.update(loss_GAN.item(), input_image.size(0))
            LoggerLossGL1.update(loss_pixel.item(), input_image.size(0))
            LoggerLossG.update(loss_G.item(), input_image.size(0))
            LoggerLossDfake.update(loss_real.item(), input_image.size(0))
            LoggerLossDreal.update(loss_fake.item(), input_image.size(0))
            LoggerLossD.update(loss_D.item(), input_image.size(0))
            lossMask8s.update(masked_loss8s.item(), input_image.size(0))
            lossMask4s.update(masked_loss4s.item(), input_image.size(0))
            lossMask2s.update(masked_loss2s.item(), input_image.size(0))

            # ---------------------
            #        Visualize
            # ---------------------

            if  i == 1 :
                self.writer.add_images('train/Goutput',deNorm(fake_input),current_index)
                self.writer.add_images('train/target',deNorm(target),current_index)
                self.writer.add_images('train/input',deNorm(input_image),current_index)
                self.writer.add_images('train/mask',mask.repeat((1,3,1,1)),current_index)
                self.writer.add_images('train/attention2s',mask2s.repeat(1,3,1,1),current_index)
                self.writer.add_images('train/attention4s',mask4s.repeat(1,3,1,1),current_index)
                self.writer.add_images('train/attention8s',mask8s.repeat(1,3,1,1),current_index)

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss D: {loss_d:.4f} | Loss G: {loss_g:.4f} | Loss L1: {loss_l1:.6f} '.format(
                        batch=i + 1,
                        size=len(self.train_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss_d=LoggerLossD.avg,
                        loss_g=LoggerLossGGAN.avg,
                        loss_l1=LoggerLossGL1.avg
                        )
            bar.next()

        bar.finish()
        self.writer.add_scalar('train/loss/GAN', LoggerLossGGAN.avg, epoch)
        self.writer.add_scalar('train/loss/D', LoggerLossD.avg, epoch)
        self.writer.add_scalar('train/loss/L1', LoggerLossGL1.avg, epoch)
        self.writer.add_scalar('train/loss/G', LoggerLossG.avg, epoch)
        self.writer.add_scalar('train/loss/Dreal', LoggerLossDreal.avg, epoch)
        self.writer.add_scalar('train/loss/Dfake', LoggerLossDfake.avg, epoch)

        self.writer.add_scalar('train/loss_Mask8s', lossMask8s.avg, epoch)
        self.writer.add_scalar('train/loss_Mask4s', lossMask4s.avg, epoch)
        self.writer.add_scalar('train/loss_Mask2s', lossMask2s.avg, epoch)


    def validate(self, epoch):

        self.current_epoch = epoch
        batch_time = AverageMeter()
        data_time = AverageMeter()
        psnres = AverageMeter()
        ssimes = AverageMeter()
        lossMask8s = AverageMeter()
        lossMask4s = AverageMeter()
        lossMask2s = AverageMeter()
       
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.val_loader))
        
        with torch.no_grad():
            for i, (inputs, target) in enumerate(self.val_loader):
                
                input_image,mask,m2s,m4s,m8s = inputs

                current_index = len(self.train_loader) * epoch+i
                valid = torch.ones((input_image.size(0), self.patch, self.patch),requires_grad=False).cuda()
                fake =  torch.zeros((input_image.size(0), self.patch, self.patch),requires_grad=False).cuda()
     
                reverse_mask = 1 - mask

                if self.args.gpu:
                    input_image = input_image.cuda()
                    mask = mask.cuda()
                    m2s = m2s.cuda()
                    m4s = m4s.cuda()
                    m8s = m8s.cuda()
                    reverse_mask = reverse_mask.cuda()
                    target = target.cuda()
                    valid.cuda()
                    fake.cuda()


                # 32,64,128
                output,mask8s,mask4s,mask2s = self.model(torch.cat((input_image,mask),1))

                output = deNorm(output)
                target = deNorm(target)

                masked_loss8s  = self.attentionLoss8s(mask8s,m8s)
                masked_loss4s = self.attentionLoss4s(mask4s,m4s)
                masked_loss2s  = self.attentionLoss2s(mask2s,m2s)

                ## psnr and  ssim calculator.
                mse = self.criterion_GAN(output,target)
                psnr = 10 * log10(1 / mse.item()) 
                ssim = pytorch_ssim.ssim(output, target)      

                psnres.update(psnr, input_image.size(0))
                ssimes.update(ssim, input_image.size(0))
                lossMask8s.update(masked_loss8s.item(), input_image.size(0))
                lossMask4s.update(masked_loss4s.item(), input_image.size(0))
                lossMask2s.update(masked_loss2s.item(), input_image.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | SSIM: {ssim:.4f} | PSNR: {psnr:.4f}'.format(
                            batch=i + 1,
                            size=len(self.val_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            ssim=ssimes.avg,
                            psnr=psnres.avg
                            )
                bar.next()
        bar.finish()
        
        self.writer.add_scalar('val/SSIM', ssimes.avg, epoch)
        self.writer.add_scalar('val/PSNR', psnres.avg, epoch)
        self.writer.add_scalar('train/loss_Mask8s', lossMask8s.avg, epoch)
        self.writer.add_scalar('train/loss_Mask4s', lossMask4s.avg, epoch)
        self.writer.add_scalar('train/loss_Mask2s', lossMask2s.avg, epoch)

        self.metric = psnres.avg
