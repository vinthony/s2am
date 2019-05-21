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


import json
from .BasicMachine import BasicMachine


class MaskedRASC(BasicMachine):
    def __init__(self,**kwargs):
        BasicMachine.__init__(self,**kwargs)

        if kwargs['pixelloss']:
            self.outputLoss, self.attentionLoss = nn.MSELoss(), nn.MSELoss()
        else:
            self.outputLoss, self.attentionLoss = nn.MSELoss(), nn.BCELoss()

        if self.args.gpu:
            self.outputLoss = self.outputLoss.cuda()
            self.attentionLoss = self.attentionLoss.cuda()


    def train(self,epoch):

        self.current_epoch = epoch

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        lossMask = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.train_loader))

        for i, (inputs, target) in enumerate(self.train_loader):

            if self.args.gpu:
                inputs = inputs.cuda()
                mask = inputs[:,3:4,:,:].cuda()
                target = target[0].cuda()
            else:
                target = target[0]
                mask = inputs[:,3:4,:,:]

            output,maskc1 = self.model(inputs)
            current_index = len(self.train_loader) * epoch + i

            if  i == 1 :
                self.writer.add_images('train/output',output,current_index)
                self.writer.add_images('train/target',target,current_index)
                self.writer.add_images('train/input',inputs[:,0:3,:,:],current_index)
                self.writer.add_images('train/mask',mask.repeat(1,3,1,1),current_index)
                self.writer.add_images('train/attention',maskc1.repeat(1,3,1,1),current_index)


            # here two choice: mseLoss or NLLLoss
            masked_loss  = 1e10 * self.attentionLoss(maskc1,mask)
            L2_loss = 1e10 * self.outputLoss(output,target)

            total_loss = L2_loss + masked_loss

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            losses.update(L2_loss.item(), inputs.size(0))
            lossMask.update(masked_loss.item(), inputs.size(0))


               # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss L2: {loss_label:.4f} | Loss Mask: {loss_mask:.4f}'.format(
                        batch=i + 1,
                        size=len(self.train_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss_label=losses.avg,
                        loss_mask=lossMask.avg
                        )
            bar.next()
        bar.finish()

        self.writer.add_scalar('train/loss_L2', losses.avg, epoch)
        self.writer.add_scalar('train/loss_Mask', lossMask.avg, epoch)



    def validate(self, epoch):

        self.current_epoch = epoch
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        lossMask = AverageMeter()
        psnres = AverageMeter()
        ssimes = AverageMeter()

       
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.val_loader))
        with torch.no_grad():
            for i, (inputs, target) in enumerate(self.val_loader):

                # measure data loading time
                if self.args.gpu:
                    inputs = inputs.cuda() # image and bbox
                    mask   = inputs[:,3:4,:,:].cuda()
                    target = target[0].cuda()


                output, maskc1  = self.model(inputs)

                loss = self.outputLoss(output, target)

                # here two choice: mseLoss or NLLLoss
                masked_loss  = 1e10 * self.attentionLoss(maskc1,mask)
                L2_loss = 1e10 * loss
                psnr = 10 * log10(1 / loss.item())       
                total_loss = L2_loss + masked_loss

                ssim = pytorch_ssim.ssim(output,target)

                losses.update(L2_loss.item(), inputs.size(0))
                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim, inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_L2: {loss_label:.4f} | Loss_Mask: {loss_mask:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}'.format(
                            batch=i + 1,
                            size=len(self.val_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss_label=losses.avg,
                            loss_mask=lossMask.avg,
                            psnr=psnres.avg,
                            ssim=ssimes.avg
                            )
                bar.next()
        bar.finish()
        
        self.writer.add_scalar('val/loss_L2', losses.avg, epoch)
        self.writer.add_scalar('val/lossMask', lossMask.avg, epoch)
        self.writer.add_scalar('val/PSNR', psnres.avg, epoch)
        self.writer.add_scalar('val/SSIM', ssimes.avg, epoch)

        if self.best_acc < psnres.avg:
            self.is_best = True
            self.best_acc = psnres.avg
