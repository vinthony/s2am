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

class SemiRASC(BasicMachine):
    def __init__(self, datasets = None, models = None, args = None, **kwargs):
        BasicMachine.__init__(self,**kwargs)

    def train(self,epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()

        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.train_loader))

        for i, (inputs, target) in enumerate(self.train_loader):
            # measure data loading time

            if self.args.gpu:
                inputs = inputs.cuda()
                mask = target[:,3:4,:,:].cuda()
                bbox = inputs[:,3:4,:,:].cuda()
                target = target[0].cuda()
            else:
                target = target[0]
                mask = target[:,3:4,:,:]
                bbox = inputs[:,3:4,:,:]

            
            output = self.model(inputs)

            if  i == 1 :
                current_index = len(self.train_loader) * epoch+i
                self.writer.add_images('train/output',output,current_index)
                self.writer.add_images('train/target',target,current_index)
                self.writer.add_images('train/input',inputs[:,0:3,:,:],current_index)
                self.writer.add_images('train/mask',mask.repeat(1,3,1,1),current_index)
                self.writer.add_images('train/bbox',bbox.repeat(1,3,1,1),current_index)


            L2_loss = 1e10 * self.loss(output,target)
            total_loss = L2_loss 

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            losses.update(L2_loss.item(), inputs.size(0))

               # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss L2: {loss_label:.4f}'.format(
                        batch=i + 1,
                        size=len(self.train_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss_label=losses.avg
                        )
            bar.next()
        bar.finish()
        self.writer.add_scalar('train/loss_L2', losses.avg, epoch)


    def validate(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        ssimes = AverageMeter()
        psnres = AverageMeter()
       
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        bar = Bar('Processing', max=len(self.val_loader))
        with torch.no_grad():
            for i, (inputs, target) in enumerate(self.val_loader):

                # measure data loading time
                if self.args.gpu:
                    inputs = inputs.cuda()
                    mask = target[:,3:4,:,:].cuda()
                    bbox = inputs[:,3:4,:,:].cuda()
                    target = target[0].cuda()
                else:
                    target = target[0]
                    mask = target[:,3:4,:,:]
                    bbox = inputs[:,3:4,:,:]


                output = self.model(inputs)
                mse = self.loss(output, target)

                L2_loss = 1e10 * mse
                psnr = 10 * log10(1 / mse.item())   
                ssim = pytorch_ssim.ssim(output, target)    

                losses.update(L2_loss.item(), inputs.size(0))
                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim, inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_L2: {loss_label:.4f} | SSIM: {ssim:.4f} | PSNR: {psnr:.4f}'.format(
                            batch=i + 1,
                            size=len(self.val_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss_label=losses.avg,
                            psnr=psnres.avg,
                            ssim=ssimes.avg,
                            )
                bar.next()
        bar.finish()
        
        self.writer.add_scalar('val/loss_L2', losses.avg, epoch)
        self.writer.add_scalar('val/PSNR', psnres.avg, epoch)
        self.writer.add_scalar('val/SSIM', ssimes.avg, epoch)


