import torch
from progress.bar import Bar
import scripts.utils.pytorch_ssim as pytorch_ssim
import json
from .BasicMachine import BasicMachine

import scripts.models as models


class GANRASC(BasicMachine):
    def __init__(self,**kwargs):
        BasicMachine.__init__(self,**kwargs)

        self.model = models.__dict__[self.args.arch]()
        self.discriminator = models.__dict__[self.args.darch]()

        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_L1 = torch.nn.L1Loss()

        if self.args.gpu:    
            self.model.cuda()
            self.discriminator.cuda()
            self.criterion_GAN.cuda()
            self.criterion_L1.cuda()

        self.optimizer_G = torch.optim.Adam(self.model.parameters(), lr=args.lr,  betas=(0.9,0.999), weight_decay=args.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr,  betas=(0.9,0.999), weight_decay=args.weight_decay)


    def train(self,epoch):

        self.current_epoch = epoch

        batch_time = AverageMeter()
        data_time = AverageMeter()
        LoggerLossG = AverageMeter()
        LoggerLossGGAN = AverageMeter()
        LoggerLossGL1 = AverageMeter()

        LoggerLossD = AverageMeter()
        LoggerLossDreal = AverageMeter()
        LoggerLossDfake = AverageMeter()

       
        # switch to train mode
        self.model.train()
        discriminator.train()

        end = time.time()
        patch =  self.args.input_size//2**4 
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.train_loader))

        for i, (inputs, target) in enumerate(self.train_loader):

            current_index = len(self.train_loader) * epoch+i
            valid = torch.ones((inputs.size(0), patch, patch),requires_grad=False).cuda()
            fake =  torch.zeros((inputs.size(0), patch, patch),requires_grad=False).cuda()

            if args.gpu:
                inputs = inputs.cuda()
                target = target[:,0:3,:,:].cuda()
                valid.cuda()
                fake.cuda()

            # ---------------
            # Train model
            # --------------

            self.optimizer_G.zero_grad()
            fake_input = self.model(inputs)
            pred_fake = self.discriminator(fake_input, inputs[:,0:3,:,:])
            loss_GAN = self.criterion_GAN(pred_fake, valid)
            loss_pixel = self.criterion_L1(fake_input, target) # fake in
            loss_G = loss_GAN + 100 * loss_pixel
            loss_G.backward()
            self.optimizer_G.step()

            # --------------------
            #  Train Discriminator
            # --------------------
            self.optimizer_D.zero_grad()
            pred_real = self.discriminator(target, inputs[:,0:3,:,:])
            loss_real = self.criterion_GAN(pred_real, valid)
            pred_fake = self.discriminator(fake_input.detach(),inputs[:,0:3,:,:])
            loss_fake = self.criterion_GAN(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            self.optimizer_D.step()

            # ---------------------
            #        Logger
            # ---------------------

            LoggerLossGGAN.update(loss_GAN.item(), inputs.size(0))
            LoggerLossGL1.update(loss_pixel.item(), inputs.size(0))
            LoggerLossG.update(loss_G.item(), inputs.size(0))
            LoggerLossDfake.update(loss_real.item(), inputs.size(0))
            LoggerLossDreal.update(loss_fake.item(), inputs.size(0))
            LoggerLossD.update(loss_D.item(), inputs.size(0))

            
            # ---------------------
            #        Visualize
            # ---------------------

            if  i == 1 :
                self.writer.add_image('train/Goutput',fake_input,current_index)
                self.writer.add_image('train/target',target,current_index)
                self.writer.add_image('train/input',inputs[:,0:3,:,:],current_index)
                self.writer.add_image('train/mask',inputs[:,3:4,:,:],current_index)


             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss D: {loss_d:.4f} | Loss G: {loss_g:.4f} | Loss L1: {loss_l1:.6f}'.format(
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



    def validate(self, epoch):

        self.current_epoch = epoch
        batch_time = AverageMeter()
        data_time = AverageMeter()
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
                    inputs = inputs.cuda()
                    target = target[:,0:3,:,:].cuda()

                output = model(inputs)

                ## psnr and  ssim calculator.
                mse = self.criterion_GAN(output,target)
                psnr = 10 * log10(1 / mse.item()) 
                ssim = pytorch_ssim.ssim(output, target)      

                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim, inputs.size(0))

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

        if self.best_acc < psnres.avg:
            self.is_best = True
            self.best_acc = psnres.avg
