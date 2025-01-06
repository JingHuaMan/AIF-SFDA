'''
    This is an implementation of the DAMAN architecture in Fig.2. The network consists of
    1. Foreground prediction network
    2. Background prediction network
    3. Edge prediction network
    4. Image prediction network
    CORAL loss is used for domain adaptation at the intermediate levek
'''
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from others.other_components.daman.daman_networks.daman_core import UNet, EdgeNet, Decoder
from others.other_components.daman.daman_util.daman_losses import DiceLoss, MSELoss, CrossEntropyLoss
import time


class DaManNet(nn.Module):
    def __init__(self, savepath='/code/logs/', model_pretrained=None, save_intervl=10, in_channel=1, out_channel=1):
        super().__init__()
        self.model_pretrained = model_pretrained
        self.save_intervl = save_intervl

        self.active_elu = nn.ELU()
        self.active_relu = nn.ReLU()
        self.active_tanh = nn.Tanh()
        self.active_sigmoid = nn.Sigmoid()

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)

        self.conv_2_16 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv_16_32 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_16_1 = nn.Conv2d(in_channels=16, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

        self.deconv_32_16 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=2,
                                               output_padding=1)
        self.deconv_16_1 = nn.ConvTranspose2d(in_channels=16, out_channels=out_channel, kernel_size=3, padding=1, stride=2,
                                              output_padding=1)
        self.deconv_same_48_32 = nn.ConvTranspose2d(in_channels=48, out_channels=32, kernel_size=3, padding=1, stride=1)

        self.EdgNet = EdgeNet()
        self.UNet = UNet(in_channel=in_channel)
        self.RecNet = Decoder(out_channel=in_channel)
        self.RecBg = Decoder(out_channel=out_channel, do_final_relu=False)

        self.cum_loss = 0.
        self.save_model_path = savepath

    def forward(self, x):
        """
        The DAMAN architecture backbone is defined here
        """
        out_unet_512x512x16, mid_512x512x16, mid_256x256x32, _, mid_64x64x128 = self.UNet(x)

        # Foreground prediction
        out_mask_512x512x1 = self.conv_16_1(out_unet_512x512x16)
        y_fg = self.active_sigmoid(out_mask_512x512x1)

        #  Background Prediction
        y_bg = self.RecBg(mid_64x64x128)
        y_bg = self.active_sigmoid(y_bg)

        # Edge Prediction
        in1 = mid_512x512x16  # A very early layer in the base UNet
        in2 = out_unet_512x512x16  # The penultimate layer in the base UNet
        in12 = torch.cat((in1, in2), dim=1)  # Concatenate them
        in3 = self.deconv_32_16(mid_256x256x32)  # A middle layer of the base Unet
        edg_in_512x512x48 = torch.cat((in12, in3), dim=1)
        edg_in_512x512x32 = self.deconv_same_48_32(edg_in_512x512x48)
        edg_in = self.active_relu(edg_in_512x512x32)
        out_edg_512x512x1 = self.EdgNet(edg_in)
        y_edg = self.active_sigmoid(out_edg_512x512x1)

        # Image Reconstruction
        y_rec = self.RecNet(mid_64x64x128)

        # CORAL subspace feature
        mid_fmap = mid_64x64x128

        return y_fg, y_bg, y_edg, y_rec, mid_fmap

    def adaptive_diceloss(self, fg, bg, edg, y_fg, y_bg, y_edg):
        """
        Adaptive Dice Loss term from Eq. 12, using min-max
        """
        loss_fn = DiceLoss()
        E1 = loss_fn(y_fg, fg)
        E2 = loss_fn(1. - y_bg, 1. - bg)
        E3 = loss_fn(y_edg, edg)
        deno = np.sqrt((E1.item()) ** 2 + (E2.item()) ** 2 + (E3.item()) ** 2 + 1e-30)
        mu1, mu2, mu3 = E1.item() / deno, E2.item() / deno, E3.item() / deno
        loss = mu1 * E1 + mu2 * E2 + mu3 * E3
        return loss

    def adaptive_mseloss(self, fg, bg, edg, y_fg, y_bg, y_edg):
        """
        Adaptive MSE Loss term from Eq. 12, using min-max
        """
        loss_fn = MSELoss()
        E1 = loss_fn(y_fg, fg)
        E2 = loss_fn(1. - y_bg, 1. - bg)
        E3 = loss_fn(y_edg, edg)
        deno = np.sqrt((E1.item()) ** 2 + (E2.item()) ** 2 + (E3.item()) ** 2 + 1e-30)
        mu1, mu2, mu3 = E1.item() / deno, E2.item() / deno, E3.item() / deno
        loss = mu1 * E1 + mu2 * E2 + mu3 * E3
        return loss

    def domainadapt_loss(self, image_src, f_src, y_rec, image_tgt, lambda_coral=1.):
        # Coral Loss + Y-net prediction Loss
        src_recon_loss = MSELoss()(image_src, y_rec)
        total_loss_coral = 0
        tgt_recon_loss = 0
        if image_tgt is not None:
            _, _, _, rec_tgt, f_tgt = self(image_tgt)
            sz = f_src.size()  # [n_batch, n_feature, 32, 32]
            n_tgt = f_tgt.shape[0]
            n_src = sz[0]
            npts = min(n_src, n_tgt)  # Sometimes source and target batches may have different number of entries
            for s in range(sz[1]):
                loss_coral = coral(f_src[0:npts, s, :, :].view(npts, sz[2] * sz[3]),
                                   f_tgt[0:npts, s, :, :].view(npts, sz[2] * sz[3])
                                   )
                total_loss_coral = total_loss_coral + lambda_coral * loss_coral
            tgt_recon_loss = MSELoss()(image_tgt, rec_tgt)
        else:
            total_loss_coral = torch.tensor(total_loss_coral)
            tgt_recon_loss = torch.tensor(tgt_recon_loss)

        return src_recon_loss, tgt_recon_loss, total_loss_coral

    def train_net(self, train_loader_src, train_loader_tgt, num_epochs, batch_sz, optimizer, scheduler, lr=5e-4,
                  lambda_coral=1e-2):
        model = self
        start_epoch = 0
        cum_loss = 0.
        for epoch in range(start_epoch, num_epochs):
            time_start = time.perf_counter()
            epoch_loss, loss_tensor = self.train_epoch(train_loader_src, train_loader_tgt, batch_sz, lambda_coral,
                                                       optimizer, scheduler, epoch)
            time_elapsed = (time.perf_counter() - time_start)
            print("Time/epoch=%5.2f" % time_elapsed)
            cum_loss += epoch_loss
            if (epoch + 1) % self.save_intervl == 0 or epoch == num_epochs - 1:
                model_save_name = self.save_model_path + "DAMAN-" + str(epoch + 1) + ".pt"
                torch.save(model.state_dict(), model_save_name)
                print("saving model")

    def input_from_loaders(self, datapoint, list_tar, num_tar, batch):
        datapoint['image'] = datapoint['image'].type(
            torch.FloatTensor)  # typecasting to FloatTensor as it is compatible with CUDA
        datapoint['masks'] = datapoint['masks'].type(torch.FloatTensor)
        datapoint['backgrnd'] = datapoint['backgrnd'].type(torch.FloatTensor)
        datapoint['edgeweight'] = datapoint['edgeweight'].type(torch.FloatTensor)
        if torch.cuda.is_available():  # If GPU available Converting a Torch Tensor to Autograd Variable
            image_src = Variable(datapoint['image'].cuda())
            if (batch < num_tar):
                image_tgt = Variable(list_tar[batch][1]['image'].type(torch.FloatTensor).cuda())
            else:
                image_tgt = None
            masks = Variable(datapoint['masks'].cuda())
            bg_masks = Variable(datapoint['backgrnd'].cuda())
            edgeweight = Variable(datapoint['edgeweight'].cuda())
        else:
            image_src = Variable(datapoint['image'])
            i = batch # TODO check if this is incorrect
            if (i < num_tar):
                image_tgt = Variable(list_tar[batch][1]['image'].type(torch.FloatTensor))
            else:
                image_tgt = None
            masks = Variable(datapoint['masks'])
            bg_masks = Variable(datapoint['backgrnd'])
            edgeweight = Variable(datapoint['edgeweight'])
        return image_src, image_tgt, masks, bg_masks, edgeweight

    def train_epoch(self, train_loader_src, train_loader_tgt, batch_sz, lambda_coral, optimizer, scheduler, epoch):
        self.cuda()
        self.train()
        list_tar = list(enumerate(train_loader_tgt))
        num_tar = len(list_tar)
        count = 0.
        loss_man_cum = 0.
        loss_coral_cum = 0.
        total_loss_cum = 0.

        for i, datapoint in enumerate(train_loader_src):
            optimizer.zero_grad()
            count += 1
            image_src, image_tgt, masks, bg_masks, edgeweight = self.input_from_loaders(datapoint, list_tar, num_tar,
                                                                                        batch=i)

            y_fg, y_bg, y_edg, y_rec, f_src = self(image_src)

            loss_man = self.adaptive_diceloss(fg=masks, bg=bg_masks, edg=edgeweight,
                                              y_fg=y_fg, y_bg=y_bg, y_edg=y_edg)

            src_recon_loss, tgt_recon_loss, total_loss_coral = self.domainadapt_loss(image_src, f_src,
                                                                                     y_rec, image_tgt, lambda_coral=1.0)
            m1 = loss_man.item()
            m2 = total_loss_coral.item() + src_recon_loss.item() + tgt_recon_loss.item()
            m = np.sqrt(m1 ** 2 + m2 ** 2 + 1e-30)
            loss = (m1 / m) * loss_man + (m2 / m) * total_loss_coral + (m2 / m) * (src_recon_loss + tgt_recon_loss)
            loss_man_cum += loss_man.data.cpu().numpy()
            loss_coral_cum += total_loss_coral.data.cpu().numpy()
            total_loss_cum += loss.data.cpu().numpy()

            loss.backward()
            optimizer.step()

            print_stats(epoch=epoch, iter=i, max_iter=len(train_loader_src),
                        man_loss=loss_man.data.cpu().numpy(),
                        coral_loss=total_loss_coral.data.cpu().numpy(),
                        total_loss=loss.data.cpu().numpy())

        loss_man_avg = loss_man_cum / count
        loss_coral_avg = loss_coral_cum / count
        loss_avg = total_loss_cum / count
        print('[%s] Epoch %5d - Avg. Loss MAN: %.6f - Avg. Loss CORAL: %.6f - Avg. Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_man_avg, loss_coral_avg, loss_avg))

        return loss_avg, loss


def print_stats(epoch, iter, max_iter, man_loss, coral_loss, total_loss, interval=50):
    if np.mod(iter, interval) == 0:
        print("Epoch:%5d-MiniBatch:%5d/%5d-Supervised_Loss:%.5f-Reg._Loss:%.5f-Loss=%.5f" % (
        epoch, iter, max_iter, man_loss, coral_loss, total_loss))


def coral(source, target):
    # print(source.shape, target.shape)

    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.mm(xm.view(xm.size(1), xm.size(0)), xm)
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.mm(xmt.view(xmt.size(1), xmt.size(0)), xmt)
    loss = torch.mean(torch.pow(xc - xct, 2))
    return loss
