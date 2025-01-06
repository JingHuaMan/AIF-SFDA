'''
    This is a lightweight implementation of DAMAN, where a single encoder-decoder network is used (see Fig. 6(c)) in paper
    The single network outputs four predicted images:
    1. predicted foreground image: \hat{y}_b
    2. predicted background image: \hat{y}_f
    3. predicted edge map: \hat{y}_e
    4. full image reconstruction: \hat{x}
    Depending on your application, this network could be slightly less accurate (for touching cells), but it is easier to train.
'''

import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib

matplotlib.use('TkAgg')

from others.other_components.daman.daman_networks.daman_unet import UNetEncoder2D, UNetDecoder2D
from others.other_components.daman.daman_util.daman_losses import DiceLoss, BCELoss, MSELoss
import time


class DaManLiteNet(nn.Module):
    """
    class for the Domain-Adapted MAN for coral loss function
    """

    def __init__(self, in_channels=1, feature_maps=16, out_channels=4, levels=7, group_norm=False, adaptive=False,
                 savepath='./logs/', save_intervl=5):
        super().__init__()
        self.model_save_freq = save_intervl
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.adaptive = adaptive
        self.save_model_path = savepath
        self.active_sigmoid = nn.Sigmoid()
        self.encoder = UNetEncoder2D(in_channels=in_channels, feature_maps=feature_maps, levels=levels,
                                     group_norm=group_norm)
        self.segmentation_decoder = UNetDecoder2D(out_channels=out_channels, feature_maps=feature_maps, levels=levels,
                                                  group_norm=group_norm)

    def forward(self, inputs):
        encoder_outputs, encoded = self.encoder(inputs)
        decoder_outputs, segmentation_outputs = self.segmentation_decoder(encoded, encoder_outputs)
        features = []
        for encoder_output in encoder_outputs:
            features.append(encoder_output)
        features.append(encoded)
        for decoder_output in decoder_outputs:
            features.append(decoder_output)
        yrec = segmentation_outputs[:, 0, :, :]
        y_fg = segmentation_outputs[:, 1, :, :]
        y_bg = segmentation_outputs[:, 2, :, :]
        y_edg = segmentation_outputs[:, 3, :, :]
        yrec, y_fg, y_bg, y_edg = yrec[np.newaxis], y_fg[np.newaxis], y_bg[np.newaxis], y_edg[np.newaxis]
        y_fg = self.active_sigmoid(y_fg)
        y_bg = self.active_sigmoid(y_bg)
        y_edg = self.active_sigmoid(y_edg)

        return features, yrec, y_fg, y_bg, y_edg

    def adaptive_diceloss(self, fg, bg, edg, y_fg, y_bg, y_edg):
        loss_fn = DiceLoss()
        y_fg = y_fg.permute(1, 0, 2, 3)
        y_bg = y_bg.permute(1, 0, 2, 3)
        y_edg = y_edg.permute(1, 0, 2, 3)
        E1 = loss_fn(y_fg, fg)
        E2 = loss_fn(1. - y_bg, 1. - bg)
        E3 = loss_fn(y_edg, edg)

        deno = np.sqrt((E1.item()) ** 2 + (E2.item()) ** 2 + (E3.item()) ** 2 + 1e-30)
        mu1, mu2, mu3 = E1.item() / deno, E2.item() / deno, E3.item() / deno
        loss = mu1 * E1 + mu2 * E2 + mu3 * E3
        return loss

    def adaptive_bceloss(self, fg, bg, edg, y_fg, y_bg, y_edg):
        loss_fn = BCELoss()
        y_fg = y_fg.permute(1, 0, 2, 3)
        y_bg = y_bg.permute(1, 0, 2, 3)
        y_edg = y_edg.permute(1, 0, 2, 3)
        E1 = loss_fn(y_fg, fg)
        E2 = loss_fn(y_bg, bg)
        E3 = loss_fn(y_edg, edg)
        concave_comb = True
        if concave_comb:
            deno = np.sqrt((E1.item()) ** 2 + (E2.item()) ** 2 + (E3.item()) ** 2 + 1e-30)
            mu1, mu2, mu3 = E1.item() / deno, E2.item() / deno, E3.item() / deno
        else:
            mu1, mu2, mu3 = 0.43, 0.1, 0.43
        loss = mu1 * E1 + mu2 * E2 + mu3 * E3
        return loss

    def domainadapt_loss(self, image_src, f_src, y_rec, image_tgt, lambda_coral=1.):
        src_recon_loss = MSELoss()(image_src, y_rec)
        total_loss_coral = 0
        tgt_recon_loss = 0
        if image_tgt is not None:
            f_tgt, rec_tgt, _, _, _ = self(image_tgt)
            mid = int(len(f_src) / 2)
            f_j_src = f_src[mid]
            f_j_tar = f_tgt[mid]
            sz = f_j_src.size()
            for s in range(sz[1]):
                loss_coral = coral(f_j_src[:, s, :, :].view(sz[0], sz[2] * sz[3]),
                                   f_j_tar[:, s, :, :].view(sz[0], sz[2] * sz[3]))
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
        model_save_freq = self.model_save_freq
        cum_loss = 0.
        for epoch in range(start_epoch, num_epochs):
            epoch_loss, loss_tensor = self.train_epoch(train_loader_src, train_loader_tgt, batch_sz, lambda_coral,
                                                       optimizer, scheduler, epoch)
            cum_loss += epoch_loss
            if (epoch + 1) % model_save_freq == 0 or epoch == num_epochs - 1:
                model_save_name = self.save_model_path + "DAMANLite-" + str(epoch + 1) + ".pt"
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
            image_src, image_tgt, masks, bg_masks, edgeweight = image_src.cuda(), image_tgt.cuda(), masks.cuda(), bg_masks.cuda(), edgeweight.cuda()

            f_src, y_rec, y_fg, y_bg, y_edg = self(image_src)
            loss_supervised = self.adaptive_diceloss(fg=masks, bg=bg_masks, edg=edgeweight, y_fg=y_fg, y_bg=y_bg,
                                                     y_edg=y_edg)
            src_recon_loss, tgt_recon_loss, total_loss_coral = self.domainadapt_loss(image_src, f_src,
                                                                                     y_rec, image_tgt, lambda_coral=1.0)
            m1 = loss_supervised.item()
            m2 = total_loss_coral.item() + src_recon_loss.item() + tgt_recon_loss.item()
            """
            Balance the losses via min-max strategy
            """
            m = np.sqrt(m1 ** 2 + m2 ** 2 + 1e-30)
            loss = (m1 / m) * loss_supervised + (m2 / m) * total_loss_coral
            loss_man_cum += loss_supervised.data.cpu().numpy()
            loss_coral_cum += total_loss_coral.data.cpu().numpy()
            total_loss_cum += loss.data.cpu().numpy()

            loss.backward()
            optimizer.step()

            print_stats(epoch=epoch, iter=i, max_iter=len(train_loader_src),
                        man_loss=loss_supervised.data.cpu().numpy(),
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
        print("Epoch:%5d-MiniBatch:%5d/%5d-MAN_loss:%.5f-Coral_Loss:%.5f-Loss=%.5f" % (
        epoch, iter, max_iter, man_loss, coral_loss, total_loss))


def coral(source, target):
    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.mm(xm.view(xm.size(1), xm.size(0)), xm)
    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.mm(xmt.view(xmt.size(1), xmt.size(0)), xmt)
    # frobenius norm between source and target
    loss = torch.mean(torch.pow(xc - xct, 2))
    return loss
