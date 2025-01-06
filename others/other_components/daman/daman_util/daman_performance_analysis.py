import numpy as np
import pickle
from others.other_components.daman.daman_util.daman_helps_docker import BinaryMorphology as bwmorph
from others.other_components.daman.daman_util.daman_helps_docker import Thresholding as T
from scipy import ndimage as ndi
from skimage import measure


def dice_loss(pred, target, smooth=0.001):
    intersection = np.sum(pred * target)
    loss = ((2. * intersection) / (np.sum(pred) + np.sum(target) + smooth))
    return loss


def mse_loss(pred, target, smooth=1.):
    pred = pred.astype(int)
    target = target.astype(int)
    loss = np.sqrt(np.mean((pred - target) ** 2))
    return loss


def ReconstructObjects(y_fg, y_bg, y_edg, data, min_ar=50, min_intensity=0.15):
    fg_seg, _ = T(y_fg).otsu_threshold()
    bg_seg, _ = T(y_bg).otsu_threshold()
    edg_seg, th = T(y_edg).otsu_threshold()
    edg_seg = 1. * (y_edg > 0.1 * th)
    edg_seg = bwmorph(edg_seg).bwclose(r=5)
    fg_estimated = 0.5 * (fg_seg + 1. - bg_seg)
    scissored_img = 1. * ((fg_estimated * (1. - edg_seg)) > 0.1)  # eliminate edge and chop off joined cells
    cells = bwmorph(scissored_img).bwareaopen(area=min_ar)
    cells = bwmorph(cells).bwfill()
    seg = bwmorph(cells).bwopen(r=3)
    label_img, _ = bwmorph(seg).bwlabel()
    ncc = int(label_img.max())

    # Now look at individual cells and dilate it, as their borders have been chopped off
    stack = np.zeros((ncc, label_img.shape[0], label_img.shape[1]))
    for labl in range(1, ncc + 1):
        bw = np.zeros(label_img.shape)
        bw[label_img == labl] = 1
        bw = bwmorph(bw).bwdilate(r=3)
        bw = bwmorph(bw).bwareaopen(area=min_ar)
        avg_in = np.sum(data * bw) / (bw.sum() + 1e-10)
        if avg_in > min_intensity:  # 0.1
            stack[labl - 1] = bw * labl
        else:
            stack[labl - 1] = bw * 0.
    if ncc > 0:
        label_img = np.max(stack, axis=0)

    return label_img

    fg_seg, _ = T(y_fg).otsu_threshold()
    bg_seg, _ = T(y_bg).otsu_threshold()
    edg_seg, th = T(y_edg).otsu_threshold()
    edg_seg = bwmorph(edg_seg).bwclose(r=12)
    fg_estimated = 0.5 * (fg_seg + 1. - bg_seg)  # Average the foreground and background signal

    scissored_img = 1. * ((fg_estimated * (1. - edg_seg)) > 0.1)  # eliminate edge and chop off joined cells
    cells = bwmorph(scissored_img).bwareaopen(area=min_ar)
    cells = bwmorph(cells).bwfill()
    seg = bwmorph(cells).bwopen(r=3)

    label_img, _ = bwmorph(seg).bwlabel()
    ncc = int(label_img.max())

    # Now look at individual cells and dilate it, as their borders have been chopped off
    stack = np.zeros((ncc, label_img.shape[0], label_img.shape[1]))
    for labl in range(1, ncc + 1):
        bw = np.zeros(label_img.shape)
        bw[label_img == labl] = 1
        bw = bwmorph(bw).bwdilate(r=6)
        bw = bwmorph(bw).bwareaopen(area=min_ar)
        avg_in = np.sum(data * bw) / (bw.sum() + 1e-10)
        if avg_in > min_intensity:  # 0.1
            stack[labl - 1] = bw * labl
        else:
            stack[labl - 1] = bw * 0.
    if ncc > 0:
        label_img = np.max(stack, axis=0)

    return label_img

    with open(dict_fpath, 'rb') as handle:
        saved_dict = pickle.load(handle)
    ndata = len(saved_dict['gt'])
    percell_dice = []
    for idx in range(ndata):
        gt = saved_dict['res'][idx]  # reverse prediction and gt
        fg = saved_dict['gt'][idx]
        gt = 1. * (gt > 0.)
        pred = 1. * (fg > 0.)

        # pred = bwmorph(pred).bwerode(2)
        labeled_segments_gt = ndi.label(gt)[0]
        labeled_segments_pred = ndi.label(pred)[0]
        # Number of conn_comps
        np_gt = np.max(labeled_segments_gt)
        reg_prop_gt = measure.regionprops(labeled_segments_gt.astype(int))
        np_pred = np.max(labeled_segments_pred)
        reg_prop_pred = measure.regionprops(labeled_segments_pred.astype(int))
        dice_percell = np.zeros((np_gt))  # {}
        mse_percell = np.zeros((np_gt))  # {}

        # import pdb; pdb.set_trace()
        # Now look at each label in ground truth
        for i in range(np_gt):
            if reg_prop_gt[i].area < 5000 and reg_prop_gt[i].area > 400:
                cell_gt = np.zeros(np.shape(gt))
                cell_pred = np.zeros(np.shape(pred))
                cell_gt[reg_prop_gt[i].coords[:, 0], reg_prop_gt[i].coords[:, 1]] = 1
                centroid_gt = reg_prop_gt[i].centroid
                dist = [np.sqrt((x.centroid[0] - centroid_gt[0]) ** 2 + (x.centroid[1] - centroid_gt[1]) ** 2)
                        for x in reg_prop_pred]

                position = np.argsort(dist)
                if len(position) == 0:
                    dice_percell[i] = 0
                else:
                    pp = position[0]
                    # if dist[pp] < 300:
                    cell_pred[reg_prop_pred[pp].coords[:, 0], reg_prop_pred[pp].coords[:, 1]] = 1
                    cell_pred = bwmorph(cell_pred).bwopen(1)
                    percell_dice.append(dice_loss(cell_gt, cell_pred))
                # mse_percell[i] = mse_loss(cell_gt, cell_pred)
        # print("Completed data -", idx, "/", ndata)
    np.save(savepath + '.npy', percell_dice)
    print("Saved at: ", savepath)
    return percell_dice