import numpy as np
import torch
import random
from random import randint
from skimage.color import label2rgb
from torch.utils.data import Dataset, DataLoader
import glob
from others.other_components.daman.daman_util.daman_helps_docker import StdIO as IO
from others.other_components.daman.daman_util.daman_helps_docker import StdIP as IP


def man_augment(image, masks, edgewt):
    """
    Assumes that the sample is {'image', 'masks', 'edgeweight'}
    :param sample:
    :return:
    """
    vertical_prob = 0.5
    horizontal_prob = 0.5
    gamma_prob = 0.3
    bright_prob = 0.3
    if (random.random() < vertical_prob):
        image = np.flipud(image)
        masks = np.flipud(masks)
        edgewt = np.flipud(edgewt)

    if (random.random() < horizontal_prob):
        image = np.fliplr(image)
        masks = np.fliplr(masks)
        edgewt = np.fliplr(edgewt)

    if (random.random() < bright_prob):  # brightness change
        value = randint(2, 15)
        image = image + value
        image[image > 255] = 255.
        image[image < 0] = 0.
        masks = masks
        edgewt = edgewt

    if (random.random() < gamma_prob):  # Gaussian noise
        im = image
        row, col = im.shape
        const = 2
        b = 2
        a = 0.1
        gamma = (1 / ((b - a) * random.random() + a))
        gamma_im = const * (im ** gamma)
        image = gamma_im
        masks = masks
        edgewt = edgewt

    image = IP.linstretch(image)
    # sample_return = {'image':image, 'masks':masks, 'edgeweight':edgewt}

    return image, masks, edgewt


class CombineTrainDataset(Dataset):  # Defining the class to load both source and
    """
    Consolidates the training set to contain both the source and target data
    The data is tagged to remember whether it comes from Source or Target
    """

    def __init__(self, base_directory='/data/', transform=True):
        self.base_dir = base_directory
        self.transform = transform
        src_train_path = base_directory + 'source/train_img/'  # Source trainfiles
        tgt_train_path = base_directory + 'target/train_img/'  # Target trainfiles
        src_list = glob.glob(src_train_path + "*.png")
        tgt_list = glob.glob(tgt_train_path + "*.png")
        # Now join the two source and target data lists. Later, tag which data is source and which is target
        self.imglist = src_list + tgt_list

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        """
        This function fetches the data from the dataloader which has {'image', 'masks', 'edgeweight'}
        :param idx:
        :return:
        """
        img_fname = self.imglist[idx]
        image = IO.imread_2d(img_fname)
        Nr, Nc = image.shape[0], image.shape[1]
        no_of_masks = 2
        masks = np.zeros((Nr, Nc))  # empty mask in case nothing available
        bg_masks = np.zeros((Nr, Nc))  # empty mask in case nothing available
        edgeweight_map = np.zeros((Nr, Nc))  # empty mask in case nothing available

        # First check if this image belongs to the source or target
        if img_fname.split('/')[-3] == 'source':
            is_src = True
        else:
            is_src = False
        # Currently labels are only available for source-trainsets
        if (is_src):
            mask_fname = img_fname.replace('train_img', 'train_fg_label')
            edge_fname = img_fname.replace('train_img', 'train_edg_label')
            masks = IP.linstretch(IO.imread_2d(mask_fname))
            bg_masks = 1.0 - masks
            edgeweight_map = IO.imread_2d(edge_fname)
            edgeweight_map = (edgeweight_map - np.min(edgeweight_map)) / np.ptp(edgeweight_map)

        # Data augmentation
        if self.transform:
            img_aug, masks_aug, edgwt_aug = man_augment(image.copy(), masks.copy(), edgeweight_map.copy())
            bg_masks_aug = 1.0 - IP.linstretch(masks_aug)
        else:
            img_aug = 1. * image.copy()
            masks_aug = 1. * masks.copy()
            bg_masks_aug = 1. * bg_masks.copy()
            edgwt_aug = 1. * edgeweight_map.copy()

        # Now store the data in the (num_ch, Nr, Nc) format
        sample_chw = {}
        sample_chw['image'] = np.ascontiguousarray(img_aug[np.newaxis, ...], dtype='float')
        sample_chw['masks'] = np.ascontiguousarray(masks_aug[np.newaxis, ...], dtype='float')
        sample_chw['backgrnd'] = np.ascontiguousarray(bg_masks_aug[np.newaxis, ...], dtype='float')
        sample_chw['edgeweight'] = np.ascontiguousarray(edgwt_aug[np.newaxis, ...], dtype='float')
        # sample_chw['count'] = no_of_masks
        if is_src == True:
            sample_chw['is_src'] = 1
        else:
            sample_chw['is_src'] = 0
        return sample_chw


class ImageDataset(Dataset):  # Defining the class to load datasets
    def __init__(self, base_directory='/data/', src=True, trainset=True, transform=True, resize=True, res_factor=4,
                 times_augment=10):
        self.base_dir = base_directory
        self.is_src = src
        self.is_trainset = trainset
        self.transform = transform
        self.resize = resize
        self.xaugment = times_augment
        self.resize_factor = res_factor

        # Select the image-list
        if src == True:
            path1 = 'source/'
        else:
            path1 = 'target/'
        if trainset == True:
            path2 = 'train_img/'
        else:
            path2 = 'test_img/'

        self.img_path = base_directory + path1 + path2
        self.imglist = glob.glob(self.img_path + "*.png")

    def __len__(self):
        return min(1000, self.xaugment * len(self.imglist))

    def __getitem__(self, idx):
        """
        This function fetches the data from the dataloader which has {'image', 'masks', 'edgeweight'}
        :param idx:
        :return:
        """
        idx = np.mod(idx, len(self.imglist))

        img_fname = self.imglist[idx]
        image = IO.imread_2d(img_fname)
        Nr, Nc = image.shape[0], image.shape[1]
        # no_of_masks = 2
        masks = np.zeros((Nr, Nc))  # empty mask in case nothing available (target)
        bg_masks = np.zeros((Nr, Nc))  # empty mask in case nothing available (target)
        edgeweight_map = np.zeros((Nr, Nc))  # empty mask in case nothing available (target)

        # Read the image, mask, background, edge_mask for each training sample in source
        if (self.is_trainset and self.is_src):
            mask_fname = img_fname.replace('train_img', 'train_fg_label')
            edge_fname = img_fname.replace('train_img', 'train_edg_label')
            masks = 1. * (IO.imread_2d(mask_fname) > 0.1)
            bg_masks = 1.0 - masks
            edgeweight_map = IO.imread_2d(edge_fname)
            edgeweight_map = (edgeweight_map - np.min(edgeweight_map)) / np.ptp(edgeweight_map)
            edgeweight_map = 1. * (edgeweight_map > 0.1)

        # Read the image-masks pair for any test data
        if (self.is_trainset is False):
            mask_fname = img_fname.replace('test_img', 'test_label')
            masks = 1. * (IO.imread_2d(mask_fname) > 0.1)

        # Data augmentation
        if self.transform:
            img_aug, masks_aug, edgwt_aug = man_augment(image.copy(), masks.copy(), edgeweight_map.copy())
            bg_masks_aug = 1.0 - 1. * (masks_aug > 0.1)
            edgwt_aug = 1. * (edgwt_aug > 0.1)
        else:
            img_aug = 1. * image
            masks_aug = 1. * masks
            bg_masks_aug = 1. * bg_masks
            edgwt_aug = 1. * edgeweight_map

        # Now store the data in the (num_ch, Nr, Nc) format
        sample_chw = {}

        if self.resize:
            img_aug = IP.imresize(img_aug, des_res_mm=self.resize_factor)
            masks_aug = 1. * (IP.imresize(masks_aug, des_res_mm=self.resize_factor) > 0.1)
            bg_masks_aug = 1. * (IP.imresize(bg_masks_aug, des_res_mm=self.resize_factor) > 0.1)
            edgwt_aug = IP.imresize(edgwt_aug, des_res_mm=self.resize_factor)

        sample_chw['image'] = np.ascontiguousarray(img_aug[np.newaxis, ...], dtype='float')
        sample_chw['masks'] = np.ascontiguousarray(masks_aug[np.newaxis, ...], dtype='float')
        sample_chw['backgrnd'] = np.ascontiguousarray(bg_masks_aug[np.newaxis, ...], dtype='float')
        sample_chw['edgeweight'] = np.ascontiguousarray(edgwt_aug[np.newaxis, ...], dtype='float')
        # sample_chw['count'] = no_of_masks
        if self.is_src == True:
            sample_chw['is_src'] = 1
        else:
            sample_chw['is_src'] = 0
        return sample_chw


def sample_labeled_input(data, labels, input_shape):
    # randomize seed
    np.random.seed()

    # generate random position
    x = np.random.randint(0, data.shape[0] - input_shape[0] + 1)
    y = np.random.randint(0, data.shape[1] - input_shape[1] + 1)
    z = np.random.randint(0, data.shape[2] - input_shape[2] + 1)

    # extract input and target patch
    input = data[x:x + input_shape[0], y:y + input_shape[1], z:z + input_shape[2]]
    if len(labels) > 0:
        target = labels[x:x + input_shape[0], y:y + input_shape[1], z:z + input_shape[2]]
    else:
        target = []

    return input, target


def sample_unlabeled_input(data, input_shape):
    np.random.seed()
    x = np.random.randint(0, data.shape[0] - input_shape[0] + 1)
    y = np.random.randint(0, data.shape[1] - input_shape[1] + 1)
    z = np.random.randint(0, data.shape[2] - input_shape[2] + 1)
    input = data[x:x + input_shape[0], y:y + input_shape[1], z:z + input_shape[2]]
    return input


def gaussian_window(size, sigma=1):
    # half window sizes
    hwz = size[0] // 2
    hwy = size[1] // 2
    hwx = size[2] // 2

    # construct mesh grid
    if size[0] % 2 == 0:
        axz = np.arange(-hwz, hwz)
    else:
        axz = np.arange(-hwz, hwz + 1)
    if size[1] % 2 == 0:
        axy = np.arange(-hwy, hwy)
    else:
        axy = np.arange(-hwy, hwy + 1)
    if size[2] % 2 == 0:
        axx = np.arange(-hwx, hwx)
    else:
        axx = np.arange(-hwx, hwx + 1)
    xx, zz, yy = np.meshgrid(axx, axz, axy)

    # normal distribution
    gw = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2. * sigma ** 2))

    # normalize so that the mask integrates to 1
    gw = gw / np.sum(gw)

    return gw


# load a network
def load_net(model_file):
    return torch.load(model_file)


# returns an image overlayed with a labeled mask
# x is assumed to be a grayscale numpy array
# y is assumed to be a numpy array of integers for the different labels, all zeros will be opaque
def overlay(x, y, alpha=0.3, bg_label=0):
    return label2rgb(y, image=x, alpha=alpha, bg_label=bg_label, colors=[[0, 1, 0]])