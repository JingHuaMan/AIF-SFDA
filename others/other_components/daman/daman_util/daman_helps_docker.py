import numpy as np
import scipy as sp
import scipy.misc as miscio
import scipy.ndimage as ndimage
import cv2
# from matplotlib import pyplot as plt
import glob
#import SimpleITK as sitk
from skimage.io import imsave

from skimage.morphology import black_tophat, skeletonize, convex_hull_image, remove_small_objects, remove_small_holes, label
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening
from skimage.morphology import erosion, dilation, closing, opening
from skimage.measure import regionprops
from skimage.morphology import disk
# from matplotlib import pyplot as plt
from skimage.filters import gaussian
from scipy.ndimage import gaussian_laplace
from scipy import ndimage as spndim


eps = 1e-8


def bwdist(a):
    """
    Intermediary function. 'a' has only True/False vals,
    so we convert them into 0/1 values - in reverse.
    True is 0, False is 1, distance_transform_edt wants it that way.
    """
    return ndimage.distance_transform_edt(a == 0)

class StdIO:

    # @staticmethod
    # def imshow(img, title='display', col_map='gray', imadjust=True, pts=[], pt_col='y'):
    #     fig = plt.figure(title)
    #     if imadjust:
    #         img = (img - np.min(img))/(eps + np.max(img) - np.min(img))
    #     plt.imshow(img, cmap=col_map)
    #     if len(pts) > 0:
    #         x, y = list(pts[:, 0]), list(pts[:, 1])
    #         plt.scatter(x, y, c=pt_col, s=20, zorder=2)
    #     plt.tight_layout()
    #     plt.axis('off')
    #     plt.show()


    @staticmethod
    def imread_2d(fname, grayscale=True):
        img = ndimage.imread(fname, flatten=grayscale, mode=None)
        if img.max() > 0:
            img = img/img.max()
        return 1. * img

    @staticmethod
    def imwrite_2d(fname, img, cmin=0, cmax=255):
        # miscio.toimage(img, cmin=cmin, cmax=cmax, mode='I').save(fname)
        imsave(fname, img)

    @staticmethod
    def imread_multiple(folder_path, ext='.png', return_image=True, des_res_mm=None):
        """
        returns the stack of image as stack[ii, :, :]
        shape: (N, Nr, Nc)
        """
        all_data = glob.glob(folder_path + '*' + ext)
        test_img = StdIO.imread_2d(all_data[0])
        if des_res_mm is not None:
            test_img = StdIP.imresize(test_img, des_res_mm=des_res_mm)
        # import pdb; pdb.set_trace()

        Nr, Nc = test_img.shape[0], test_img.shape[1]
        N = len(all_data)
        if return_image == True:
            # import pdb; pdb.set_trace()
            imstack = np.zeros((N, Nr, Nc), dtype='float')
            for ii in range(N):
                img = StdIO.imread_2d(all_data[ii])
                if des_res_mm is not None:
                    img = StdIP.imresize(img, des_res_mm=des_res_mm)
                imstack[ii] =  img/(eps + np.max(img))
            return imstack
        else:
            return all_data


class StdIP:
    @staticmethod
    def im2double(a):
        a = a.astype(np.float)
        a /= np.abs(a).max()
        return a

    @staticmethod
    def mask2phi(init_a):
        phi = bwdist(init_a) - bwdist(1 - init_a) + StdIP.im2double(init_a) - 0.5
        return -phi

    @staticmethod
    def bwdist(a):
        """
        Intermediary function. 'a' has only True/False vals,
        so we convert them into 0/1 values - in reverse.
        True is 0, False is 1, distance_transform_edt wants it that way.
        """
        return ndimage.distance_transform_edt(a == 0)


    @staticmethod
    def percentile_enhance(img, p1=0., p2=100.):
        """
        Keep values between p1-th and p2-th percentile
        :param p1: lower percentile
        :param p2: higher percentile
        :return: binary image
        """
        low = min(p1, p2)
        high = max(p1, p2)
        low_val = np.percentile(img, low)
        high_val = np.percentile(img, high)
        # print low
        # enh_img = np.copy(img)
        enh_img = (img - low_val)/(high_val - low_val)
        enh_img[enh_img < 0.] = 0.
        enh_img[enh_img > 1.] = 1.
        return enh_img

    @staticmethod
    def imresize(img, orig_res_mm=1., des_res_mm=0.5, interpolation='cubic'):
        frac = orig_res_mm/des_res_mm
        a = sp.misc.imresize(img, frac, interp=interpolation, mode='F')*1.
        if (a.max() > 0.):
            a = a/a.max()
        return a

    @staticmethod
    def linstretch(img):
        if img.max() > 0:
            img = (img - np.min(img)) / (eps + np.max(img) - np.min(img))
        return img

    @staticmethod
    def numpy_to_opencv(img):
        img = img/(eps + np.max(img))
        ocv_img = (img * 255).astype('uint8')
        return ocv_img

    @staticmethod
    def opencv_to_numpy(ocv_img):
        np_img = ocv_img.astype('float')/255.
        return np_img

"""
Set of morphological filters which will work on binary images only.
usage BinaryMorphology(img).method()
"""
class BinaryMorphology:
    def __init__(self, img):
        """
        Initializer of the object
        :param img: the binary image. If non binary, forced to binary by thresholding at 0.1
        """
        self.bw = (img > 0.1) * 1.  # force convert to binary

    def bwdilate(self, r=1):
        se = disk(r)
        return binary_dilation(self.bw > 0, selem=se)

    def bwerode(self, r=1):
        se = disk(r)
        return binary_erosion(self.bw > 0, selem=se)

    def bwopen(self, r=1):
        se = disk(r)
        return binary_opening(self.bw > 0, selem=se)

    def bwclose(self, r=1):
        se = disk(r)
        return binary_closing(self.bw > 0, selem=se)

    def bwskel(self):
        """
        Skeletonize a binary image
        :return:
        """
        return 1. * skeletonize(self.bw)

    def bwdist(self):
        """
        Intermediary function. 'a' has only True/False vals,
        so we convert them into 0/1 values - in reverse.
        True is 0, False is 1, distance_transform_edt wants it that way.
        """
        return ndimage.distance_transform_edt(self.bw == 0)

    def bwareaopen(self, area=20):
        """
        Area opening
        :return:
        """
        return remove_small_objects(self.bw > 0., min_size=area) * 1.

    def bwareaclose(self, area=20):
        """
        Area opening
        :return:
        """
        return remove_small_holes(self.bw > 0., min_size=area) * 1.

    def bwlabel(self):
        """
        Label the connected components
        :return: Labelled objects with colors
        """
        label_img = label(self.bw)
        return 1. * label_img, label_img.max()

    def bwfill(self):
        from scipy.ndimage.morphology import binary_fill_holes
        return binary_fill_holes(self.bw > 0.)

    def klargestregions(self, k=1):
        label_img = label(self.bw)
        num_cc = label_img.max()
        k = max(1, min(k, num_cc))
        props = regionprops(label_img)
        area_list = []
        for ii in range(num_cc):
            area_list.append(props[ii].area)

        top_k_comp = np.zeros(label_img.shape)
        if k > 1:
            top_k_labels = np.argsort(area_list)[::-1][0:k] + 1
            for jj in range(k):
                top_k_comp[label_img==top_k_labels[jj]] = top_k_labels[jj]
        else:   # simpler problem of finding largest component
            top_label = np.argmax(area_list) + 1
            top_k_comp[label_img==top_label] = 1
        return top_k_comp


"""
Class of gray-morphological filters
Usage: GrayMorphology(img).method()
"""
class GrayMorphology():

    def __init__(self, img):
        """
        Initializer of the object.
        :param img: grayscale image to perform filtering on
        """
        self.img = StdIP.im2double(img)

    def imdilate(self, r=1):
        se = disk(r)
        return dilation(self.img, selem=se)* 1.

    def imerode(self, r=1):
        se = disk(r)
        return erosion(self.img, selem=se)* 1.

    def imopen(self, r=1):
        se = disk(r)
        return opening(self.img, selem=se)* 1.

    def imclose(self, r=1):
        se = disk(r)
        return closing(self.img, selem=se)* 1.

    def areaopen(self, area=100, l1=0., l2=1.):

        l1 = int(min(255*self.img.min(), 255* (l1/(l1 + 1e-5))))
        l2 = int(max(self.img.max()*255, (255 * (l2 / (l2 + 1e-5)))))
        pixel_range = range(l1, l2 + 1)

        recon_img = np.zeros(self.img.shape)

        for level in pixel_range:
            tmp_bw_img = 1. * (255. * self.img > level)
            tmp_clean_img = BinaryMorphology(tmp_bw_img).bwareaopen(area=area)
            recon_img += tmp_clean_img

        recon_img = StdIP.linstretch(recon_img/len(pixel_range))
        return recon_img


class BinaryMorphology3D:
    def __init__(self, img, r=1):
        """
        Initializer of the object
        :param img: the binary image. If non binary, forced to binary by thresholding at 0.1
        """
        self.bw = (img > 0.1) * 1.  # force convert to binary
        sz = 6*r + 1
        xx, yy, zz = np.meshgrid(range(sz), range(sz), range(sz))
        x0, y0, z0 = (sz - 1)/2,  (sz - 1)/2,  (sz - 1)/2
        f = (xx - x0)**2 + (yy - y0)**2 + (zz - z0)**2 - r*r
        self.se = 1.*(f <= 0)

    def bwdilate(self, r=1):
        se = self.se
        dil_img = spndim.binary_dilation(self.bw, structure=self.se)
        return 1. * (dil_img > 0.)

    def bwclose(self, r=1):
        se = self.se
        res_img = spndim.binary_closing(self.bw, structure=self.se)
        return 1. * (res_img > 0.)



"""
A class of a few commonly used spatial filters, implemented as static methods
"""

class Filter():

    @staticmethod
    def gaussian_filter(img, sigma_mm=1., res=1.):
        sigma = sigma_mm/res
        blurred_img = gaussian(img, sigma)
        return blurred_img

    @staticmethod
    def log_filter(img, sigma_mm=1., res=1.):
        log = gaussian_laplace(img, sigma=sigma_mm/res)
        return log

    @staticmethod
    def laplacian_filter(img):
        g = np.gradient(img)
        g1 = np.gradient(g[0])
        g2 = np.gradient(g[1])
        gxx, gyy = g1[0], g2[1]
        return gxx + gyy

    @staticmethod
    def gradient_filter(img, sigma_mm=1., res=1.):
        """
        :return: g[0]: gradient along Y, g[1]: gradient along X
        """
        if sigma_mm > 0:
            f = Filter.gaussian_filter(img, sigma_mm, res)
        else:
            f = np.copy(img)
        g = np.gradient(f)
        g_mag = np.sqrt(g[0]**2 + g[1]**2)
        return g, g_mag

    @staticmethod
    def median_filter(img, r=1):
        img_ocv = StdIP.numpy_to_opencv(img)
        smooth_ocv = cv2.medianBlur(img_ocv, ksize=r)
        return StdIP.opencv_to_numpy(smooth_ocv)


class Thresholding:
    def __init__(self, img):
        self.img = StdIP.im2double(img)

    def otsu_threshold(self):
        img_cv = StdIP.numpy_to_opencv(self.img)
        tval, th_cv = cv2.threshold(img_cv, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_img = StdIP.opencv_to_numpy(th_cv)
        return bin_img, tval/255.

    def adaptive_threshold(self, type='gaussian', win_sz=11):
        if type=='gaussian':
            th_option = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        elif type=='mean':
            th_option = cv2.ADAPTIVE_THRESH_MEAN_C
        else:
            th_option = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        img_cv = StdIP.numpy_to_opencv(self.img)
        bw_cv = cv2.adaptiveThreshold(img_cv, 255, th_option, cv2.THRESH_BINARY, win_sz, 2)
        return StdIP.opencv_to_numpy(bw_cv)

    def percentile_threshold(self, p1=0., p2=100.):
        """
        Keep values between p1-th and p2-th percentile
        :param p1: lower percentile
        :param p2: higher percentile
        :return: binary image
        """
        low = min(p1, p2)
        high =max(p1, p2)
        low_val = np.percentile(self.img.flatten(), low)
        high_val = np.percentile(self.img.flatten(), high)
        bin_img = np.ones(self.img.shape, dtype='float')
        bin_img[self.img < low_val] = 0.
        bin_img[self.img > high_val] = 0.
        return bin_img
