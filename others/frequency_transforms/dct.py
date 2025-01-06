import torch
import torch_dct

from others.frequency_transforms import FrequencyTransformPrototype


class DCT(FrequencyTransformPrototype):
    """ Discrete Cosine Transform.
        Based on torch-dct, but due to the strange behaviour of the library, we need to do some extra work.
    """
    def __init__(self, norm=None):
        self.norm = norm

    def function(self, img):
        frequency_map = torch.zeros_like(img, dtype=torch.float32, device=img.device)
        for b in range(img.shape[0]):
            for c in range(img.shape[1]):
                frequency_map[b, c, :, :] = torch_dct.dct_2d(img[b, c, :, :].float(), norm=self.norm) / 1000
        return frequency_map

    def inverse_function(self, frequency_map):
        img = torch.zeros_like(frequency_map, dtype=torch.float32, device=frequency_map.device)
        for b in range(frequency_map.shape[0]):
            for c in range(frequency_map.shape[1]):
                img[b, c, :, :] = torch_dct.idct_2d(frequency_map[b, c, :, :].float() * 1000, norm=self.norm)
        return img

    def normalize_frequency_map(self, frequency_map, visual=False):
        frequency_map = torch.log(torch.abs(frequency_map) + 1e-4)
        frequency_map = (frequency_map - frequency_map.mean()) / frequency_map.std()
        return torch.sigmoid(frequency_map)
