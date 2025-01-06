import torch

from others.frequency_transforms import FrequencyTransformPrototype


class DFT(FrequencyTransformPrototype):
    def function(self, img):
        # do fft to img, do fftshift, and divide its real part and imaginary part
        img_fft = torch.fft.fft2(img)
        img_fft = torch.fft.fftshift(img_fft)
        return torch.concat([img_fft.real, img_fft.imag], dim=1)

    def inverse_function(self, frequency_map):
        channel_size = frequency_map.size(1)
        img_fft = frequency_map[:, :channel_size // 2] + 1j * frequency_map[:, channel_size // 2:]
        img_fft = torch.fft.ifftshift(img_fft)
        return torch.abs(torch.fft.ifft2(img_fft))

    def normalize_frequency_map(self, frequency_map, visual=False):
        frequency_map = torch.log(torch.abs(frequency_map) + 1e-4)
        frequency_map = (frequency_map - frequency_map.mean()) / frequency_map.std()
        if visual:
            frequency_map = torch.abs(
                frequency_map[:, :frequency_map.size(1) // 2] + 1j * frequency_map[:, frequency_map.size(1) // 2:])
        return torch.sigmoid(frequency_map)
