import cv2
import numpy as np
import pywt
import torch

from others.frequency_transforms import FrequencyTransformPrototype


class NSCT(FrequencyTransformPrototype):
    def __init__(self, scales=3, directions=8):
        self.scales = scales
        self.directions = directions

    def function(self, img):
        """ Forward NSCT Transform """
        low_pass, high_pass = self.nonsubsampled_wavelet_transform(img)
        directional_subbands = self.directional_filter_bank(high_pass)
        return low_pass, directional_subbands

    def inverse_function(self, frequency_map):
        """ Inverse NSCT Transform """
        low_pass, directional_subbands = frequency_map
        high_pass = self.inverse_directional_filter_bank(directional_subbands)
        img_recon = self.inverse_nonsubsampled_wavelet_transform(low_pass, high_pass)
        return img_recon

    def nonsubsampled_wavelet_transform(self, img):
        # Perform Non-Subsampled Wavelet Transform (NSWT)
        coeffs = pywt.swt2(img.squeeze().cpu().numpy(), wavelet='db1', level=self.scales, start_level=0, trim_approx=True)
        low_pass = torch.tensor(coeffs[0][0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        high_pass = [torch.tensor(coeff[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0) for coeff in coeffs]
        return low_pass, high_pass

    def directional_filter_bank(self, high_pass):
        # Perform Directional Filter Bank (DFB)
        directional_subbands = []
        for hp in high_pass:
            hp_np = hp.squeeze().cpu().numpy()
            for direction in range(self.directions):
                angle = direction * (180 / self.directions)
                kernel = self.create_gabor_filter(angle)
                filtered = cv2.filter2D(hp_np, -1, kernel)
                directional_subbands.append(torch.tensor(filtered, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        return directional_subbands

    def inverse_directional_filter_bank(self, directional_subbands):
        # Inverse Directional Filter Bank (DFB)
        high_pass_combined = torch.zeros_like(directional_subbands[0])
        for ds in directional_subbands:
            high_pass_combined += ds
        return high_pass_combined

    def inverse_nonsubsampled_wavelet_transform(self, low_pass, high_pass):
        # Inverse Non-Subsampled Wavelet Transform (NSWT)
        coeffs = [(low_pass.squeeze().cpu().numpy(), high_pass[i].squeeze().cpu().numpy()) for i in range(self.scales)]
        img_recon = pywt.iswt2(coeffs, wavelet='db1')
        return torch.tensor(img_recon, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def create_gabor_filter(self, theta, sigma=1.0, lamda=1.0, gamma=0.5):
        # Create Gabor filter for given orientation
        sigma_x = sigma
        sigma_y = float(sigma) / gamma
        nstds = 3  # Number of standard deviations
        xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
        xmax = np.ceil(max(1, xmax))
        ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
        ymax = np.ceil(max(1, ymax))
        xmin = -xmax
        ymin = -ymax
        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
        rotx = x * np.cos(theta) + y * np.sin(theta)
        roty = -x * np.sin(theta) + y * np.cos(theta)
        g = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi * rotx / lamda)
        return g