import torch

from others.losses import CustomLoss


class TVNorm(CustomLoss):
    """ Total variation norm, or called V^beta norm
        Reference: Understanding Deep Image Representations by Inverting Them (CVPR2015)
        (actually TV norm appears earlier, but this paper seems to be the first to normalize discrete image? I am not sure)
    """
    def __init__(self, do_horizontal=True, do_vertical=True, beta=1):
        """ do_horizontal and do_vertical cannot be both False

        :param do_horizontal: calculate horizontal variation
        :param do_vertical: calculate vertical variation
        :param beta: beta < 1: keep sharpness, beta > 1: remove artefact
        """
        super(TVNorm, self).__init__()
        self.do_horizontal = do_horizontal
        self.do_vertical = do_vertical
        assert do_horizontal or do_vertical, 'do_horizontal and do_vertical cannot be both False'
        self.beta = beta

    def __call__(self, x):
        result = 0
        if self.do_horizontal:
            result += torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        if self.do_vertical:
            result += torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return result ** (self.beta / 2)

    def forward(self, x):
        return self.__call__(x)


class PerimeterAreaRatio(CustomLoss):
    """ Area / (Perimeter ** 2), used to regularize the segmentation prediction (or pseudo label)
        In the original paper, it's used to set a weight for the weak supervised segmentation loss in KD architecture
        Reference: Source free domain adaptation for medical image segmentation with fourier style mining (MIA 2022)
    """
    def __init__(self, epsilon=1e-6):
        super(PerimeterAreaRatio, self).__init__()
        self.epsilon = epsilon

    def forward(self, image):
        # horizontal gradient
        horizontal_gradient = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        # vertical gradient
        vertical_gradient = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
        return torch.sum(image) / (torch.sum(torch.sqrt(horizontal_gradient ** 2 + vertical_gradient ** 2)) + self.epsilon)

