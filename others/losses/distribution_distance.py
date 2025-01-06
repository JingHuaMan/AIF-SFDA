import torch

from others.losses import CustomLoss


class MaximumMeanDiscrepancy(CustomLoss):
    def __init__(self, kernel='rbf', kernel_mul=2.0, kernel_num=5):
        super(MaximumMeanDiscrepancy, self).__init__()
        self.kernel = kernel
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def forward(self, x, y):
        x_num = x.size(0)
        xy = torch.cat([x, y], 0)
        xy = xy.view(xy.size(0), -1)
        if self.kernel == 'rbf':
            delta = xy.unsqueeze(0) - xy.unsqueeze(1)
            delta = delta.pow(2).sum(2)
            bandwidth = []
            for i in range(self.kernel_num):
                bandwidth.append(delta.median() * (self.kernel_mul ** (self.kernel_num // 2 - i)))
            bandwidth = torch.stack(bandwidth)
            bandwidth = bandwidth.view(1, 1, 1, 1, self.kernel_num)
            delta = - delta.unsqueeze(4) / bandwidth
            exp_delta = (- delta.abs().exp()).sum(4)
            mmd = (exp_delta[:x_num, :x_num] + exp_delta[x_num:, x_num:] - exp_delta[:x_num, x_num:] - exp_delta[x_num:, :x_num]).mean()
        else:
            raise ValueError('no such kernel %s' % self.kernel)
        return mmd


class JSDivergence(CustomLoss):
    def __init__(self):
        super(JSDivergence, self).__init__()

    def forward(self, x, y):
        m = 0.5 * (x + y)
        js = 0.5 * (torch.nn.functional.kl_div(x, m) + torch.nn.functional.kl_div(y, m))
        return js


class BiDirectionalKLDivergence(CustomLoss):
    def __init__(self):
        super(BiDirectionalKLDivergence, self).__init__()

    def forward(self, x, y):
        kld = torch.nn.functional.kl_div(x, y) + torch.nn.functional.kl_div(y, x)
        return kld



