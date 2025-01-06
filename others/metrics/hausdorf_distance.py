import torch
import torch.nn.functional as F
from torchmetrics import Metric


class HausdorffDistance2D(Metric):
    def __init__(self, task='binary', num_classes=None, ignore_index=None, threshold=None, average='macro',
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.task = task
        self.num_classes = num_classes if task == 'multiclass' else 1
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.average = average
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("samples_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("samples_count", default=torch.tensor(0), dist_reduce_fx="sum")

        # Define Sobel kernels
        self.sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float()
        self.sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float()

    def _get_border(self, mask):
        device = mask.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        gx = F.conv2d(mask.float().unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
        gy = F.conv2d(mask.float().unsqueeze(0).unsqueeze(0), sobel_y, padding=1)
        g = torch.sqrt(gx ** 2 + gy ** 2)
        return (g > 0).float().squeeze()

    def _hausdorff_distance(self, set1, set2):
        set1 = set1.float()
        set2 = set2.float()
        d_matrix = torch.cdist(set1.unsqueeze(0), set2.unsqueeze(0), p=2).squeeze()
        return max(d_matrix.min(0)[0].max().item(), d_matrix.min(1)[0].max().item())

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.task == 'binary':
            self._update_binary(preds, target)
        elif self.task == 'multiclass':
            self._update_multiclass(preds, target)
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

    def _update_binary(self, preds, target):
        if self.threshold is not None:
            preds = (preds > self.threshold).float()

        b, c, w, h = preds.shape
        for i in range(b):
            for j in range(c):
                pred = preds[i, j]
                tgt = target[i, j]
                pred_border = self._get_border(pred)
                tgt_border = self._get_border(tgt)
                pred_points = torch.nonzero(pred_border, as_tuple=False).float()
                tgt_points = torch.nonzero(tgt_border, as_tuple=False).float()
                if pred_points.numel() > 0 and tgt_points.numel() > 0:
                    hausdorff_dist = self._hausdorff_distance(pred_points, tgt_points)
                    self.sum += hausdorff_dist
                    self.count += 1
                    self.samples_sum += hausdorff_dist
                    self.samples_count += 1

    def _update_multiclass(self, preds, target):
        b, c, w, h = preds.shape
        for i in range(b):
            for j in range(c):
                if self.threshold is not None:
                    pred = (preds[i, j] > self.threshold).float()
                else:
                    pred = preds[i, j]

                tgt = target[i, j]
                for cls in range(self.num_classes):
                    if cls == self.ignore_index:
                        continue
                    pred_class = (pred == cls).float()
                    tgt_class = (tgt == cls).float()
                    pred_border = self._get_border(pred_class)
                    tgt_border = self._get_border(tgt_class)
                    pred_points = torch.nonzero(pred_border, as_tuple=False).float()
                    tgt_points = torch.nonzero(tgt_border, as_tuple=False).float()
                    if pred_points.numel() > 0 and tgt_points.numel() > 0:
                        hausdorff_dist = self._hausdorff_distance(pred_points, tgt_points)
                        self.sum += hausdorff_dist
                        self.count += 1
                        self.samples_sum += hausdorff_dist
                        self.samples_count += 1

    def compute(self):
        if self.average == 'macro':
            return (self.sum / self.count) if self.count > 0 else torch.tensor(0.0)
        elif self.average == 'micro':
            return (self.sum / self.count) if self.count > 0 else torch.tensor(0.0)
        elif self.average == 'samples':
            return (self.samples_sum / self.samples_count) if self.samples_count > 0 else torch.tensor(0.0)
        else:
            raise ValueError(f"Unknown average type: {self.average}")

# Example usage:
# Binary case:
# preds = torch.rand(4, 1, 256, 256) > 0.5
# target = torch.rand(4, 1, 256, 256) > 0.5
# metric = HausdorffDistance2D(task='binary', threshold=0.5)
# metric.update(preds, target)
# result = metric.compute()
# print(f"Binary Hausdorff Distance: {result}")

# Multi-class case:
# num_classes = 3
# preds = torch.randint(0, num_classes, (4, 1, 256, 256))
# target = torch.randint(0, num_classes, (4, 1, 256, 256))
# metric = HausdorffDistance2D(task='multiclass', num_classes=num_classes, ignore_index=0, threshold=0.5, average='samples')
# metric.update(preds, target)
# result = metric.compute()
# print(f"Multi-Class Hausdorff Distance: {result}")
