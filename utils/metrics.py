from functools import partial

import torch
import torchmetrics

from others.metrics.hausdorf_distance import HausdorffDistance2D

metrics_name_cls_map = {
    'iou': torchmetrics.JaccardIndex,
    'f1': torchmetrics.F1Score,
    'accuracy': torchmetrics.Accuracy,
    'auroc': torchmetrics.AUROC,
    'aur': torchmetrics.AUROC,
    'dice': torchmetrics.Dice,
    'mcc': torchmetrics.MatthewsCorrCoef,
    'hausdorff': HausdorffDistance2D
}

# metrics that have the attribute "task" should be treated differently
list_have_attr_task = ['iou', 'f1', 'accuracy', 'auroc', 'aur', 'mcc', 'hausdorff']


class MyMetrics(torchmetrics.Metric):
    """ MyMetrics is a wrapper for torchmetrics.Metric.
        It can calculate multiple metrics according to the options,
        and can calculate the standard variance of the metrics.
    """

    def __init__(self, opt, device):
        super().__init__()
        self.metrics_list = [x.lower() for x in opt.metrics_list]
        self.result_dict = {}
        for metrics_name in self.metrics_list:
            if not metrics_name in metrics_name_cls_map:
                raise NotImplementedError('Metric ' + metrics_name + ' is not implemented.')

            metrics_cls = metrics_name_cls_map[metrics_name]
            if metrics_name in list_have_attr_task:
                # in this selection branch, all metrics have the attribute "task"
                metrics_part = partial(metrics_cls, task='binary') \
                    if opt.output_nc == 1 else partial(metrics_cls, task='multiclass', num_classes=opt.output_nc)
            else:
                metrics_part = partial(metrics_cls, ignore_index=0 if opt.output_nc == 1 else None)
            if metrics_name in ['aur', 'auroc', 'auc']:
                self.result_dict[metrics_name] = metrics_part(average='samples').to(device=device)
            else:
                self.result_dict[metrics_name] = metrics_part(threshold=opt.metrics_threshold, average='samples').to(device=device)

        self.require_std = opt.metrics_calculate_std_var
        if self.require_std:
            for metrics_name in self.metrics_list:
                self.add_state(metrics_name + '_list', default=torch.tensor([]), dist_reduce_fx=None)
        self.to(device)

    def update(self, preds, target):
        target = target.int()
        if self.require_std:
            for metrics_name in self.metrics_list:
                metrics = self.result_dict[metrics_name]
                metrics.update(preds, target)
                metrics_list = getattr(self, metrics_name + '_list')
                metrics_list = torch.cat([metrics_list, metrics.compute().unsqueeze(0)])
                setattr(self, metrics_name + '_list', metrics_list)
                metrics.reset()
        else:
            for metrics_name, metrics in self.result_dict.items():
                metrics.update(preds, target)

    def compute(self):
        if self.require_std:
            result = {}
            for metrics_name in self.metrics_list:
                metrics_list = getattr(self, metrics_name + '_list')
                result[metrics_name + '_mean'] = metrics_list.mean().item()
                result[metrics_name + '_std'] = metrics_list.std().item()
            return result
        result = {}
        for metrics_name, metrics in self.result_dict.items():
            # print(metrics.compute().shape)
            result[metrics_name] = metrics.compute().item()
        return result
