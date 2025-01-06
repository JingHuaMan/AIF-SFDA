import torch
from torch import nn

import others
from others.backbones.unet import Unet
from others.baseline_methods import find_model_using_name
from utils import multi_class_segmentation_reduce_dim, color_multi_class_label
from .base_model import BaseModel


class PRETRAINUNETModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--norm_layer', type=str, default='InstanceNorm2d',
                            help='normalization layer used in the backbone')
        parser.add_argument('--baseline_name', type=str, default=None, help='baseline model')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['loss_main']
        self.model_names = ['net_main']
        self.visual_names = ['image_input', 'label', 'out_seg']

        if opt.baseline_name is not None:
            self.net_main = find_model_using_name(opt.baseline_name)(opt.input_nc, opt.output_nc).to(device=self.device)
        else:
            self.net_main = Unet(opt.input_nc, opt.output_nc, last_layer='Identity',
                                 norm_layer=getattr(nn, opt.norm_layer)).to(device=self.device)
        # self.net_main = Unet(opt.input_nc, opt.output_nc, last_layer='Identity', norm_layer=nn.InstanceNorm2d).to(device=self.device)
        # self.net_main = Unet(opt.input_nc, opt.output_nc, last_layer='Identity').to(device=self.device)

        if self.opt.is_train:
            # define loss functions
            self.criterion_segmentation = torch.nn.BCEWithLogitsLoss() if opt.output_nc == 1 else torch.nn.CrossEntropyLoss()

    def set_input(self, data_dict):
        self.image_paths = data_dict['source_path']

        self.image_input = data_dict['image_original'].to(self.device)
        self.label = data_dict['label'].to(self.device)
        self.mask = data_dict['mask'].to(self.device)

    def forward(self):
        self.out_seg = self.net_main(self.image_input)

    def compute_visuals(self):
        _, self.out_seg = multi_class_segmentation_reduce_dim(self.out_seg)
        if self.opt.output_nc > 1:
            self.out_seg = color_multi_class_label(self.out_seg, self.opt.output_nc)

    def optimize_parameters(self):
        self.forward()
        self.optimizers_zero_grad()
        self.loss_main = self.criterion_segmentation(self.out_seg, self.label)
        self.loss_main.backward()
        self.optimizers_step()

    def update_metrics(self):
        self.metrics.update(torch.sigmoid(self.out_seg), self.label)  # TODO
