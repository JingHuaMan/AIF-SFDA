import torch
from torch import nn

from others.backbones.unet import Unet
from others.baseline_methods import find_model_using_name
from others.frequency_transforms import get_frequency_transform
from others.losses.mi_upper_boundary import CLUB
from others.losses.vector_distance import CosineSimilarity
from utils import multi_class_segmentation_reduce_dim, color_multi_class_label, logger, schedulers, \
    optimizers, lightning_fabric_utils
from . import freeze_net, ema_update
from .base_model import BaseModel


class AIFSFDAModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--frequency_transform', type=str, default='dct', help='frequency transform method')
        parser.add_argument('--norm_layer', type=str, default='InstanceNorm2d',
                            help='normalization layer used in the backbone')
        parser.add_argument('--baseline_name', type=str, default=None, help='baseline model')

        parser.add_argument('--do_filter_smooth', action='store_true', help='whether to smooth the filter')
        parser.add_argument('--filter_smooth_sigma', type=float, default=1., help='sigma for filter smoothing')
        parser.add_argument('--filter_smooth_kernel_size', type=int, default=5, help='kernel size for filter smoothing')

        if is_train:
            parser.add_argument('--segmentation_pretrain_path', type=str, default=None)
            parser.add_argument('--filter_pretrain_path', type=str, default=None)
            parser.add_argument('--alpha_0', type=float, default=1., help='weight for segmentation loss')
            parser.add_argument('--alpha_1', type=float, default=1., help='weight for mi loss')
            parser.add_argument('--alpha_2', type=float, default=1., help='weight for log likelihood loss')
            parser.add_argument('--alpha_3', type=float, default=1., help='weight for consistency loss')
            parser.add_argument('--ema_smooth_factor', type=float, default=0.999, help='smooth factor in EMA procedure')
            parser.add_argument('--label_threshold', type=float, default=0., help='threshold for pseudo label')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = (['loss_seg', 'loss_mi', 'loss_llh', 'loss_con'])
        # teacher model will NOT be loaded to and saved from disk during inferencing
        self.model_names = ['net_filter', 'net_student', 'club']
        self.visual_names = (['image_input', 'label', 'mask', 'frequency_map', 'filter_map', 'frequency_map_filtered',
                              'image_filtered', 'seg_student'] +
                             (['high_confident_map', 'pseudo_label'] if opt.is_train else []))

        self.net_filter = Unet(2 * opt.input_nc if opt.frequency_transform == 'dft' else opt.input_nc, 1, depth=3,
                               last_layer='Sigmoid', norm_layer=getattr(nn, opt.norm_layer)).to(self.device)

        if opt.baseline_name is not None:
            self.net_student = find_model_using_name(opt.baseline_name)(opt.input_nc, opt.output_nc,
                                                                        intermediate_output=True).to(device=self.device)
        else:
            self.net_student = Unet(opt.input_nc, opt.output_nc, last_layer='Identity', output_mode=0,
                                    norm_layer=getattr(nn, opt.norm_layer)).to(device=self.device)

        self.frequency_transform = get_frequency_transform(opt.frequency_transform)()
        self.club = CLUB(1024).to(device=self.device)

        if opt.do_filter_smooth:
            # generate a Gaussian kernel according to the sigma and kernel size
            kernel_1d = torch.arange(opt.filter_smooth_kernel_size) - opt.filter_smooth_kernel_size // 2
            kernel_1d = torch.exp(-kernel_1d ** 2 / (2 * opt.filter_smooth_sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            self.smooth_kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).unsqueeze(0).unsqueeze(0).to(self.device)

        if opt.is_train:
            if opt.baseline_name is not None:
                self.net_teacher = find_model_using_name(opt.baseline_name)(opt.input_nc, opt.output_nc,
                                                                            intermediate_output=True).to(
                    device=self.device)
            else:
                self.net_teacher = Unet(opt.input_nc, opt.output_nc, last_layer='Identity', output_mode=0,
                                        norm_layer=getattr(nn, opt.norm_layer), ).to(device=self.device)
            freeze_net(self.net_teacher)
            self.criterion_ce = torch.nn.BCEWithLogitsLoss() if opt.output_nc == 1 else torch.nn.CrossEntropyLoss()
            self.criterion_con = CosineSimilarity()

    def set_input(self, data_dict):
        self.image_paths = data_dict['source_path']
        self.image_input = data_dict['image_original'].to(self.device)
        self.label = data_dict['label'].to(self.device)
        self.mask = data_dict['mask'].to(self.device)

    def forward(self):
        self.frequency_map = self.frequency_transform(self.image_input)
        self.filter_map = self.net_filter(self.frequency_transform.normalize_frequency_map(self.frequency_map))
        if self.opt.do_filter_smooth:
            self.filter_map = nn.functional.conv2d(self.filter_map, self.smooth_kernel,
                                                   padding=self.smooth_kernel.shape[2] // 2)
        self.frequency_map_filtered = self.frequency_map * self.filter_map
        self.image_filtered = self.frequency_transform(self.frequency_map_filtered, inverse=True) * self.mask
        self.image_filtered = self.image_filtered * self.image_input.mean() / self.image_filtered.mean()

        self.logits_student, *_, self.embedding_student = self.net_student(self.image_filtered)

        if self.opt.is_train:
            self.logits_teacher, *_, self.embedding_teacher = self.net_teacher(self.image_input)
            self.pseudo_logits, self.pseudo_label = multi_class_segmentation_reduce_dim(self.logits_teacher)
            self.pseudo_logits, self.pseudo_label = self.pseudo_logits.detach(), self.pseudo_label.detach().float()

    def compute_visuals(self):
        self.label = color_multi_class_label(self.label, self.opt.output_nc) if self.opt.output_nc > 1 else self.label
        _, self.seg_student = multi_class_segmentation_reduce_dim(self.logits_student)
        self.seg_student = color_multi_class_label(self.seg_student, self.opt.output_nc)
        self.image_filtered = self.image_filtered.clamp(min=0, max=1)
        self.frequency_map = self.frequency_transform.normalize_frequency_map(self.frequency_map, visual=True)
        self.frequency_map_filtered = self.frequency_transform.normalize_frequency_map(self.frequency_map_filtered,
                                                                                       visual=True)

        if self.opt.is_train:
            self.pseudo_label = color_multi_class_label(self.pseudo_label, self.opt.output_nc)

    def optimize_parameters(self):
        def cal_loss_seg():
            self.high_confident_map = self.pseudo_logits > self.opt.label_threshold
            return self.criterion_ce(self.logits_student[self.high_confident_map],
                                     self.pseudo_label[self.high_confident_map])

        self.optimizer_filter.zero_grad()
        self.forward()
        self.loss_seg = cal_loss_seg()
        self.loss_mi = self.club(self.embedding_student, self.embedding_teacher)
        self.backward_loss(self.opt.alpha_0 * self.loss_seg + self.opt.alpha_1 * self.loss_mi)
        self.optimizer_filter.step()

        self.optimizer_student.zero_grad()
        self.forward()
        self.loss_seg = cal_loss_seg()
        self.loss_llh = -self.club.loglikelihood(self.embedding_student, self.embedding_teacher)
        self.loss_con = self.criterion_con(self.embedding_student, self.embedding_teacher)
        self.backward_loss(self.loss_seg + self.opt.alpha_2 * self.loss_llh + self.opt.alpha_3 * self.loss_con)
        self.optimizer_student.step()

        ema_update(self.net_student, self.net_teacher, self.opt.ema_smooth_factor)

    def set_optimizers(self):
        self.optimizer_filter = optimizers.CommonOptimizer(self.opt, self.net_filter.parameters())
        self.optimizer_student = optimizers.CommonOptimizer(self.opt, list(self.net_student.parameters()) + list(
            self.club.parameters()))
        self.optimizers = [self.optimizer_filter, self.optimizer_student]
        self.schedulers = [schedulers.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

        if self.opt.use_lightning_fabric:
            lightning_fabric_utils.get_fabric().setup(self.net_filter, self.optimizer_filter)
            lightning_fabric_utils.get_fabric().setup(self.net_student, self.optimizer_student)
            lightning_fabric_utils.get_fabric().setup(self.club, self.optimizer_student)

    def set_model_dicts(self):
        if self.opt.is_train:
            if self.opt.filter_pretrain_path is not None:
                logger.debug('loading the filter model from %s' % self.opt.filter_pretrain_path)
                self.net_filter.load_state_dict(
                    torch.load(self.opt.filter_pretrain_path, map_location=str(self.device)))
            if self.opt.segmentation_pretrain_path is not None:
                logger.debug('loading the segmentation model from %s' % self.opt.segmentation_pretrain_path)
                segmentation_state_dict = torch.load(self.opt.segmentation_pretrain_path, map_location=str(self.device))
                self.net_student.load_state_dict(segmentation_state_dict)
                self.net_teacher.load_state_dict(segmentation_state_dict)
        else:
            super().set_model_dicts()

    def update_metrics(self):
        self.metrics.update(self.logits_student, self.label)
