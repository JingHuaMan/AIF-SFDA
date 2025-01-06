import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod

from utils import logger, optimizers, schedulers, lightning_fabric_utils
from utils.metrics import MyMetrics


class BaseModel(ABC):
    """ This class is an abstract base class (ABC) for models.
        To create a subclass, you need to implement the following five functions:
            -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
            -- <set_input>:                     unpack data from dataset and apply preprocessing.
            -- <forward>:                       produce intermediate results.
            -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
            -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """ Initialize the BaseModel class.
            When creating your custom class, you need to implement your own initialization.
            In this function, you should first call <BaseModel.__init__(self, opt)>
            Then, you need to define four lists:
                -- self.loss_names (str list):          specify the training losses that you want to plot and save.
                -- self.model_names (str list):         define networks used in our training.
                -- self.visual_names (str list):        specify the images that you want to display and save.
                -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.

        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        # get device name: CPU or GPU
        if opt.use_lightning_fabric:
            self.device = lightning_fabric_utils.get_device()
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        # save all the checkpoints to save_dir, and the phase here is set to 'train'
        self.save_dir = os.path.join(opt.results_dirname, opt.name, opt.secondary_dirname, 'train', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        # default network structure, used to discard the code warning
        self.net_main = None
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        if not opt.is_train:
            self.metrics = MyMetrics(opt, self.device)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """ Add new model-specific options, and rewrite default values for existing options.

        :param parser: original option parser
        :param is_train: whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        :return the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, data_dict):
        """ Unpack input data from the dataloader and perform necessary pre-processing steps.

        :param data_dict: includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """ Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """ Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def set_optimizers(self):
        """ Create optimizers for all the networks; called at the end of <__init__> function."""
        model_list = [getattr(self, net_name) for net_name in self.model_names]
        self.optimizers = [optimizers.CommonOptimizer(self.opt, model.parameters()) for model in model_list]
        self.schedulers = [schedulers.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

        if self.opt.use_lightning_fabric:
            for model, optimizer in zip(model_list, self.optimizers):
                lightning_fabric_utils.get_fabric().setup(model, optimizer)

    def set_model_dicts(self):
        """ Load the model from the disk, and set the model to the device."""
        # load_epoch and load_path are mutually exclusive
        if self.opt.load_epoch is not None:
            load_path_list = [os.path.join(self.save_dir, '%s_%s_%s_%s.pth' % (
                self.opt.load_epoch, model_name, self.opt.name, self.opt.secondary_dirname)) for model_name in
                              self.model_names]
            self.load_networks(load_path_list)
        elif self.opt.load_path is not None:
            if len(self.opt.load_path) != len(self.model_names):
                raise ValueError('The number of load paths should be the same as the number of models.')
            self.load_networks(self.opt.load_path)

    def setup(self):
        """ Load and print networks; create schedulers"""
        if self.opt.is_train:
            self.set_optimizers()

        self.set_model_dicts()
        self.print_networks()

    def load_networks(self, load_filename_list):
        """ Load all the networks from the disk.

        :param load_filename_list: list of the path (absolute or relative) of the saved model
        """
        for load_filename, model_name in zip(load_filename_list, self.model_names):
            logger.debug('loading the model %s from %s' % (model_name, load_filename))
            net = getattr(self, model_name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            state_dict = torch.load(load_filename, map_location=str(self.device))
            net.load_state_dict(state_dict)

    def print_networks(self):
        """ Print the total number of parameters in the network and network architecture"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                logger.debug(net)
                logger.debug('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))

    def train(self):
        """ Make models train mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self):
        """ Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    @torch.no_grad()
    def test(self):
        """ Forward function used in test time.

            This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
            It also calls <compute_visuals> to produce additional visualization results
        """
        self.forward()
        self.update_metrics()
        self.compute_visuals()

    @abstractmethod
    def compute_visuals(self):
        """ Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def optimizers_zero_grad(self):
        """ Clear the gradients of all optimized variables"""
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def optimizers_step(self):
        """ Update the gradients of all optimized variables"""
        for optimizer in self.optimizers:
            optimizer.step()

    def update_learning_rate(self):
        """ Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        logger.info('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """ Return visualization images. train.py will display these images with visdom, and save the images to an HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """ Return training losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def save_networks(self, prefix):
        """ Save all the networks to the disk.

        :param prefix: the prefix of the file name, usually the epoch during training
        """
        for model_name in self.model_names:
            save_filename = '%s_%s_%s_%s.pth' % (prefix, model_name, self.opt.name, self.opt.secondary_dirname)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, model_name)
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), save_path)
                net.to(device=self.device)
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def backward_loss(self, loss, retain_graph=False):
        """ Calculate gradients for the model; called in <optimize_parameters> function
            Support both the default PyTorch backward() and the Lightning Fabric backward()

        :param loss: the loss to calculate gradients
        :param retain_graph: whether to retain the computational graph
        """
        if self.opt.use_lightning_fabric:
            lightning_fabric_utils.get_fabric().backward(loss, retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """ Set requires_grad=False for all the networks to avoid unnecessary computations

        :param nets: a list of networks
        :param requires_grad: if the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @abstractmethod
    def update_metrics(self):
        """ Update the metric results of the model, used for test/validation phase

            You should call self.metrics.update(preds, target) to update the metric results
        """
        pass

    def get_metric_results(self):
        """ Get the metric results of the model, used for test/validation phase

        :return: a dictionary of metric results, with key as the metric name and value as the metric value
        """
        return self.metrics.compute()
