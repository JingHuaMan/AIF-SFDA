import dominate
import numpy as np
import os
import sys

import visdom
from dominate.tags import meta, h3, table, tr, td, p, a, img, br

import utils
from utils import logger
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class WebPageGenerator:
    """This WebPageGenerator class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes

        :param web_dir: a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
        :param title: the webpage name
        :param refresh: how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header (title) to the HTML file

        :param text: the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, path_list, text_list, link_list, width=400):
        """add images to the HTML file

        :param path_list: a list of image paths
        :param text_list: a list of image names shown on the website
        :param link_list: a list of hyperref links; when you click an image, it will redirect you to a new page
        :param width: the images will be resized to width x width
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(path_list, text_list, link_list):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        """save the current content to the HTML file"""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()

    def add_images_in_a_line_with_header(self, prefix, postfix_list, width=256):
        """ Add a sequence of images in a line.
            All images should be saved to <web_dir>/images/ directory beforehand.

        :param prefix: the title and the prefix of all images in the line
        :param postfix_list: a list of postfixes of images
        :param width: the images will be resized to width x width
        """

        self.add_header(prefix)
        image_list, text_list, link_list = [], [], []
        for postfix in postfix_list:
            image_name = '%s_%s.png' % (str(prefix), str(postfix))
            image_list.append(image_name)
            text_list.append(postfix)
            link_list.append(image_name)
        self.add_images(image_list, text_list, link_list, width=width)


class Visualizer:
    """ This class includes several functions that can display/save images and print/save logging information.
        It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """ Initialize the Visualizer class
            Step 1: Cache the training/test options
            Step 2: connect to a visdom server
            Step 3: create an HTML object for saving HTML filters
            Step 4: create a logging file to store training losses

        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.is_train and not opt.no_html
        self.win_size = opt.display_window_size
        self.name = opt.name
        self.saved = False
        self.column_num = opt.display_column_num

        # connect to a visdom server given <display_port> and <display_server>
        if self.display_id > 0:
            def create_visdom_connections():
                return visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env,
                                     raise_exceptions=True)
            try:
                self.vis = create_visdom_connections()
            except ConnectionError:
                self.create_visdom_server()
                self.vis = create_visdom_connections()

        # create an HTML object
        if self.use_html:
            self.web_dir = os.path.join(opt.results_dirname, opt.name, opt.secondary_dirname, opt.phase, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            logger.debug('create web directory %s...' % self.web_dir)
            os.makedirs(self.img_dir, exist_ok=True)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_server(self):
        """ If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.opt.display_port
        logger.info('\n\nCould not connect to Visdom server. \n Trying to start a temporary server....')
        logger.debug('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """ Display current results on visdom; save current results to an HTML file.

        :param visuals: dictionary of images to display or save
        :param epoch: the current epoch
        :param save_result: if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            column_num = self.column_num
            if column_num > 0:  # show all the images in one visdom panel
                column_num = min(column_num, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                # create a table css
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                # create a table of images
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = utils.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % column_num == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                assert len(images) != 0
                white_image = np.ones_like(images[0]) * 255
                while idx % column_num != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=column_num, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_server()

            else:  # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = utils.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_server()

        # save images to an HTML file if they haven't been saved
        if self.use_html and (save_result or not self.saved):
            self.saved = True
            # save images to the disk
            for keys, image in visuals.items():
                image_numpy = utils.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch_%.3d_%s.png' % (epoch, keys))
                utils.save_image(image_numpy, img_path)

            # update website
            webpage = WebPageGenerator(self.web_dir, 'Experiment name = %s' % self.name)
            for n in range(epoch, 0, -1):
                webpage.add_images_in_a_line_with_header('epoch_%.3d' % n, visuals.keys(), width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """ display the current losses on visdom display: dictionary of error labels and values

        :param epoch: current epoch
        :param counter_ratio: progress (percentage) in the current epoch, between 0 and 1
        :param losses: training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        vis_x = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
        vis_y = np.array(self.plot_data['Y'])

        # there is a bug with Visdom, the size of the second dimension could not be 1 if y is a 2-d vector
        try:
            self.vis.line(
                X=vis_x,
                Y=vis_y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'x_label': 'epoch',
                    'y_label': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_server()

    def print_current_losses(self, epoch, iters, losses, time_computation, time_data):
        """ print current losses on console; also save the losses to the disk

        :param epoch: current epoch
        :param iters: current training iteration during this epoch (reset to 0 at the end of every epoch)
        :param losses: training losses stored in the format of (name, float) pairs
        :param time_computation: computational time per data point (normalized by batch_size)
        :param time_data: data loading time per data point (normalized by batch_size)
        """
        message = 'epoch: %d, iteration: %d, computation time: %.3f ms, data loading time: %.3f ms, ' % (
        epoch, iters, 1000 * time_computation, 1000 * time_data)

        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        logger.info(message)
