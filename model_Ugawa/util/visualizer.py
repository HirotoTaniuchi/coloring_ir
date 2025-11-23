import numpy as np
import torchvision
import torch
import os
import ntpath
import time
from . import util
from . import html
#from scipy.misc import imresize
from PIL import Image
import wandb
import kornia as K
from torchvision import transforms
from util.util import change_colorspace_to_rgb

# my function
# Tensorboardで画像を表示
def tensorboard_visualize_images(writer, visuals, keys, inv_norm, total_steps, batchsize, colorspace, l1_channel, width=256):
    if batchsize==1:    # batchsizeが1のときはlwir fake realの順に並べて表示
        visuals_list = []
        for key in keys:    # 画像ごとに処理
            new_visuals = torch.zeros((visuals[key][0].shape[0], width, width))
            # Lチャネル以外を0にする
            if (colorspace == 'Lab' or colorspace == 'YCbCr') and l1_channel == 'luminance':
                for i in range(visuals[key][0].shape[0]):
                    if i == 0:    # Lチャネルの場合
                        new_visuals[i] = transforms.Resize(width)(visuals[key][0][i].unsqueeze(0))
                    else:    # a b チャネルの場合
                        new_visuals[i] = torch.zeros_like(visuals[key][0][i].unsqueeze(0))
            elif (colorspace == 'HSV' and l1_channel == 'luminance'):
                for i in range(visuals[key][0].shape[0]):
                    if i == 2:    # Vチャネルの場合
                        new_visuals[i] = transforms.Resize(width)(visuals[key][0][i].unsqueeze(0))
                    else:    # H S チャネルの場合
                        new_visuals[i] = torch.zeros_like(visuals[key][0][i].unsqueeze(0))
            else:
                new_visuals = transforms.Resize(width)(visuals[key][0])
            visuals_list.append(new_visuals.clone())
        
        grid = torchvision.utils.make_grid(visuals_list)
        grid = inv_norm(grid)
        grid = change_colorspace_to_rgb(grid, colorspace)
        #if l1_channel == 'luminance':
        #    grid = K.color.rgb_to_grayscale(grid)
        writer.add_image(tag='images', img_tensor=grid, global_step=total_steps)
        writer.flush()
    else:   # batchsize > 1のときはlwir fake realを別々に表示
        for key in keys:
            writer.add_images(tag=key, img_tensor=inv_norm(visuals[key]), global_step=total_steps, dataformats='NCHW')
        writer.flush()

# Tensorboardでロスを表示
def tensorboard_visualize_losses(writer, losses, keys, total_steps):
    for key in keys:
        writer.add_scalars(main_tag='Losses', tag_scalar_dict={key:losses[key]}, global_step=total_steps)
    writer.flush()

# wandbで画像を表示 
def wandb_visualize_images(visuals):
    images_dict = {}
    for key in visuals.keys():
        np_image = np.transpose(visuals[key].to('cpu').detach().numpy().copy()[0], (1, 2, 0))
        images_dict[key] = wandb.Image(np_image)
    wandb.log(images_dict)

# created by the guy who writes original codes
# save image to the disk
def save_images(webpage, visuals, image_path, iter, freq, inv_norm, aspect_ratio=1.0, width=256, colorspace='RGB'):
    inv_norm = inv_norm    # transforms.Normalize(mean=[-1, -1, -1],std=[2, 2, 2])
    image_dir_other, image_dir_fake_B, image_dir_real_A, image_dir_real_B = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    if iter % freq == 0:
        webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im_data = inv_norm(im_data)
        im_data = change_colorspace_to_rgb(im_data, colorspace)
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        if 'fake_B' in image_name:
            save_path = os.path.join(image_dir_fake_B, image_name)
        elif 'real_A' in image_name:
            save_path = os.path.join(image_dir_real_A, image_name)
        elif 'real_B' in image_name:
            save_path = os.path.join(image_dir_real_B, image_name)
        else:
            save_path = os.path.join(image_dir_other, image_name)

        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            #im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            im = np.array(Image.fromarray(im).resize((h, int(w * aspect_ratio)), resample=4))

        if aspect_ratio < 1.0:
            #im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            im = np.array(Image.fromarray(im).resize((int(h / aspect_ratio), w), resample=4))

        util.save_image(im, save_path)
        if iter % freq == 0:
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
    if iter % freq == 0:
        webpage.add_images(ims, txts, links, width=width)
    
# save image to the disk
def save_images2(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims = []
    txts = []
    links = []
    #print(visuals['real_A']) 
    image_name = '%s.png' % name
    save_path1 = os.path.join(image_dir, 'input/',image_name)
    save_path2 = os.path.join(image_dir, 'output/', image_name)
    save_path3 = os.path.join(image_dir, 'target/', image_name)
    util.save_image(util.tensor2im(visuals['real_A']), save_path1)
    util.save_image(util.tensor2im(visuals['fake_B']), save_path2)
    util.save_image(util.tensor2im(visuals['real_B']), save_path3)

class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, raise_exceptions=True, env=opt.display_env)
            
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result, colorspace, inv_norm):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    #image = K.enhance.denormalize(image, 0.5, 0.5)
                    image = inv_norm(image)
                    image = change_colorspace_to_rgb(image, colorspace)
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except ConnectionError:
                    print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
                    exit(1)

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                #image = K.enhance.denormalize(image, 0.5, 0.5)
                image = inv_norm(image)
                image = change_colorspace_to_rgb(image, colorspace)
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []
                for label, image in visuals.items():
                    #image = K.enhance.denormalize(image, 0.5, 0.5)
                    image = inv_norm(image)
                    image = change_colorspace_to_rgb(image, colorspace)
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
