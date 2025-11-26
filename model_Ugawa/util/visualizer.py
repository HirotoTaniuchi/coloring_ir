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
def save_images(webpage, visuals, image_path, iter, freq, inv_norm, aspect_ratio=1.0, width=256, colorspace='RGB',
                skip_reals=False, skip_4ch=False):
    # 変更: ラベルでディレクトリ判定 / ファイル名からラベル除去 / fake_4ch生成 / skipフラグ対応
    image_dir_other, image_dir_fake_B, image_dir_real_A, image_dir_real_B = webpage.get_image_dir()
    # fake_4ch ディレクトリ生成
    image_dir_fake_4ch = os.path.join(os.path.dirname(image_dir_fake_B), 'fake_4ch')
    os.makedirs(image_dir_fake_4ch, exist_ok=True)

    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    if iter % freq == 0:
        webpage.add_header(name)
    ims, txts, links = [], [], []

    # 保存 (real_A / fake_B / real_B / その他)
    for label, im_data in visuals.items():
        if skip_reals and label in ('real_A', 'real_B'):
            continue
        im_proc = inv_norm(im_data)
        im_proc = change_colorspace_to_rgb(im_proc, colorspace)
        im = util.tensor2im(im_proc)
        image_name = f'{name}.png'  # ラベル除去
        if label == 'fake_B':
            save_path = os.path.join(image_dir_fake_B, image_name)
        elif label == 'real_A':
            save_path = os.path.join(image_dir_real_A, image_name)
        elif label == 'real_B':
            save_path = os.path.join(image_dir_real_B, image_name)
        else:
            save_path = os.path.join(image_dir_other, f'{name}_{label}.png')  # その他は識別のためラベル残す

        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = np.array(Image.fromarray(im).resize((h, int(w * aspect_ratio)), resample=4))
        if aspect_ratio < 1.0:
            im = np.array(Image.fromarray(im).resize((int(h / aspect_ratio), w), resample=4))

        util.save_image(im, save_path)
        if iter % freq == 0:
            # HTML表示用 (その他はラベル付きで示す)
            show_name = image_name if label in ('fake_B', 'real_A', 'real_B') else f'{name}_{label}.png'
            ims.append(show_name)
            txts.append(label)
            links.append(show_name)

    # fake_4ch生成 (fake_B + real_A → RGBA) make4ch.py仕様準拠
    if (not skip_4ch) and ('fake_B' in visuals) and ('real_A' in visuals):
        try:
            fake_B_t = inv_norm(visuals['fake_B'])
            fake_B_t = change_colorspace_to_rgb(fake_B_t, colorspace)
            real_A_t = inv_norm(visuals['real_A'])
            # Tensor → NumPy
            fake_B_np = util.tensor2im(fake_B_t)  # (H,W,3) 期待
            real_A_np = util.tensor2im(real_A_t)  # (H,W,1) または (H,W,3) の可能性あり

            # 形状/チャネル判定と1ch抽出
            single = None
            if real_A_np.ndim == 2:
                single = real_A_np
            elif real_A_np.ndim == 3:
                if real_A_np.shape[2] == 1:
                    single = real_A_np[:, :, 0]
                elif real_A_np.shape[2] == 3:
                    # 3chだが全チャネル同一なら1chとして扱う
                    c0 = real_A_np[:, :, 0]
                    c1 = real_A_np[:, :, 1]
                    c2 = real_A_np[:, :, 2]
                    if np.array_equal(c0, c1) and np.array_equal(c0, c2):
                        single = c0
                    else:
                        print(f'[fake_4ch skip] real_A が3chだが各チャネルが一致しない: {real_A_np.shape}')
                else:
                    print(f'[fake_4ch skip] real_A 不正チャネル数: {real_A_np.shape}')
            else:
                print(f'[fake_4ch skip] real_A 形状不明: {real_A_np.shape}')

            # fake_B の基本検証
            if fake_B_np.ndim != 3 or fake_B_np.shape[2] != 3:
                print(f'[fake_4ch skip] fake_B 不正形状: {fake_B_np.shape}')
            elif single is not None:
                if fake_B_np.shape[0:2] != single.shape[0:2]:
                    print(f'[fake_4ch skip] サイズ不一致: fake_B={fake_B_np.shape} real_A(single)={single.shape}')
                else:
                    # uint8保証 (make4ch.pyと同等)
                    def _to_u8(arr):
                        if arr.dtype == np.uint8:
                            return arr
                        if np.issubdtype(arr.dtype, np.floating):
                            return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                        return np.clip(arr, 0, 255).astype(np.uint8)
                    fake_B_u8 = _to_u8(fake_B_np)
                    single_u8 = _to_u8(single)
                    rgba = np.empty((fake_B_u8.shape[0], fake_B_u8.shape[1], 4), dtype=np.uint8)
                    rgba[:, :, :3] = fake_B_u8[:, :, :3]
                    rgba[:, :, 3] = single_u8
                    out_path_4ch = os.path.join(image_dir_fake_4ch, f'{name}.png')
                    Image.fromarray(rgba, mode='RGBA').save(out_path_4ch)
                    if iter % freq == 0:
                        ims.append(f'{name}.png')
                        txts.append('fake_4ch')
                        links.append(f'{name}.png')
        except Exception as e:
            print(f'[fake_4ch exception] {name}: {e}')

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
        self.epoch_basenames = {}  # 追加: epochごとのbasename保存
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
    def display_current_results(self, visuals, epoch, save_result, colorspace, inv_norm, image_paths=None):  # 追加: image_paths
        # 追加: basename 抽出
        if image_paths and len(image_paths) > 0:
            short_path = ntpath.basename(image_paths[0])
            base_name = os.path.splitext(short_path)[0]
        else:
            base_name = 'unknown'
        self.epoch_basenames[epoch] = base_name

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
                img_path = os.path.join(self.img_dir, f'epoch{epoch:03d}_{label}_{base_name}.png')  # 変更: 保存名にbasename
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []
                base_name_n = self.epoch_basenames.get(n, 'unknown')  # 追加: 過去epochのbasename取得
                for label, image in visuals.items():
                    #image = K.enhance.denormalize(image, 0.5, 0.5)
                    image = inv_norm(image)
                    image = change_colorspace_to_rgb(image, colorspace)
                    image_numpy = util.tensor2im(image)
                    img_path = f'epoch{n:03d}_{label}_{base_name_n}.png'  # 変更: 参照名にbasename
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

    def add_epoch_fakeB_sample(self, fake_B_tensor, image_paths=None):
        """実装途中。現在使用していない。削除してもおけ"""
        # 追加: エポック用 fake_B サンプル登録 (batch先頭のみ)
        try:
            path_name = 'unknown'
            if image_paths and len(image_paths) > 0:
                path_name = os.path.splitext(ntpath.basename(image_paths[0]))[0]
            # shape: (N,C,H,W) を想定 → 先頭
            self._temp_fakeB_samples.append((fake_B_tensor[0].detach().cpu(), path_name))
        except Exception as e:
            print(f'[add_epoch_fakeB_sample skip] {e}')

    def save_epoch_fakeB_grid(self, epoch, inv_norm, colorspace, writer=None):
        """実装途中。現在使用していない。削除してもおけ"""
        # 追加: 最大9枚 → 3x3 グリッド保存 + ソース名テキスト
        if len(self._temp_fakeB_samples) == 0:
            return
        samples = self._temp_fakeB_samples[:9]
        tensors = []
        names = []
        for t, nm in samples:
            try:
                x = inv_norm(t)
                x = change_colorspace_to_rgb(x, colorspace)
                tensors.append(x.unsqueeze(0))  # (1,C,H,W)
                names.append(nm)
            except Exception as e:
                print(f'[fakeB_grid convert skip] {nm}: {e}')
        if len(tensors) == 0:
            self._temp_fakeB_samples.clear()
            return
        batch = torch.cat(tensors, dim=0)  # (K,C,H,W)
        grid = torchvision.utils.make_grid(batch, nrow=3)
        # 保存
        if self.use_html:
            grid_np = util.tensor2im(grid)
            grid_path = os.path.join(self.img_dir, f'epoch{epoch:03d}_fakeB_grid.png')
            util.save_image(grid_np, grid_path)
            list_path = os.path.join(self.img_dir, f'epoch{epoch:03d}_fakeB_grid_sources.txt')
            with open(list_path, 'w', encoding='utf-8') as f:
                for nm in names:
                    f.write(nm + '\n')
        if writer is not None:
            writer.add_image('epoch_fakeB_grid', grid, global_step=epoch)
            writer.flush()
        self._temp_fakeB_samples.clear()
