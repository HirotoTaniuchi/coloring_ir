# ライブラリ
import torch
import torchvision
import numpy as np
import time
from PIL import Image
from skimage.color import lab2rgb
from tqdm import tqdm
import wandb
import datetime
import models.decoder
import models.resnet_for_sal
import os

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# TIC-CGANから
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer, tensorboard_visualize_images, tensorboard_visualize_losses


# main
if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    opt = TrainOptions().parse()
    print ("opt.sal_map",opt.sal_map)
    if opt.sal_map == 'saliency':
        decoder = models.decoder.build_decoder('/home/usrs/ugawa/lab/work/TICCGAN/sal_backbone_original/res_decoder.pth', (256, 256), opt.sal_num_feat, opt.sal_num_feat)
        state_dec = {'state_dict' : decoder.state_dict()}
        torch.save(state_dec, opt.sal_dec_path)
        resnet_p = models.resnet_for_sal.resnet50('/home/usrs/ugawa/lab/work/TICCGAN/sal_backbone_original/res_places.pth')
        state_pla = {'state_dict' : resnet_p.state_dict()}
        torch.save(state_pla, opt.sal_pla_path)
        resnet_im = models.resnet_for_sal.resnet50('/home/usrs/ugawa/lab/work/TICCGAN/sal_backbone_original/res_imagenet.pth')
        state_img = {'state_dict' : resnet_im.state_dict()}
        torch.save(state_img, opt.sal_img_path)

    if opt.tensorboard:
        os.makedirs(opt.tensorboard_dir, exist_ok=True)
        tensorboard_logdir = os.path.join(opt.tensorboard_dir, f"tensorboard_runs/{datetime.date.today()}/{opt.name}")
        writer = SummaryWriter(log_dir=tensorboard_logdir)
    
    # wandb setting
    if opt.wandb:
        wandb.init(project="TIC-Inst-Cycle", entity="rgm401")
        wandb.run.name = opt.name
        wandb.define_metric("epochs")
        wandb.define_metric("*", step_metric="epochs")
    
    # dataloader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    epochs = opt.niter + opt.niter_decay
    num_skips = 0
    
    # train loop
    torch.cuda.empty_cache()
    for epoch in tqdm(range(opt.epoch_count, epochs + 1), desc='Epoch', dynamic_ncols=True,):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # iteration
        for i, data in tqdm(enumerate(dataset), desc='batch', dynamic_ncols=True, total=len(dataset)):
            # 画像がNoneならその画像は処理しない
            if (opt.stage == 'instance' or (opt.need_bbox and opt.stage == 'full')) and data['skip']:
                num_skips += 1
                continue
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # fullの場合はbboxを入力する
            if opt.stage == 'full':
                model.set_input(data, bboxes=data['bbox'])
                model.optimize_parameters(bbox=data['bbox'], skip=data['skip'])
            else:
                model.set_input(data)
                model.optimize_parameters()

        ########################### 画像表示 ###########################
            inv_normalize = transforms.Normalize(mean=[-1, -1, -1],
                                                std=[2, 2, 2])     # mean→-mean/std, std→1/std
            if total_steps % opt.display_freq == 0:
                visuals = model.get_current_visuals()
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(
                    visuals,
                    epoch,
                    save_result,
                    opt.colorspace,
                    inv_norm=inv_normalize,
                    image_paths=data.get('A_paths', None)
                )
                # 追加: fake_B収集
                if 'fake_B' in visuals:
                    visualizer.add_epoch_fakeB_sample(visuals['fake_B'], image_paths=data.get('A_paths', None))
                # Tensorboard
                if opt.tensorboard:
                    tensorboard_visualize_images(writer=writer, visuals=visuals, keys=visuals.keys(), inv_norm=inv_normalize, total_steps=total_steps, batchsize=opt.batchSize, colorspace=opt.colorspace, l1_channel=opt.l1_channel)

            # lossの表示
            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

                # Tensorboard
                if opt.tensorboard:
                    tensorboard_visualize_losses(writer=writer, losses=losses, keys=losses.keys(), total_steps=total_steps)

                # wandb
                if opt.wandb:
                    wandb.log({"loss": losses, "epochs": opt.epoch_count + epochs * total_steps / (dataset_size * epochs)})
                
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
        ###########################################################################################

            if total_steps % opt.save_latest_freq == 0:
                # print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                tqdm.write('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            tqdm.write('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # 追加: エポック終了時にfake_Bグリッド生成・保存
        inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        writer_for_grid = writer if (opt.tensorboard and 'writer' in locals()) else None
        visualizer.save_epoch_fakeB_grid(epoch, inv_norm=inv_normalize, colorspace=opt.colorspace, writer=writer_for_grid)

        tqdm.write('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    writer.close()    
    print(f"Total epochs: {epochs + 1 - opt.epoch_count}")
    print(f"The number of skips [total: {num_skips}] [epoch average: {num_skips/(epochs + 1 - opt.epoch_count)}]")
