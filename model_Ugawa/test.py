import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.lab2rgb import lab2rgb
from util.visualizer import save_images
from util.visualizer import save_images2
from util import html
from torchvision import transforms

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)

    # 結果を保存するディレクトリ名前に付帯情報を追加
    if 'day' in opt.dataroot_A:
        day_or_night = 'day_'
    elif 'video_and_night' in opt.dataroot_A:
        day_or_night = 'video_and_night_'
    elif 'night' in opt.dataroot_A:
        day_or_night = 'night_'
    else:
        day_or_night = 'other_'

    option = ''
    if opt.clahe:
        option += '_clahe'
    if opt.A_invert and opt.A_invert_prob == 1:
        option += '_Ainv'
    if 'FLIR_ADAS_1_3_TICCGAN' in opt.dataroot_A:
        option += '_YOLO'

    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{day_or_night}{opt.which_epoch}{option}')
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    freq = 1
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        # print(data['A'].size())
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 10 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        inv_normalize = transforms.Normalize(mean=[-1, -1, -1],std=[2, 2, 2]) 
        save_images(webpage, visuals, img_path, iter=i, freq=freq, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, colorspace=opt.colorspace, inv_norm=inv_normalize)

    webpage.save()
