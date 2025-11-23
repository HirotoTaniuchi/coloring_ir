import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        if opt.stage == 'global':
            from data.aligned_dataset import AlignedDataset
            dataset = AlignedDataset()

    elif opt.dataset_mode == 'unaligned':

        if opt.stage == 'global':
            from data.unaligned_dataset import Global_Dataset_TIR
            dataset = Global_Dataset_TIR()
        elif opt.stage == 'instance':
            from data.unaligned_dataset import Instance_Dataset_TIR
            dataset = Instance_Dataset_TIR()
        elif opt.stage == 'full':
            from data.unaligned_dataset import Full_Dataset_TIR
            dataset = Full_Dataset_TIR()
        else:
            raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
        
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data
