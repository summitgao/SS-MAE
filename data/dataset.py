import torch
import random
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def min_max(x):
    min = np.min(x)
    max = np.max(x)
    return (x - min) / (max - min)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def applyPCA(data, n_components):
    h, w, b = data.shape
    pca = PCA(n_components=n_components)
    data = np.reshape(pca.fit_transform(np.reshape(data, (-1, b))), (h, w, -1))
    return data


class HXDataset(Dataset):

    def __init__(self, hsi, hsi_pca, X, pos, windowSize, gt=None, transform=None, train=False):
        modes = ['symmetric', 'reflect']
        self.train = train
        self.pad = windowSize // 2
        self.windowSize = windowSize

        self.hsi = np.pad(hsi, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.hsi_pca = np.pad(hsi_pca, ((self.pad, self.pad),
                                        (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.X = None
        if (len(X.shape) == 2):
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad)), mode=modes[windowSize % 2])
        elif (len(X.shape) == 3):
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])

        self.pos = pos
        self.gt = None
        if gt is not None:
            self.gt = gt
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        h, w = self.pos[index, :]
        hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize]
        hsi_pca = self.hsi_pca[h: h + self.windowSize, w: w + self.windowSize]
        X = self.X[h: h + self.windowSize, w: w + self.windowSize]
        if self.transform:
            hsi = self.transform(hsi).float()
            hsi_pca = self.transform(hsi_pca).float()
            X = self.transform(X).float()
            trans = [transforms.RandomHorizontalFlip(1.),
                     transforms.RandomVerticalFlip(1.)]
            if self.train:
                if random.random() < 0.5:
                    i = random.randint(0, 1)
                    hsi = trans[i](hsi)
                    X = trans[i](X)
                    hsi_pca = trans[i](hsi_pca)
        if self.gt is not None:
            gt = torch.tensor(self.gt[h, w] - 1).long()
            return hsi.unsqueeze(0), X, gt, hsi_pca.unsqueeze(0)
        return hsi.unsqueeze(0), X, h, w, hsi_pca.unsqueeze(0)

    def __len__(self):
        return self.pos.shape[0]


def getData(hsi_path, X_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers, args):
    '''
    hsi: Hyperspectral image data
    X: Other modal data
    gt: Ground truth labels, where 0 represents unlabeled
    train_index: Indices for training data
    test_index: Indices for testing data
    pretrain_index: Indices for pretraining data
    trntst_index: Indices for training and testing data, used for visualizing labeled data
    all_index: Indices for all data, including unlabeled data, used for visualizing all data or pretraining

    '''

    hsi = loadmat(hsi_path)[keys[0]]
    X = loadmat(X_path)[keys[1]]
    gt = loadmat(gt_path)[keys[2]]
    train_index = loadmat(index_path)[keys[3]]
    test_index = loadmat(index_path)[keys[4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)

    all_index = loadmat(index_path)[keys[5]]
    if args.is_pretrain:
        np.random.shuffle(all_index)
        pretrain_index = np.zeros((args.pretrain_num, 2), dtype=np.int32)
        count = 0
        for i in all_index:
            # print(i)
            if 15 < i[0] < hsi.shape[0] - 15:
                if 15 < i[1] < hsi.shape[1] - 15:
                    pretrain_index[count] = i
                    count += 1
                    if count == args.pretrain_num:
                        break

    X = min_max(X)
    hsi = min_max(hsi)
    print(X.shape)
    print(hsi.shape)
    # PCA is used to reduce the dimensionality of the HSI
    hsi_pca = applyPCA(hsi, channels)

    # Build Dataset
    HXpretrainset = HXDataset(hsi, hsi_pca, X, pretrain_index,
                              windowSize, transform=ToTensor(), train=True)
    HXtrainset = HXDataset(hsi, hsi_pca, X, train_index,
                           windowSize, gt, transform=ToTensor(), train=True)
    HXtestset = HXDataset(hsi, hsi_pca, X, test_index,
                          windowSize, gt, transform=ToTensor())
    HXtrntstset = HXDataset(hsi, hsi_pca, X, trntst_index,
                            windowSize, transform=ToTensor())
    HXallset = HXDataset(hsi, hsi_pca, X, all_index,
                         windowSize, transform=ToTensor())

    # Build Dataloader
    pretrain_loader = DataLoader(
        HXpretrainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    train_loader = DataLoader(
        HXtrainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(
        HXtestset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    trntst_loader = DataLoader(
        HXtrntstset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=True)
    all_loader = DataLoader(
        HXallset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    print("Success!")
    return pretrain_loader, train_loader, test_loader, trntst_loader, all_loader


def getHouston2018Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers, args):
    print("Houston2018!")
    # Houston2018 mat keys
    houston2018_keys = ['houston_hsi', 'houston_lidar', 'houston_gt', 'houston_train', 'houston_test', 'houston_all']

    return getData(hsi_path, lidar_path, gt_path, index_path, houston2018_keys, channels, windowSize, batch_size,
                   num_workers, args)


def getBerlinData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers, args):
    print("Berlin!")
    # Berlin mat keys
    berlin_keys = ['berlin_hsi', 'berlin_sar', 'berlin_gt', 'berlin_train', 'berlin_test', 'berlin_all']

    return getData(hsi_path, sar_path, gt_path, index_path, berlin_keys, channels, windowSize, batch_size, num_workers,
                   args)


def getAugsburgData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers, args):
    print("Augsburg!")
    # Augsburg mat keys
    augsburg_keys = ['augsburg_hsi', 'augsburg_sar', 'augsburg_gt', 'augsburg_train', 'augsburg_test', 'augsburg_all']

    return getData(hsi_path, sar_path, gt_path, index_path, augsburg_keys, channels, windowSize, batch_size,
                   num_workers, args)


def getHSData(datasetType, channels, windowSize, batch_size=64, num_workers=0, args=None):
    if (datasetType == "Berlin"):
        hsi_path = "data/Berlin/berlin_hsi.mat"
        sar_path = "data/Berlin/berlin_sar.mat"
        gt_path = "data/Berlin/berlin_gt.mat"
        index_path = "data/Berlin/berlin_index.mat"
        pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = getBerlinData(hsi_path, sar_path, gt_path, index_path,
                                                                             channels, windowSize, batch_size,
                                                                             num_workers, args)
    elif (datasetType == "Augsburg"):
        hsi_path = "data/Augsburg/augsburg_hsi.mat"
        sar_path = "data/Augsburg/augsburg_sar.mat"
        gt_path = "data/Augsburg/augsburg_gt.mat"
        index_path = "data/Augsburg/augsburg_index.mat"
        pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = getAugsburgData(hsi_path, sar_path, gt_path, index_path,
                                                                               channels,
                                                                               windowSize, batch_size, num_workers,
                                                                               args)
    elif (datasetType == "Houston2018"):
        hsi_path = "data/Houston2018/houston_hsi.mat"
        lidar_path = "data/Houston2018/houston_lidar.mat"
        gt_path = "data/Houston2018/houston_gt.mat"
        index_path = "data/Houston2018/houston_index.mat"
        pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = getHouston2018Data(hsi_path, lidar_path, gt_path,
                                                                                  index_path,
                                                                                  channels, windowSize, batch_size,
                                                                                  num_workers, args)

    return pretrain_loader, train_loader, test_loader, trntst_loader, all_loader
