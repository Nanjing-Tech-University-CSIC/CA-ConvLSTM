import torch
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset,DataLoader
# lin/SCS_SST_SSH_DATA/sst_Y.npy

scs_datas_path = '../../SCS_SST_SSH_DATA/sst_X.npy'
scs_labels_path = '../../SCS_SST_SSH_DATA/sst_Y.npy'

scsSSH_datas_path = '../../SCS_SST_SSH_DATA/ssh_X.npy'
scsSSH_labels_path = '../../SCS_SST_SSH_DATA/ssh_Y.npy'
class DualInputDataset(Dataset):
    def __init__(self, split):
        super(DualInputDataset, self).__init__()

        # 加载第一组数据 X 和 Y
        self.scs_datas = np.load(scs_datas_path)  # (10207, 10, 1, 64, 64)
        self.scs_targets = np.load(scs_labels_path)

        # 加载第二组数据 X 和 Y
        self.scsSSH_datas = np.load(scsSSH_datas_path)
        self.scsSSH_targets = np.load(scsSSH_labels_path)

        if split == 'train':
            self.scs_datas = self.scs_datas[:7144]
            self.scs_targets = self.scs_targets[:7144]
            self.scsSSH_datas = self.scsSSH_datas[:7144]
            self.scsSSH_targets = self.scsSSH_targets[:7144]
        elif split == 'valid':
            self.scs_datas = self.scs_datas[7144:8164]
            self.scs_targets = self.scs_targets[7144:8164]
            self.scsSSH_datas = self.scsSSH_datas[7144:8164]
            self.scsSSH_targets = self.scsSSH_targets[7144:8164]
        else:
            self.scs_datas = self.scs_datas[8164:]
            self.scs_targets = self.scs_targets[8164:]
            self.scsSSH_datas = self.scsSSH_datas[8164:]
            self.scsSSH_targets = self.scsSSH_targets[8164:]

        print('Loaded SCS-sst sstdatas: {}'.format(len(self.scs_datas)))
        print('Loaded SCS-sst targets: {}'.format(len(self.scs_targets)))
        print('Loaded SCS-SSH datas: {}'.format(len(self.scsSSH_datas)))
        print('Loaded SCS-SSH targets: {}'.format(len(self.scsSSH_targets)))

    def __getitem__(self, index):
        scs_input = torch.from_numpy(self.scs_datas[index])  # [10,1,64,64]
        scs_target = torch.from_numpy(self.scs_targets[index])  # [10,1,64,64]
        scsSSH_input = torch.from_numpy(self.scsSSH_datas[index])
        scsSSH_target = torch.from_numpy(self.scsSSH_targets[index])

        return scs_input, scs_target, scsSSH_input, scsSSH_target

    def __len__(self):
        return len(self.scs_datas)
class ALLSCSDataset(Dataset):
    def __init__(self, args, split):

        super(ALLSCSDataset, self).__init__()

        # 加载X
        self.datas = np.load(scs_datas_path)   # (10207, 10, 1, 64, 64)
        # 加载Y
        self.targets = np.load(scs_labels_path)

        if split == 'train':
            # 取训练数据
            self.datas = self.datas[:7144]
            self.targets = self.targets[:7144]


        elif split == 'valid':
            # 取验证集数据
            self.datas = self.datas[7144:8164]
            self.targets = self.targets[7144:8164]

        else:
            # 取测试集数据
            self.datas = self.datas[8164:]
            self.targets = self.targets[8164:]

        print('Loaded X:{}'.format(len(self.datas)))
        print('Loaded Y:{}'.format(len(self.targets)))
        # 打印输出数据点的数目
        print('Loaded {} {} samples'.format(self.__len__(), split))

    def __getitem__(self, index):

        inputs = torch.from_numpy(self.datas[index])     # [10,1,64,64]
        targets = torch.from_numpy(self.targets[index])  # [10,1,64,64]
        return inputs,targets

    def __len__(self):
        # 返回数据集的长度
        return self.datas.shape[0]

# 输入样本集合X和对对应的标签集合Y，返回可用于迭代的dataloader
def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False

    # 将 X,Y 由np.array 转换为 Tensor格式
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)


    # TensorFloat = torch.cuda.FloatTensor
    # X, Y = TensorFloat(X), TensorFloat(Y)  # 将 X，Y 转换为  TensorFloat 格式

    data = torch.utils.data.TensorDataset(X, Y)
    """
    torch.utils.data.TensorDataset(X,Y)，这里的X是样本，Y是X对应的label
        对给定的tensor数据(样本和标签)，将它们包装成dataset。注意，如果是numpy的array，或者Pandas的DataFrame需要先转换成Tensor。
    """
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    """
    torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=shuffle, drop_last=drop_last)
    data 是上一步刚刚制作的样本X和Y的组合。batch_size 表示每一批有多少个样本。drop_last：如果数据集大小不能被batch size整除，
    则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)
    数据加载器，组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。它可以对我们上面所说的数据集Dataset作进一步的设置.
    """
    return dataloader
