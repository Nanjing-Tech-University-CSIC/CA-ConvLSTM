import torch
import os
import torch.nn as nn
import numpy as np
import argparse
from datetime import datetime
from models.CA_ConvLSTM import MultiLayerConvLSTM

from Train.trainer import Trainer
from Train.TrainInits import init_seed
from Train.TrainInits import print_model_parameters
from configs.config_3x3_16_3x3_32_3x3_64 import config
# 取消warnings
import warnings
from data.dataloader.get_loader import DualInputDataset
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

DEVICE = 'cuda'

#parser
args = argparse.ArgumentParser(description='arguments')

args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')  # 用于指定在哪一个显卡上运行，默认是cuda：0
args.add_argument('--cuda', default=True, type=bool)
args.add_argument('--seed', default=1234, type=int)
args.add_argument('--batch_size', default=16, type=int)
args.add_argument('--test_batch_size', default=1, type=int)
args.add_argument('--epochs', default=50, type=int) #默认是100
args.add_argument('--lr_init', default=config.lr_init, type=float)
args.add_argument('--early_stop', default=config.early_stop, type=eval)
args.add_argument('--early_stop_patience', default=config.early_stop_patience, type=int)
args.add_argument('--result_dir', default='./experiments/', type=str)
args.add_argument('--model_save_dir', default='./checkpoint/best_models/', type=str)
args.add_argument('--current_time', default='20250512010101', type=str)
args.add_argument('--mae_thresh', default=config.mae_thresh, type=eval)
args.add_argument('--mape_thresh', default=config.mape_thresh, type=float)
args.add_argument('--milestones', default=config.milestones)
args.add_argument('--gamma', default=config.gamma, type=float)
args.add_argument('--modelName', default='CA-ConvLSTM', type=str)



args = args.parse_args() # 解析命令行参数
init_seed(args.seed) # 指定各个随机数种子，目的是为了让实验结果可复现

# 检查cuda是否可用，并且根据device来指定 index 的显卡
if torch.cuda.is_available():
    args.device = 'cuda'
else:
    args.device = 'cpu'


# 模型参数设置
input_dim = 1  # 输入通道数（例如 RGB 图像的 3 通道）
hidden_dim = 64  # 隐藏层通道数
kernel_size = 3  # 卷积核大小
num_layers = 3  # CA-ConvLSTM 层数
output_dim = 1  # 输出通道数（例如预测图像的 3 通道）
# seq_len = 10  # 输入序列的时间步数
# future_steps = 10  # 需要预测的未来时间步数

# 创建模型实例
model = MultiLayerConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, output_dim)
# 假设模型名为 model
model = model.to('cuda')  # 将模型移动到 GPU

args.modelName = 'CA-ConvLSTM'



# 初始化模型，使用的是 Xavier方法
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)

# 打印模型
print_model_parameters(model, only_num=False)

train_dataset = DualInputDataset('train')
print(type(train_dataset))

# 创建验证数据集
valid_dataset = DualInputDataset('valid')
print(type(valid_dataset))

# 创建测试数据集
test_dataset = DualInputDataset('test')
print(type(test_dataset))

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,  num_workers=0, shuffle=True, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,  num_workers=0, shuffle=True, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,  num_workers=0, shuffle=False, pin_memory=True, drop_last=False)

print(type(train_loader))
print(type(valid_loader))
print(type(test_loader))

# 示例：打印第一个样本和标签
print(train_dataset[0])

# loss
loss = torch.nn.L1Loss().cuda()

# optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8)


current_time = datetime.now().strftime('%Y%m%d%H%M%S')
log_name = '{}-{}.log'.format(args.modelName, current_time)
args.current_time = current_time

# 根据训练模型的名字重新创建路径
os.makedirs(args.result_dir + args.current_time + '-' + args.modelName)
args.result_dir = './experiments/{}-{}/'.format(args.current_time, args.modelName)
print(args.result_dir)

# trainer 实例化
trainer = Trainer(model=model,
                  loss=loss,
                  optimizer=optimizer,
                  train_loader=train_loader,
                  val_loader=valid_loader,
                  test_loader=test_loader,
                  log_name=log_name,
                  result_dir=args.result_dir,
                  current_time=args.current_time,
                  scheduler=scheduler,
                  args=args)

# 训练模型
trainer.train()



