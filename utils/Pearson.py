import torch
from numpy import mean
import numpy as np
from audtorch.metrics.functional import pearsonr


# 读取预测值和真值
pred_sst = np.load(r'E:\Students-21\FengLiu\Deep_Learning_Code\SST_forecasting_FengLiu\experiments\20220520201715\20220520201715-pred.npy') #shape = ((2826, 1, 1, 32, 32))
true_sst = np.load(r'E:\Students-21\FengLiu\Deep_Learning_Code\SST_forecasting_FengLiu\experiments\20220520201715\20220520201715-true.npy')

length = len(pred_sst)

# 转换维度
pred = pred_sst.reshape(length, 1024) # 变成了 （2826，1024）
true = true_sst.reshape(length, 1024)

# 转换为tensor，才可以用permute
pred = torch.from_numpy(pred)
true = torch.from_numpy(true)

pred = pred.permute(1, 0) # 变成（1024，2826）
true = true.permute(1, 0)


total_R = []
for i in range(1024):
    total_R.append(pearsonr(pred[i], true[i]).item())

total_R = np.array(total_R)

print(total_R)  # [0.94888449 0.94791496 0.94256407 ... 0.97277045 0.97214335 0.97290087]

R = mean(total_R)  # 0.9627899621846154
print(R)

print('done')
