import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from data.data_prepare.Normalize_and_reverse_Normalize import SST_minmaxscaler

# 导入数据, 这里以SA——ConvLSTM的训练结果为例
true_path = r'E:\Students-21\FengLiu\Deep_Learning_Code\SST_forecasting_FengLiu\experiments\20220624145508-SA-ConvLSTM\20220624145508-pred.npy'
pred_path = r'E:\Students-21\FengLiu\Deep_Learning_Code\SST_forecasting_FengLiu\experiments\20220624145508-SA-ConvLSTM\20220624145508-true.npy'

# true.shape = pred.shape = (2825, 1, 32, 32)
true_sst = np.load(true_path)
pred_sst = np.load(pred_path)

true = SST_minmaxscaler(true_sst)
pred = SST_minmaxscaler(pred_sst)


print('开始计算相似度,未逆标准化，数据在【0，1】之间：')
for i in range(10):
    print(compare_ssim(true[i][0], pred[i][0]))

print('开始计算相似度,进行逆标准化，数据是真实的温度数据：')
for i in range(10):
    print(compare_ssim(true_sst[i][0], pred_sst[i][0]))