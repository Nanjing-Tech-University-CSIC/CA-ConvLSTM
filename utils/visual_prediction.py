import numpy as np
import matplotlib.pyplot as plt


def get_pred_and_true_figure(true, pred, start_day, end_day, save_path, longitude, latitude):
    """
    将预测结果中真实值进进行比较，绘制变化曲线图
    :param true:  传入的真值
    :param pred: 传入的预测值
    :param day_index: 需要绘制的天数索引
    :param save_path: 对比的的保存地址
    :param longitude: 可视化经纬度索引
    :param latitude: 可视化经纬度索引
    :return:
    """
    x_index = range(start_day, end_day) # X轴的取值范围

    # 计算某个地点的平均误差
    sst_day_error = 0
    temp = abs(true - pred) # 将预测值和真正值相减，并求绝对值
    error_list = temp.reshape((len(true), 32, 32))
    for k in range(start_day, end_day):
        sst_day_error += error_list[k, longitude, latitude]
    average_error = sst_day_error / (end_day - start_day)  # 计算平均误差
    # 将不用的列表置空
    temp = []
    error_list = []

    plt.cla() # 每次画图之前，清空画布

    # 设置图像的长宽
    plt.style.use('bmh')
    plt.style.available
    plt.figure(figsize=(16, 5))

    plt.title('location:({},{}),Average Error:{:.2f}K'.format(longitude, latitude, average_error))
    # 需要传入的数据格式是 (2826, 10, 64, 64)
    plt.plot(x_index, true[start_day:end_day, 0, longitude, latitude], color='blue', label='True')
    plt.plot(x_index, pred[start_day:end_day, 0, longitude, latitude], color='red', label='Prediction')
    plt.legend(loc=0)  # 设置最适应的图例
    plt.xlabel('Date/day')
    plt.ylabel('Temperature/K')
    plt.savefig(save_path + '({},{})_SST'.format(longitude, latitude))



# ConvLSTM的结果可视化
pred_sst_ConvLSTM = np.load(r'E:\Students-21\FengLiu\Deep_Learning_Code\SST_forecasting_FengLiu\experiments\20221026214329-ConvLSTM\reverse_total_pred_sst.npy')
true_sst_ConvLSTM = np.load(r'E:\Students-21\FengLiu\Deep_Learning_Code\SST_forecasting_FengLiu\experiments\20221026214329-ConvLSTM\reverse_total_true_sst.npy')

# SA-ConvLSTM的结果可视化
# pred_sst_SA_ConvLSTM = np.load(r'E:\Students-21\FengLiu\Deep_Learning_Code\SST_forecasting_FengLiu\experiments\20220630133152-TrajGRU\first_day_pred_sst.npy')
# true_sst_SA_ConvLSTM = np.load(r'E:\Students-21\FengLiu\Deep_Learning_Code\SST_forecasting_FengLiu\experiments\20220630133152-TrajGRU\first_day_true_sst.npy')


location = [0, 5, 10, 15, 20, 25, 30] # 将（0,0） (5,5) ....(30,30)对于点的温度进行可视化
for i in range(len(location)):
    get_pred_and_true_figure(true=pred_sst_ConvLSTM,
                             pred=true_sst_ConvLSTM,
                             start_day=0,
                             end_day=400,
                             save_path='E:/Students-21/FengLiu/Deep_Learning_Code/SST_forecasting_FengLiu/experiments/20221026214329-ConvLSTM/SST_visual/',
                             longitude=location[i],
                             latitude=location[i])

# 获取（16，16）的整体趋势
# get_pred_and_true_figure(true=true_sst,
#                          pred=pred_sst,
#                          day_index=2800,
#                          save_path='D:/Code/Pycharm/SST_forecasting_FengLiu/experiments/20220520201715/SST_visual/',
#                          longitude=16,
#                          latitude=16)