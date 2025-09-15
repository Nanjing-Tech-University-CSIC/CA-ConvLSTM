import logging
from datetime import datetime
import torch
import math
import os
import time
import copy
import numpy as np
import data.dataloader.get_loader
from utils.logger import get_logger
import matplotlib.pyplot as plt
from utils.metrics import All_Metrics
from data.data_prepare.Normalize_and_reverse_Normalize import SCS_SST_reverse_minmaxscaler
from data.data_prepare.Normalize_and_reverse_Normalize import SCS_SSH_reverse_minmaxscaler

def read_npy(file_path):
    return np.load(file_path)

# 计算 MAE
def mean_absolute_error(y_true, y_pred, mask):
    masked_y_true = y_true[mask == 1]
    masked_y_pred = y_pred[mask == 1]
    return np.mean(np.abs(masked_y_true - masked_y_pred))

# 计算 RMSE
def root_mean_squared_error(y_true, y_pred, mask):
    masked_y_true = y_true[mask == 1]
    masked_y_pred = y_pred[mask == 1]
    return np.sqrt(np.mean((masked_y_true - masked_y_pred) ** 2))

# 计算 R²
def r2_score(y_true, y_pred, mask):
    masked_y_true = y_true[mask == 1]
    masked_y_pred = y_pred[mask == 1]
    ss_res = np.sum((masked_y_true - masked_y_pred) ** 2)
    ss_tot = np.sum((masked_y_true - np.mean(masked_y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan  # 避免除零错误

# 主函数
def calculate(true_file_path, pred_file_path, mask_file_path, output_file_path):
    # 读取数据
    try:
        y_true = read_npy(true_file_path)  # 形状 (batchsize, 10, 1, 64, 64)
        y_pred = read_npy(pred_file_path)  # 形状 (batchsize, 10, 1, 64, 64)
        mask = read_npy(mask_file_path)    # 形状 (64, 64)，值为 0 或 1
    except FileNotFoundError:
        print("文件路径错误，请检查路径!")
        return
    except ValueError as e:
        print(f"数据读取错误: {e}")
        return

    # 检查 mask 形状是否匹配
    if mask.shape != (64, 64):
        print("错误: mask 形状不匹配 (64,64)，请检查文件!")
        return
    # 将 mask 扩展为 (2040, 64, 64) 的形状
    mask = np.expand_dims(mask, axis=0)  # 形状变为 (1, 64, 64)
    mask = np.repeat(mask, y_true.shape[0], axis=0)  # 广播为和y_true_day的维度一致 (batchsize, 64, 64)
    print(mask.shape)
    # 初始化存储每天平均指标的列表
    mae_day_avg = []
    rmse_day_avg = []
    r2_day_avg = []

    # 计算每一天的平均指标
    days = list(range(y_true.shape[1]))  # 翻转天数顺序
    for day in days:
        # 提取当前天的数据
        y_true_day = y_true[:, day, 0, :, :]  # (batchsize, 64, 64)
        y_pred_day = y_pred[:, day, 0, :, :]  # (batchsize, 64, 64)
        # print(y_true_day.shape)
        # 计算所有 batch 的均值，应用 mask
        mae = mean_absolute_error(y_true_day, y_pred_day, mask)
        rmse = root_mean_squared_error(y_true_day, y_pred_day, mask)
        r2 = r2_score(y_true_day, y_pred_day, mask)

        # 存储每天的平均指标
        mae_day_avg.append(mae)
        rmse_day_avg.append(rmse)
        r2_day_avg.append(r2)

    # 将结果保存到文本文件并打印
    with open(output_file_path, "w", encoding="utf-8") as f:  # 指定 UTF-8 编码
        for i in range(len(mae_day_avg)):
            day_num = i + 1
            result_line = f"Day {day_num}: MAE: {mae_day_avg[i]:.4f}, RMSE: {rmse_day_avg[i]:.4f}, R²: {r2_day_avg[i]:.4f}\n"
            f.write(result_line)  # 写入文件
            print(result_line.strip())  # 同时打印到控制台
def get_loss_figure(save_path, train_loss_list, val_loss_list, epoch):
    """
    传入训练误差列表和验证误差列表后，绘制损失函数曲线，并且保存到指定的路径下
    :param save_path:
    :param train_loss_list:
    :param val_loss_list:
    :param epoch:
    :return:
    """
    x = range(0,epoch)
    train_loss = train_loss_list
    val_loss = val_loss_list
    plt.title('Train and Val Loss Analysis')
    plt.plot(x, train_loss, color='black', label='Train')
    plt.plot(x, val_loss, color='blue', label='Val')
    plt.legend(loc=0) # 设置最适应的图例
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(save_path + "Loss_Figure")


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader, log_name, result_dir, current_time, scheduler, args):
        """

        :param model: 传入模型
        :param loss: 损失函数
        :param optimizer: 优化器
        :param train_loader:
        :param val_loader:
        :param test_loader:
        """
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # 获取每一轮的样本数，即每一个epoch中，（X,Y）有多少对
        self.train_per_epoch = len(train_loader)
        self.val_per_epoch = len(val_loader)
        self.result_dir = result_dir
        self.current_time = current_time
        self.scheduler = scheduler

        # 创建log日志
        get_logger(fname=log_name)
        logging.info('Experiment log path in: {}'.format(log_name))
        self.args = args

        # 设置最好的模型、loss曲线的保存路径
        self.best_path = args.result_dir + '{}-best_model.pth'.format(current_time)
        self.last_path = args.result_dir + '{}-last_model.pth'.format(current_time)


    def val_epoch(self, epoch, val_dataloader):
        """
        :param epoch:
        :param val_dataloader:
        :return: 返回该轮的验证误差
        """
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target,data_ssh, target_ssh) in enumerate(val_dataloader):
                data = data.cuda()
                target = target.cuda()
                target_ssh = target_ssh.cuda()  # target.shape = torch.Size([5, 10, 1, 32, 32])
                data_ssh = data_ssh.cuda()  # data.shape = torch.Size([5, 10, 1, 32, 32])
                output,output_ssh,_,_= self.model(data,data_ssh)
                loss = self.loss(output.cuda(), target)
                loss2 = self.loss(output_ssh.cuda(), target_ssh)
                total_loss1 = 0.8*loss + 0.2*loss2
                total_val_loss += total_loss1.item()
###############################！调试处代码！########################################################
                # 调试用代码,正式训练时候需要删除
                # if batch_idx == 10:
                #     break

        val_loss = total_val_loss / len(val_dataloader)
        self.scheduler.step(val_loss)  # 学习率衰减
        logging.info('**************************************Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        logging.info('******当前的学习率为：{}.4f'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        return val_loss


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target,data_ssh, target_ssh) in enumerate(self.train_loader):
            data = data.cuda()
            target = target.cuda()
            target_ssh = target_ssh.cuda()
            data_ssh = data_ssh.cuda()

            # print("data.shape: {},target.shape: {}",data.shape,target.shape)
            target = torch.cat((data[:, 1:10, :, :, :], target), dim=1)
            target_ssh = torch.cat((data_ssh[:, 1:10, :, :, :], target_ssh), dim=1)

            self.optimizer.zero_grad()
            output,output_ssh,predictions_warmup_1,predictions_warmup_2= self.model(data,data_ssh)

            output = torch.cat([predictions_warmup_1, output], dim=1)
            output_ssh = torch.cat([predictions_warmup_2, output_ssh], dim=1)

            loss = self.loss(output.cuda(), target)
            loss2 = self.loss(output_ssh.cuda(), target_ssh)
            total_loss1 = 0.8*loss + 0.2*loss2
            total_loss1.backward()

            # 梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += total_loss1.item()

            # log information, 每100次记录一次，这里的100,后期可以换成args参数形式
            if batch_idx % 100 == 0:
                logging.info('Train Epoch {}: {}/{} Loss: {:.6f} time：{}'.format(epoch, batch_idx, self.train_per_epoch, total_loss1.item(),datetime.now()))
# ###############################！调试处代码！########################################################
#             # 调试用代码,正式训练时候需要删除
            # if batch_idx == 10:
            #     logging.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(epoch, batch_idx, self.train_per_epoch, loss.item()))
            #     break            
        train_epoch_loss = total_loss / self.train_per_epoch
        logging.info('******Train Epoch {}: average Loss: {:.6f}'.format(epoch, train_epoch_loss))


        return train_epoch_loss


    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            print("当前时间：",datetime.now())
            train_epoch_loss = self.train_epoch(epoch) # 获取当前epoch的训练loss值
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader) # 获取当前epoch的验证loss

            # 将误差记录到列表中
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)

            # # 执行 early stop 策略, 只有ConvLSTM需要
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
                logging.info('*********************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
            else:
                not_improved_count += 1
                best_state = False

            # 训练时，early_stop = True ，容忍轮次为15
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience and epoch > 50:
                    logging.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break

        # 结束所有epoch训练后
        training_time = time.time() - start_time
        logging.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # 将 train_loss_list 和 val_loss_list 列表保存起来，用于绘制loss曲线图
        np.save(self.result_dir + 'train_loss.npy', train_loss_list)
        np.save(self.result_dir + 'val_loss.npy', val_loss_list)
        # 绘制loss曲线图，并且把曲线图保存到该路径上
        get_loss_figure(save_path=self.result_dir, train_loss_list=train_loss_list,
                        val_loss_list=val_loss_list,
                        epoch=self.args.epochs)
        logging.info("Train and val loss have saved in {} ".format(self.result_dir))

        torch.save(best_model, self.best_path)
        logging.info("Saving current best model to " + self.best_path)

        self.model.load_state_dict(best_model)
        self.test(model=self.model,
                  args=self.args,
                  data_loader=self.test_loader,
                  result_path=self.result_dir,
                  current_time=self.current_time)

    @staticmethod
    def test(model, args, data_loader, result_path, current_time):  # 对已经训练好的模型进行测试
        """

        :param model:
        :param agrs:
        :param data_loader: test_loader 设置为了batch——size = 1， shuffle = False
        :param model_path:
        :param result_path:
        :param test_only: 是一个布尔值，为True表示仅仅调用test函数进行测试模型，为False表示是程序在训练完后，调用test函数，不需要再重新加载模型
        :return:
        """

        model.eval()

        # 保存所有天的数据，方便分析
        total_pred = []
        total_true = []
        total_pred_ssh = []
        total_true_ssh = []

        logging.info("********Entering Test Mode")
        length = len(data_loader)
        with torch.no_grad():
            for batch_idx, (data, target,data_ssh, target_ssh) in enumerate(data_loader):
                data = data.cuda()
                target = target.cuda()
                target_ssh = target_ssh.cuda()
                data_ssh = data_ssh.cuda()
                output,output_ssh,_,_ = model(data,data_ssh)

                # 保存所有的数据，方便后面分析
                total_pred.append(output)
                total_true.append(target)
                total_pred_ssh.append(output_ssh)
                total_true_ssh.append(target_ssh)
                if batch_idx % 100 == 0:
                    logging.info("Finish testing  {}/{}".format(batch_idx, length))

# ###############################！调试处代码！########################################################
#                 # 调试问题时临时添加代码
                # if batch_idx == 10:
                #     logging.info("Finish testing  {}/{}".format(batch_idx, length))
                #     break

        logging.info("Test the model have Done!")
        logging.info("--------------------------------------")

        total_pred_cpu = torch.tensor([item.cpu().detach().numpy() for item in total_pred])
        total_true_cpu = torch.tensor([item.cpu().detach().numpy() for item in total_true])
        total_pred_cpu_ssh = torch.tensor([item.cpu().detach().numpy() for item in total_pred_ssh])
        total_true_cpu_ssh = torch.tensor([item.cpu().detach().numpy() for item in total_true_ssh])

        total_pred = np.array(total_pred_cpu)
        total_true = np.array(total_true_cpu)
        total_pred_ssh = np.array(total_pred_cpu_ssh)
        total_true_ssh = np.array(total_true_cpu_ssh)

        # 转换为numpy数组
        total_pred_sst_reverse = SCS_SST_reverse_minmaxscaler(total_pred)
        total_true_sst_reverse = SCS_SST_reverse_minmaxscaler(total_true)

        total_pred_ssh_reverse = SCS_SSH_reverse_minmaxscaler(total_pred_ssh)
        total_true_ssh_reverse = SCS_SSH_reverse_minmaxscaler(total_true_ssh)

        print("原始形状：", total_pred_sst_reverse.shape)

        sst_reshaped_pred = total_pred_sst_reverse.reshape(-1, 10, 1, 64, 64)
        sst_reshaped_true = total_true_sst_reverse.reshape(-1, 10, 1, 64, 64)
        ssh_reshaped_pred = total_pred_ssh_reverse.reshape(-1, 10, 1, 64, 64)
        ssh_reshaped_true = total_true_ssh_reverse.reshape(-1, 10, 1, 64, 64)

        print("reshape形状：", sst_reshaped_pred.shape)
        np.save(result_path + 'reverse_total_true_sst.npy', sst_reshaped_true)
        np.save(result_path + 'reverse_total_pred_sst.npy', sst_reshaped_pred)
        logging.info('**************The total timesteps sst-reverse-data  have saved in: {}'.format(result_path))

        np.save(result_path + 'reverse_total_true_ssh.npy', ssh_reshaped_true)
        np.save(result_path + 'reverse_total_pred_ssh.npy', ssh_reshaped_pred)
        logging.info('**************The total timesteps ssh-reverse-data  have saved in: {}'.format(result_path))

       

         #指标计算
        calculate(result_path + 'reverse_total_true_sst.npy',result_path + 'reverse_total_pred_sst.npy','../../SCS_SST_SSH_DATA/maskLand=0.npy',
                  result_path + 'sst_metrics_results.txt')
        calculate(result_path + 'reverse_total_true_ssh.npy', result_path + 'reverse_total_pred_ssh.npy','../../SCS_SST_SSH_DATA/maskLand=0.npy',
                  result_path + 'ssh_metrics_results.txt')











































