import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """ 单个 ConvLSTM 单元，支持 2D 卷积 """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2  # 确保输入输出尺寸一致

        # 4个门计算（输入门、遗忘门、输出门、候选细胞状态）
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, x, h_prev, c_prev):
        """ConvLSTM 前向传播"""
        combined = torch.cat([x, h_prev], dim=1)  # 拼接输入和隐藏态
        gates = self.conv(combined)  # 卷积运算

        # 分割成 4 个门
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # 细胞状态更新
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new

class SelfAttention(nn.Module):
    """ 自注意力模块，用于 H 和 C """

    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.key = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.value = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        """ 计算自注意力 """
        batch_size, channels, height, width = x.size()

        # 生成 Query, Key 和 Value
        q = self.query(x).view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, height*width, channels)
        k = self.key(x).view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, height*width, channels)
        v = self.value(x).view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, height*width, channels)

        # 计算 QK^T，并除以 sqrt(d_k)
        d_k = q.size(-1)  # 通常是 channels 的大小
        attn_map = torch.bmm(q, k.permute(0, 2, 1)) / (d_k ** 0.5)  # 计算点积并缩放

        # 计算注意力分布
        attn_map = torch.softmax(attn_map, dim=-1)  # (batch_size, height*width, height*width)

        # 使用注意力图加权 V
        out = torch.bmm(attn_map, v)  # (batch_size, height*width, channels)

        # 恢复维度并进行残差连接
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)  # 恢复维度
        return out + x  # 残差连接

class CrossAttention(nn.Module):
    """ 交叉注意力模块（两个输入流之间） """

    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.key = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.value = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x1, x2):
        """ 计算交叉注意力，输入 x1 和 x2 """
        batch_size, channels, height, width = x1.size()

        # 生成 Query, Key 和 Value
        q = self.query(x1).view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, height*width, channels)
        k = self.key(x2).view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, height*width, channels)
        v = self.value(x2).view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, height*width, channels)

        # 计算 QK^T，并除以 sqrt(d_k)
        d_k = q.size(-1)  # 通常是 channels 的大小
        attn_map = torch.bmm(q, k.permute(0, 2, 1)) / (d_k ** 0.5)  # 计算点积并缩放

        # 计算注意力分布
        attn_map = torch.softmax(attn_map, dim=-1)  # (batch_size, height*width, height*width)

        # 使用注意力图加权 V
        out = torch.bmm(attn_map, v)  # (batch_size, height*width, channels)
        # 恢复维度并进行残差连接
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)  # 恢复维度
        return out + x1  # 残差连接
    
class MultiLayerConvLSTM(nn.Module):
    """ 多层双输入流 ConvLSTM，结合自注意力和交叉注意力 """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim):
        super(MultiLayerConvLSTM, self).__init__()
        self.num_layers = num_layers

        # 初始化 ConvLSTM 层
        self.lstm_layers_1 = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])
        self.lstm_layers_2 = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])

        # 自注意力和交叉注意力
        self.self_attn_h = nn.ModuleList([SelfAttention(hidden_dim) for _ in range(num_layers)])
        self.self_attn_c = nn.ModuleList([SelfAttention(hidden_dim) for _ in range(num_layers)])
        self.cross_attn_h = nn.ModuleList([CrossAttention(hidden_dim) for _ in range(num_layers)])
        self.cross_attn_c = nn.ModuleList([CrossAttention(hidden_dim) for _ in range(num_layers)])

        # 解码器：生成预测结果
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)
        )

    def forward(self, x1_seq, x2_seq, future_steps=10):
        """
        x1_seq, x2_seq: (batch, seq_len, C, H, W)
        future_steps: 需要预测的未来时间步数
        """
        batch, seq_len, _, height, width = x1_seq.shape
        hidden_states_1 = [torch.zeros(batch, self.lstm_layers_1[0].hidden_dim, height, width).to(x1_seq.device) for _ in range(self.num_layers)]
        cell_states_1 = [torch.zeros_like(hidden_states_1[0]) for _ in range(self.num_layers)]
        hidden_states_2 = [torch.zeros_like(hidden_states_1[0]) for _ in range(self.num_layers)]
        cell_states_2 = [torch.zeros_like(hidden_states_1[0]) for _ in range(self.num_layers)]

        # warm-up
        warmup_hidden_states_1 = []
        warmup_hidden_states_2 = []

        # 进行前向传播，直到最后的隐藏态
        for t in range(seq_len):
            x1_t = x1_seq[:, t, :, :, :]
            x2_t = x2_seq[:, t, :, :, :]

            for layer in range(self.num_layers):
                # 计算 ConvLSTM 更新
                h1, c1 = self.lstm_layers_1[layer](x1_t, hidden_states_1[layer], cell_states_1[layer])
                h2, c2 = self.lstm_layers_2[layer](x2_t, hidden_states_2[layer], cell_states_2[layer])

                # 自注意力
                h1 = self.self_attn_h[layer](h1)
                c1 = self.self_attn_c[layer](c1)
                h2 = self.self_attn_h[layer](h2)
                c2 = self.self_attn_c[layer](c2)

                tag_h1 = h1
                tag_c1 = c1

                # 交叉注意力
                h1 = self.cross_attn_h[layer](h1, h2)
                c1 = self.cross_attn_c[layer](c1, c2)
                h2 = self.cross_attn_h[layer](h2, tag_h1)
                c2 = self.cross_attn_c[layer](c2, tag_c1)



                # 更新状态
                hidden_states_1[layer], cell_states_1[layer] = h1, c1
                hidden_states_2[layer], cell_states_2[layer] = h2, c2

                x1_t, x2_t = h1, h2  # 传递到下一层

            # 存储除最后一个隐藏状态，用于生产warm up数据
            if t < seq_len-1:
                warmup_hidden_states_1.append(hidden_states_1[-1].clone())
                warmup_hidden_states_2.append(hidden_states_2[-1].clone())

        # 生成预测（基于 warmup_hidden_states）
        predictions_warmup_1 = [self.decoder(h) for h in warmup_hidden_states_1]
        predictions_warmup_2 = [self.decoder(h) for h in warmup_hidden_states_2]
        predictions_warmup_1 = torch.stack(predictions_warmup_1, dim=1)
        predictions_warmup_2 = torch.stack(predictions_warmup_2, dim=1)

        # 创建两个列表分别保存每个时间步的预测结果
        predictions_1 = []
        predictions_2 = []

        # 使用最终的隐藏态来进行预测
        for _ in range(future_steps):
            # 解码器使用最后一层隐藏态来预测
            prediction_1 = self.decoder(hidden_states_1[-1])
            prediction_2 = self.decoder(hidden_states_2[-1])

            predictions_1.append(prediction_1)
            predictions_2.append(prediction_2)

            # 将解码器的输出作为下一个时间步的输入
            x1_t = prediction_1
            x2_t = prediction_2

            # 使用预测结果作为下一个时间步的输入，进行下一步的迭代预测
            for layer in range(self.num_layers):
                # 更新隐藏状态和细胞状态
                h1, c1 = self.lstm_layers_1[layer](x1_t, hidden_states_1[layer], cell_states_1[layer])
                h2, c2 = self.lstm_layers_2[layer](x2_t, hidden_states_2[layer], cell_states_2[layer])


                # 自注意力
                h1 = self.self_attn_h[layer](h1)
                c1 = self.self_attn_c[layer](c1)
                h2 = self.self_attn_h[layer](h2)
                c2 = self.self_attn_c[layer](c2)

                tag_h1 = h1
                tag_c1 = c1

                # 交叉注意力
                h1 = self.cross_attn_h[layer](h1, h2)
                c1 = self.cross_attn_c[layer](c1, c2)
                h2 = self.cross_attn_h[layer](h2, tag_h1)
                c2 = self.cross_attn_c[layer](c2, tag_c1)

                # 更新状态
                hidden_states_1[layer], cell_states_1[layer] = h1, c1
                hidden_states_2[layer], cell_states_2[layer] = h2, c2

                x1_t, x2_t = h1, h2  # 传递到下一层

        # 将预测结果转换为 tensor，返回所有时间步的预测
        predictions_1 = torch.stack(predictions_1, dim=1)  # (batch, future_steps, C, H, W)
        predictions_2 = torch.stack(predictions_2, dim=1)  # (batch, future_steps, C, H, W)
        return predictions_1, predictions_2,predictions_warmup_1,predictions_warmup_2

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.manual_seed(0)
    # 模型参数设置
    input_dim = 1  # 输入通道数（例如 RGB 图像的 3 通道）
    hidden_dim = 64  # 隐藏层通道数
    kernel_size = 3  # 卷积核大小
    num_layers = 2  # ConvLSTM 层数
    output_dim = 1  # 输出通道数（例如预测图像的 3 通道）
    seq_len = 10  # 输入序列的时间步数
    future_steps = 10  # 需要预测的未来时间步数

    # 创建模型实例
    model = MultiLayerConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, output_dim)

    # 随机生成输入数据
    batch_size = 4  # 批量大小
    x1_seq = torch.randn(batch_size, seq_len, input_dim, 64, 64)  # 输入流1，大小 (batch, seq_len, C, H, W)
    x2_seq = torch.randn(batch_size, seq_len, input_dim, 64, 64)  # 输入流2，大小 (batch, seq_len, C, H, W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)  # 将模型转移到 GPU
    x1_seq = x1_seq.to(device)  # 将输入数据转移到 GPU
    x2_seq = x2_seq.to(device)  # 同样处理

    # 计算模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")  # 格式化输出
    print(f"Trainable Parameters: {trainable_params:,}")

    # 将输入数据传递到模型中并获取预测结果
    prediction1, prediction2,predictions_warmup_1,_ = model(x1_seq, x2_seq)

    print(prediction1.shape)
    print(predictions_warmup_1.shape)
    # 检查输出形状
    # for t in range(future_steps):
    #     print(f"Prediction 1 - Time step {t+1}: Shape = {prediction1[t].shape}, Type = {prediction1[t].dtype}")
    #     print(f"Prediction 2 - Time step {t+1}: Shape = {prediction2[t].shape}, Type = {prediction2[t].dtype}")

