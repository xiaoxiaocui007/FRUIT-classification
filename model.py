import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self, num_class=10):
        super(MyModel, self).__init__()

        # 第一层卷积&池化
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, stride=1, padding=1)  # 修改了参数
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 第二层卷积&池化
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 第三层卷积&池化
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 第四层卷积&池化
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.max_pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 第五层卷积&池化
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(64)
        self.relu5 = nn.ReLU()
        self.max_pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 将最后一层池化的结果展平
        self.flatten = nn.Flatten()

        """
        cnn直接修改这里，
        self.fc1 = nn.Linear(。。。。, 100)
        """
        # 添加两个全连接层
        self.fc1 = nn.Linear(2880, 100)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(100, num_class)

    def forward(self, x):
        x = x.unsqueeze(1)  # 在batch size和1之间添加一个维度
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.max_pool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.max_pool2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.max_pool3(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.max_pool4(x)#如果需要修改数量，断点这一行debug
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.max_pool5(x)
        x = self.flatten(x)
        x = self.relu6(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # 创建模型实例
    model = MyModel(num_class=5)

    # 测试模型
    input_tensor = torch.randn((5, 1500))  # 示例输入张量，1通道，1024个元素
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)
