# FRUIT-classification
# 水果分类

## 简介
本项目基于宽核深度卷积神经网络（WDCNN）模型对纯净水、猕猴桃、沙糖桔和柠檬汁进行分类。WDCNN模型仅用15 s，就以100%的准确率巧妙地区分了样本。通过利用深度学习的能力，传感器系统可以实现时间序列信号的快速处理和分类，从而有助于及时检测异常、预测和识别。

## 环境要求
- Python 3.8 或以上版本
- Torch
- Keras
- NumPy
- Pandas
- Scikit-learn

## 数据集
本项目使用的数据集包含多种水果的电阻信号数据。数据集是已經進行數據增强的數據集。

## 模型架构
WDCNN模型结合寬深度卷積神经网络的优点，能够有效地从信号中提取特征并进行分类。模型的主要架构包括：

卷积层：用于提取信号的特征。
池化层：用于降低特征的空间维度，同时增加对位移的不变性。
全连接层：用于最终的分类决策。
## 使用方法
### 克隆代码库到本地机器：
bash
git clone https://github.com/xiaoxiaocui007/FRUIT-classification.git/
### 进入项目目录：
bash
cd fruit-classification-wdcnn
### 运行训练脚本：
bash
python train.py
### 运行测试脚本：
bash
python test.py
### 模型训练
训练脚本train.py将加载数据集，构建WDCNN模型，并在训练集上进行训练。

### 模型评估
测试脚本test.py将加载训练好的模型，并在测试集上评估模型性能。

# 贡献
欢迎对本项目做出贡献。如果您有任何建议或发现问题，请提交Pull Request或创建Issue。

# 联系
邮箱：[2574084240@qq.com]
