# 数字识别神经网络模型（NumberAI）

## 模型介绍

### 构成

经典的神经网络模型，由三层节点构成，分别是**输入层**、**隐藏层**、**输出层**。在本项目提供的样例资源配合下，样例代码中的输入层有**28×28**即**784**个输入节点，隐藏层有**500**个神经节点，输出层有**10**个输出节点，分别代表**0~10**。每相邻层的所有节点之间都互相连接，每条边上的初始权重随机。借助**numpy**模块**矩阵**来运行神经网络来简化编程。

在使用**训练**功能时，程序会从训练集文件读取数据，根据设定的**学习率**来运行神经网络，与结果比较后将误差按权重分配回各边的权重。

在使用**测试**功能时，程序和从测试集文件读取数据，记录下神经网络模型对测试集的判断准确情况。

同时提供了**保存模型**和**读取模型**的功能，可以存取神经网络所有的的权重值数据，以免每次程序运行完毕后模型数据就消失了。

额外功能：**图片识别**，程序将读取一张图片，要求该图片必须为**28×28**像素，白底黑字，程序在转化为输入数据后投入神经网络模型进行识别，返回识别结果。

### 文件

**main.py** 程序主文件。所有的代码都在这里面。

**data_wih.txt** 模型文件。用于保存输入层到隐藏层之间的所有权重。

**data_who.txt** 模型文件。用于保存隐藏层到输出层之间的所有权重。

**image.png** 图片文件。用于让程序读取并识别。

**image-empty.png** 图片文件。为纯白的28×28像素的模板图片。

**mnist_test.csv** 测试数据集文件。

**mnist_train.csv** 训练数据集文件。

**mnist_train_100.csv** 小规模训练数据集文件。仅有100条记录，用于测试程序是否能够正常运行。

## 使用方法

直接运行**main.py**即可。按照程序提示输入数字操作即可。本处提供的模型文件已经过训练，可直接读取使用。