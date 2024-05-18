import numpy
import matplotlib.pyplot
# 引入 scipy.special	来使用S抑制函数 expit()
import scipy.special
import scipy.misc
from PIL import Image
import time


# %matplotlib inline

# a = numpy.random.rand(28, 28)
# matplotlib.pyplot.imshow(a, interpolation="nearest")
# matplotlib.pyplot.show()


class neuralNetwork:
    """神经网络类定义"""

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """初始化神经网络"""
        # 设置输入层、隐藏层、输出层的节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率（learning rate）
        self.lr = learningrate

        # 链接权重矩阵 wih 和 who
        # 权重在矩阵内部表现为 w_i_j ，表示从节点 i 链接到下一层的节点 j
        # w11 w21
        # w12 w22 等等
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

        # 激活函数为S函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        """训练神经网络"""
        # 将输入转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算一遍...和下面query函数一样
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 误差就是 target - actual （目标值 - 实际值）
        output_errors = targets - final_outputs

        # 隐藏层误差就是 output_errors 按权重分配再与隐藏节点结合得到的
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 更新隐藏层和输出层之间的权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # 更新输入层和隐藏层之间的权重
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

    def query(self, input_list):
        """查询神经网络"""
        # 将输入转换为二维数组
        inputs = numpy.array(input_list, ndmin=2).T

        # 计算要传递到隐藏层的数值
        hidden_inputs = numpy.dot(self.wih, inputs)

        # 计算隐藏层的输出数值
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算要传递到输出层的数值
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # 计算输出层的输出数值
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def save_data(self):
        """保存训练结果（权重）"""
        numpy.savetxt("data_wih.txt", self.wih)
        numpy.savetxt("data_who.txt", self.who)

    def read_data(self):
        """读取训练模型（权重）"""
        self.wih = numpy.loadtxt("data_wih.txt")
        self.who = numpy.loadtxt("data_who.txt")


def start_train():
    """读取训练集并进行训练"""

    print("开始训练...")

    # 读取训练数据集
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 记录训练用时
    start_time = time.perf_counter()

    # 训练
    for train_record in training_data_list:

        # 创建训练单轮的输入数据
        # 用','分割数据
        train_all_values = train_record.split(',')
        # 转换输入数据
        train_inputs = (numpy.asfarray(train_all_values[1:]) / 255.0 * 0.99) + 0.01
        # 创建训练单轮的样例输出
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(train_all_values[0])] = 0.99

        # 开始该轮训练
        n.train(train_inputs, targets)

    # 记录训练用时
    end_time = time.perf_counter()

    print("训练完成，用时（秒）： " + str(end_time - start_time))


def start_test():
    """读取测试集并进行测试"""

    print("开始测试...")

    # 读取测试数据集
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # 初始化计分器
    scoreboard = []

    # 测试
    for record in test_data_list:

        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        correct_label = int(all_values[0])
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)

        # 计分器统计结果
        if label == correct_label:
            scoreboard.append(1)
        else:
            scoreboard.append(0)

    # 记分器数组化
    scoreboard_array = numpy.asarray(scoreboard)
    print("测试完成，本次测试准确率：" + str(scoreboard_array.sum() / scoreboard_array.size))


def query_by_image():
    """读取图片进行识别"""
    img = Image.open("image.png").convert('L')  # Convert to grayscale
    img_array = numpy.array(img)
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01

    return numpy.argmax(n.query(img_data))


def help_list():
    """显示指令列表"""
    print("#############################")
    print("# 0 显示此列表")
    print("# 1 读取训练集并进行训练")
    print("# 2 读取测试集并进行测试")
    print("# 3 保存模型数据")
    print("# 4 读取模型数据")
    print("# 5 删除模型重新创建一个")
    print("# 6 读取图片进行识别")
    print("#############################")


# 节点数
input_nodes = 784
hidden_nodes = 500
output_nodes = 10

# 学习率
learning_rate = 0.2

# 创建神经网络实例
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


help_list()
while True:
    print("请输入：", end='')
    x = int(input().replace(" ", ""))
    if x == 0:
        help_list()
    if x == 1:
        start_train()
    if x == 2:
        start_test()
    if x == 3:
        n.save_data()
        print("保存完成")
    if x == 4:
        n.read_data()
        print("读取完成")
    if x == 5:
        del n
        n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    if x == 6:
        r = query_by_image()
        print("识别结果为： " + str(r))

