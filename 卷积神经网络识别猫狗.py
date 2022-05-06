# -*- coding: utf-8 -*-
import numpy as np
import paddle as paddle
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt
import os


BATCH_SIZE = 128
# 用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.cifar.train10(),
                          buf_size=BATCH_SIZE*100),
    batch_size=BATCH_SIZE)
# 用于测试的数据提供器，每次从缓存中随机读取批次大小的数据
test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.cifar.test10(),
                          buf_size=BATCH_SIZE),
    batch_size=BATCH_SIZE)
print('train_reader is type:',type(train_reader))

# 网络定义
def convolutional_neural_network(img):
    # 第一个卷积-池化层
    conv1 = fluid.layers.conv2d(
        input=img,  # 输入图像
        num_filters=20,  # 卷积核大小
        filter_size=5,  # 卷积核数量，它与输出的通道相同
        act='relu'  # 激活函数
    )
    pool1 = fluid.layers.pool2d(
        input=conv1,  # 输入
        pool_size=2,  # 池化核大小
        pool_type='max',  # 池化类型
        pool_stride=2  # 池化步长
    )
    conv_pool_1 = fluid.layers.batch_norm(pool1)
    # 第二个卷积-池化层
    conv2 = fluid.layers.conv2d(
        input=conv_pool_1,
        num_filters=50,
        filter_size=5,
        act='relu'
    )
    pool2 = fluid.layers.pool2d(
        input=conv2,
        pool_size=2,
        pool_type='max',
        pool_stride=2
    )
    conv_pool_2 = fluid.layers.batch_norm(pool2)
    # 第三个卷积-池化层
    conv3 = fluid.layers.conv2d(
        input=conv_pool_2,
        num_filters=50,
        filter_size=5,
        act='relu'
    )
    pool3 = fluid.layers.pool2d(
        input=conv3,
        pool_size=2,
        pool_type='max',
        pool_stride=2
    )
    # 全连接输出层，10个神经元（对应数字0~9），激活函数为softmax
    prediction = fluid.layers.fc(input=pool3, size=10, act='softmax')
    return prediction


#3代表图像三通道，32x32的彩色图片
data_shape = [3,32,32]
paddle.enable_static()
# 输入的原始图像数据，大小为1*28*28(单通道，28*28像素)
images = fluid.layers.data(
    name='image',
    shape=data_shape,
    dtype='float32'
)
# 标签，名称为label，对应输入图片的类别标签
label = fluid.layers.data(
    name='label',
    shape=[1],
    dtype='int64'
)
# 获取分类器
predict = convolutional_neural_network(images)
# 使用交叉熵损失函数，描述真实样本标签和预测概率之间的差值
cost = fluid.layers.cross_entropy(
    input=predict,
    label=label
)
# 使用类交叉熵函数计算predict和label之间的损失函数
avg_cost = fluid.layers.mean(cost)
# 计算分类准确率
acc = fluid.layers.accuracy(
    input=predict,
    label=label
)
# 使用Adam 算法进行优化，learning_rate 是学习率（其大小与网络的训练收敛速度有关，默认0.001）
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)
# 定义使用CPU还是GPU，使用CPU时use_cuda =False,使用GPU时use_cuda = True
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# 创建一个Executor实例exe
exe = fluid.Executor(place)
# 正式进行网络训练前，需先执行参数初始化
exe.run(fluid.default_startup_program())
# 定义好网络训练需要的Executor，在执行训练之前，需要告知网络传入的数据分为两部分，第一部分是image值，第二部分是label值：
feeder = fluid.DataFeeder(place=place, feed_list=[images, label])

# 展示模型训练曲线
all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []


def draw_train_process(title, iters, costs, accs, label_cost, label_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=label_acc)
    plt.legend()
    plt.grid()
    plt.show()


# 训练模型
EPOCH_NUM = 10
model_save_dir = "./cat.inference.model"
for pass_id in range(EPOCH_NUM):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                        feed=feeder.feed(data),  # 给模型喂入数据
                                        fetch_list=[avg_cost, acc])  # fetch 误差、准确率



        # 每100个batch打印一次信息 误差、准确率
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f,Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    # 每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader
        test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # 执行训练程序
                                      feed=feeder.feed(data),  # 喂入数据
                                      fetch_list=[avg_cost, acc])  # fetch 误差、准确率
        all_train_iter = all_train_iter + BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

        test_accs.append(test_acc[0])  # 每个batch的准确率
        test_costs.append(test_cost[0])  # 每个batch的误差
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))  # 每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))  # 每轮的平均准确率
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))

# 保存模型
# 如果保存路径不存在就创建
model_save_dir = r'D:\Pycharm_project\PaddlePaddle\CNN_cat_detection'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save models to %s' % (model_save_dir))
fluid.io.save_inference_model(model_save_dir,  # 保存预测Program的路径
                              ['image'],  # 预测需要feed的数据
                              [predict],  # 保存预测结果
                              exe)  # executor 保存预测模型
print('训练模型保存完成!')

# 训练过程可视化
draw_train_process("training",
                   all_train_iters,
                   all_train_costs,
                   all_train_accs,
                   "training cost",
                   "training acc")


# 模型预测
def load_image(file):
    im = Image.open(file)  # 将RGB转化为灰度图像，像素值在0~255之间
    im = im.resize((32,32), Image.ANTIALIAS)  # resize image with high-quality
    im = np.array(im).astype(np.float32)  # 把它变成一个 numpy 矩阵以匹配数据馈送格式。
    im = im.transpose((2,0,1))
    im = im / 255.0  # 归一化到【-1~1】之间
    im = np.expand_dims(im,axis=0)
    return im


img = load_image(r'D:\Pycharm_project\PaddlePaddle\cat.png')

# 加载数据并开始预测
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()
inference_program = model_save_dir  # 将训练模型目录赋值给测试模型目录
with fluid.scope_guard(inference_scope):
    # 获取训练好的模型
    # 从指定目录中加载 推理model(inference model)
    [inference_program,  # 推理Program
     feed_target_names,  # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。
     fetch_targets] = fluid.io.load_inference_model(model_save_dir,
                                                    # fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。model_save_dir：模型保存的路径
                                                    infer_exe)  # infer_exe: 运行 inference model的 executor

    infer_path = 'cat.png'
    img = Image.open(infer_path)
    plt.imshow(img)
    plt.show()
    img = load_image(infer_path)

    results = infer_exe.run(program=inference_program,  # 运行推测程序
                            feed={feed_target_names[0]: img},  # 喂入要预测的img
                            fetch_list=fetch_targets)  # 得到推测结果
# 获取概率最大的label
# argsort函数返回的是result 数组值从小到大的索引值
lab = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

print("该图片的预测结果的label 为：%s" % lab[np.argmax(results[0])])
