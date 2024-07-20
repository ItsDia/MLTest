import paddle
import numpy as np
import matplotlib.pyplot as plt
from paddle.nn.functional import mse_loss

x_data = paddle.to_tensor([[1.00], [2.00], [3.000], [4.000], [5.000], [6.000], [7.000], [8.000], [9.000], [10.000]])
y_data = paddle.to_tensor([[0.530], [1.060], [1.590], [2.120], [2.650], [3.230], [3.810], [4.390], [4.970], [5.550]])
linear = paddle.nn.Linear(in_features=1, out_features=1)
w_before_opt = linear.weight.numpy().item()
b_before_opt = linear.bias.numpy().item()

# let we define the y_predict = w * x + b

mse_loss = paddle.nn.MSELoss()
sgd_optimizer = paddle.optimizer.SGD(
    learning_rate=0.001, parameters=linear.parameters()
)
total_epoch = 10000
for i in range(total_epoch):
    y_predict = linear(x_data)
    loss = mse_loss(y_predict, y_data)
    loss.backward()
    sgd_optimizer.step()
    sgd_optimizer.clear_grad()

    if i % 1000 == 0:
        print("epoch {} loss {}".format(i, loss.numpy()))

print("finished training， loss {}".format(loss.numpy()))
w_after_opt = linear.weight.numpy().item()
b_after_opt = linear.bias.numpy().item()

print("w after optimize: {}".format(w_after_opt))
print("b after optimize: {}".format(b_after_opt))
# 7-3 sdut-C语言实验-虎子算电费
# 分数 15
# 作者 马新娟
# 单位 山东理工大学
# 为了提倡居民节约用电，某省电力公司执行“阶梯电价”，安装一户一表的居民用户电价分为两个“阶梯”：月用电量50千瓦时（含50千瓦时）以内的，电价为0.53元/千瓦时；超过50千瓦时的，超出部分的用电量，电价上调0.05元/千瓦时。
#
# 虎子决定编写程序计算一算家里电费。你也会编写这个程序吧？
#
# 输入格式:
# 输入在一行中给出虎子家的月用电量（单位：千瓦时）。
#
# 输出格式:
# 在一行中输出虎子家应支付的电费（元），结果保留两位小数，格式如：“cost = 应付电费值”；若用电量小于0，则输出"Invalid Value!"。
#
# 输入样例1:
# 10
# 输出样例1:
# cost = 5.30
# 输入样例2:
# 100
# 输出样例2:
# cost = 55.50