# -- coding: utf-8 --
# @Time : 2023/11/13 22:14
# @Author : Tsubaki_01
# @File : DFPandBFGS.py

from scipy.optimize import fmin_powell, fmin_bfgs
import numpy as np
import matplotlib.pyplot as plt
import time


def r(i, x):
    if i % 2 == 1:
        return 10 * (x[i] - x[i - 1] ** 2)  # 因为ndarray数组的index是从0开始的， i多减一个
    else:
        return 1 - x[i - 2]


def f(m):
    def result(x):
        return sum([r(i, x) ** 2 for i in range(1, m + 1)])

    return result


f = f(4)


# def f(x):
#     return 3 * x[0] ** 2 + 3 * x[1] ** 2 - x[0] ** 2 * x[1]


def getLosses(retall, target_point, func):
    """
    :param retall: 存储迭代过程中每个迭代点的列表，列表的每个元素时一个ndarray对象
    :param target_point: 最优点，是ndarray对象
    :param func: 优化函数的映射f
    :return: 返回一个列表，代表retall中每个点到最优点的欧氏距离
    """
    losses = []
    for point in retall:
        losses.append(np.abs(func(target_point) - func(point)))
    return losses


# 绘制下降曲线
def plotDownCurve(dpi, losses, labels, xlabel=None, ylabel=None, title=None, grid=True):
    plt.figure(dpi=dpi)
    for loss, label in zip(losses, labels):
        plt.plot(loss, label=label)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=18)
    plt.yscale("log")
    plt.grid(grid)
    plt.legend()


x_0 = np.array([1.5, 1, 1, 1])  # 迭代初值
target_point = np.array([1, 1, 1, 1], dtype="float32")  # 最优点

dfp_time1 = time.time()
dfp_minimum, dfp_retall = fmin_powell(func=f, x0=x_0,
                                      retall=True,
                                      disp=False)
dfp_losses = getLosses(dfp_retall, target_point, func=f)
dfp_time2 = time.time()

BFGS_time1 = time.time()
bfgs_minimum, bfgs_retall = fmin_bfgs(f=f, x0=x_0,
                                      retall=True,
                                      disp=False)
bfgs_losses = getLosses(bfgs_retall, target_point, func=f)
BFGS_time2 = time.time()

plotDownCurve(dpi=150,
              losses=[dfp_losses, bfgs_losses],
              labels=["DFP", "BFGS"],
              xlabel="iter",
              ylabel="value of $|f(x) - f(x^*)|$",
              title="losses curve of DFP and BFGS")
plt.show()

print(f"DFP最终迭代点:{dfp_minimum.round(4)}, 共经历{len(dfp_losses)}次迭代, 耗时{dfp_time2 - dfp_time1} s")
print(f"BFGS最终迭代点:{bfgs_minimum.round(4)}, 共经历{len(bfgs_losses)}次迭代, 耗时{BFGS_time2 - BFGS_time1} s")
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# x = np.linspace(-6, 6, 100)
# y = np.linspace(-6, 6, 100)
# X, Y = np.meshgrid(x, y)
# Z = 3 * X ** 2 + 3 * Y ** 2 - X ** 2 * Y
# ax.plot_surface(X, Y, Z, cmap='viridis')
# ax.scatter(dfp_minimum[0], dfp_minimum[1], f(dfp_minimum), s=200, c='blue', marker='.')
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('y')
# ax.set_title('f(x)')
#
# plt.show()
