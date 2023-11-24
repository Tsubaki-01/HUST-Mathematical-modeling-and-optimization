# -- coding: utf-8 --
# @Time : 2023/11/13 10:45
# @Author : Tsubaki_01
# @File : steepest_descent.py
import matplotlib.pyplot as plt
import numpy as np
import time


def f(x):
    """
    目标函数
    """
    # return x[0] ** 2 + 2 * x[1] ** 2 - 2 * x[0] * x[1] - 2 * x[0]
    return 3 * x[0] ** 2 + 3 * x[1] ** 2 - x[0] ** 2 * x[1]


def grad(x):
    """
    目标函数的梯度
    """
    # return np.array([2 * x[0] - 2 * x[1] - 2, 4 * x[1] - 2 * x[0]])
    return np.array([6 * x[0] - 2 * x[0] * x[1], 6 * x[1] - x[0] ** 2])


def armijo(x, pk, c1=0.1, beta=0.5):
    """
    Armijo线性搜索算法
    """
    alpha = 1
    while f(x + alpha * pk) > f(x) + c1 * alpha * np.dot(grad(x), pk):
        alpha *= beta
    return alpha


def steepest_descent(x0, max_iter=1000, tol=1e-6):
    """
    最速下降法
    """
    x = x0
    k = 0
    global ite
    while k < max_iter:
        ite = ite + 1
        g = grad(x)
        p = -g
        alpha = armijo(x, p)
        x_new = x + alpha * p
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        k += 1
    return x, f(x)


# 测试
ite = 0
# x0 = np.array([0, 0])
x0 = np.array([1.5, 1.5])
time1 = time.time()
x_star, f_min = steepest_descent(x0)
time2 = time.time()
delta_time = time2 - time1
print("最优解为：", x_star.round(4))
print("目标函数的最小值为：", f_min.round(4))
print("迭代次数", ite)
print(f"耗时 {delta_time} s")

# 画图
fig = plt.figure()
# x1, x2 = np.meshgrid(np.arange(0, 3, 0.01), np.arange(0, 3, 0.01))
x1, x2 = np.meshgrid(np.arange(-6, 6, 0.01), np.arange(-6, 6, 0.01))
# y = x1 ** 2 + 2 * x2 ** 2 - 2 * x1 * x2 - 2 * x1
y = 3 * x1 ** 2 + 3 * x2 ** 2 - x1 ** 2 * x2
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x1, x2, y, cmap='viridis')
ax.scatter(x_star[0], x_star[1], f_min, s=200, c='blue', marker='.')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('f(x)')

plt.show()
