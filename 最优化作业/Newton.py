# -- coding: utf-8 --
# @Time : 2023/11/13 10:45
# @Author : Tsubaki_01
# @File : Newton.py


from sympy import *
import numpy as np
import math
from matplotlib import pyplot as plt
import time


def get_Hessian(data):
    dx1x1 = y.diff(x1, x1).subs(x1, data[0]).subs(x2, data[1]).evalf()
    dx1x2 = y.diff(x1, x2).subs(x1, data[0]).subs(x2, data[1]).evalf()
    dx2x1 = y.diff(x2, x1).subs(x1, data[0]).subs(x2, data[1]).evalf()
    dx2x2 = y.diff(x2, x2).subs(x1, data[0]).subs(x2, data[1]).evalf()
    return np.array([[dx1x1, dx1x2], [dx2x1, dx2x2]])


def get_Jacobian(data):
    dx1 = y.diff(x1).subs(x1, data[0]).subs(x2, data[1]).evalf()
    dx2 = y.diff(x2).subs(x1, data[0]).subs(x2, data[1]).evalf()
    return np.array([[dx1], [dx2]]), math.sqrt(dx1 ** 2 + dx2 ** 2)


def plot_contour():
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = 3 * X ** 2 + 3 * Y ** 2 - X ** 2 * Y
    plt.contour(X, Y, Z, 25, colors='red')


if __name__ == "__main__":
    # 打开交互模式
    plt.ion()
    # 绘制等高线
    plot_contour()
    plt.xlabel('x1')
    plt.ylabel('x2')
    # 定义迭代过程需求解的函数
    x1, x2, y = symbols("x1 x2 y")
    y = 3 * x1 ** 2 + 3 * x2 ** 2 - x1 ** 2 * x2
    # 定义起始点
    x_initial = [1.5, 1.5]
    x_k = x_initial
    xx = []
    yy = []
    zz = []
    xx.append(round(x_k[0], 4))
    yy.append(round(x_k[1], 4))
    zz.append(round(y.subs(x1, x_k[0]).subs(x2, x_k[1]).evalf(), 4))
    # print("({:.4f},{:.4f}),    {:.4f}".format(xx[0], yy[0], zz[0]))
    plt.plot(xx, yy, c='black')
    plt.scatter(xx, yy, s=30, c='b')
    # plt.pause(3)
    time1 = time.time()
    for i in range(1, 7):
        Gk = get_Hessian(x_k)
        Gk = Gk.astype(float)
        gk, gknorm = get_Jacobian(x_k)
        gk = gk.astype(float)
        Gk_inverse = np.linalg.inv(Gk)
        dk = np.dot(Gk_inverse, gk)
        dk = dk.T[0]
        x_k = [x_k[0] - dk[0], x_k[1] - dk[1]]
        xx.append(round(x_k[0], 4))
        yy.append(round(x_k[1], 4))
        zz.append(round(y.subs(x1, x_k[0]).subs(x2, x_k[1]).evalf(), 4))
        # print("({:.4f},{:.4f}),    {:.4f}".format(xx[i], yy[i], zz[i]))
        plt.plot(xx, yy, c='black')
        plt.scatter(xx, yy, s=30, c='b')
        # plt.pause(3)
        if gknorm <= 10e-6:
            break
    time2 = time.time()
    plt.ioff()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = 3 * X ** 2 + 3 * Y ** 2 - X ** 2 * Y
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.scatter(xx[-1], yy[-1], zz[-1], s=200, c='blue', marker='.')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('f(x)')

    plt.show()

    print(f'最优解为：[ {xx[-1]},{yy[-1]} ]')
    print(f'目标函数最小值为：{zz[-1]}')
    print(f'迭代次数 6 ')
    print(f'耗时： {time2-time1} s')
