# -- coding: utf-8 --
# @Time : 2023/11/7 19:34
# @Author : Tsubaki_01
# @File : linear_programming1.py

import numpy as np
from scipy.optimize import linprog

c = np.array([1, 1, 1, 1, 1, 1, 1])
a = np.array([[-1, 0, 0, 0, 0, 0, 0],
              [0, -1, 0, 0, 0, 0, 0],
              [0, 0, -1, 0, 0, 0, 0],
              [0, 0, 0, -1, 0, 0, 0],
              [0, 0, 0, 0, -1, 0, 0],
              [0, 0, 0, 0, 0, -1, 0],
              [0, 0, 0, 0, 0, 0, -1]])
b = np.array([-12,
              -12,
              -12,
              -12,
              -12,
              -12,
              -12])

res = linprog(c, A_ub=a, b_ub=b)

print(res)
