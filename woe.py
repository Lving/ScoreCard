# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def calc_cross(data, col,  bins):
    """
    为woe的计算准备交叉表
    :param array: 待计算的数组
    :param atr_num: 列序号
    :param bins: 最优分箱
    :return: good_pct, bad_pct, 数组
    """
    y = data[:, -1]
    x = data[:, col]
    assert set(np.unique(y)) == {0,1}

    bins = np.array(bins)
    # r = len(uniq_y)  #
    c = len(bins)-1

    good_array = np.zeros(len(bins)-1)
    bad_array = np.zeros(len(bins)-1)
    for j in range(c):
        good_mask = (bins[j] <= x) & (x < bins[j+1]) & (y == 0)  # good
        bad_mask = (bins[j] <= x) & (x < bins[j+1]) & (y == 1)  # bad

        good_array[j] = sum(good_mask)  # count
        bad_array[j] = sum(bad_mask)    # count

    good_pct = good_array / good_array.sum()
    bad_pct = bad_array / bad_array.sum()

    return good_pct, bad_pct


def woe(data, col,  bins):
    """
    :param data: 待计算的数组
    :param col: 列序号
    :param bins: 最优分箱
    :return: 每个分箱区间内的woe_i
    """
    g_pct, b_pct = calc_cross(data, col, bins)
    _woe_i = np.log(g_pct / b_pct)
    return _woe_i


def iv(data, col, bins):
    """
    :param data:
    :param col:
    :param bins:
    :return:
    """
    g_pct, b_pct = calc_cross(data, col, bins)
    _woe = np.log(g_pct / b_pct)
    _iv = (g_pct - b_pct) * _woe
    return sum(_iv)


def trans_woe(X, woe_i, bins):
    """
    将一列X转换成相应的woe
    :param X:
    :param woe_i:
    :param bins:
    :return:
    """
    bins = np.array(bins)
    inx = np.searchsorted(bins, X, 'right')  # 查找位置
    return woe_i[inx-1]


if __name__ == '__main__':
    bins = [4.3, 5.5, 5.8, 7.1, 8]  # SepalLengthCm
    # print(np.searchsorted(np.array(bins), 7.1, 'right'))
    data = pd.read_csv('IRIS.csv')
    data = data[['SepalLengthCm', 'Species']].copy()
    data = data.values

    data[:, -1] = np.random.choice([0, 1], 150)
    woe_i = woe(data, 0,  bins)
    print(woe(data, 0,  bins))
    print(bins)
    print(trans_woe(np.array([5.5, 4.4, 7, 5.8, 7.4]), woe_i, bins))