"""
模型評價
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, confusion_matrix, roc_curve, recall_score
from sklearn.utils.multiclass import type_of_target

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import itertools


def metric(true_Y, pred_Y, thresholds=0.5):
    """
    :param true_Y: np.array
    :param pred_Y: np.array
    :return:
    """
    fpr, tpr, _ = roc_curve(true_Y, pred_Y)
    auc_ = auc(fpr, tpr)

    cnf_matrix = confusion_matrix(true_Y, np.where(pred_Y > thresholds, 1, 0))

    recall = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])

    precision = cnf_matrix[1, 1]/(cnf_matrix[0, 1]+cnf_matrix[1, 1])

    accuracy = (cnf_matrix[0, 0] + cnf_matrix[1, 1]) / (cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[1, 0] + cnf_matrix[1, 1])

    return auc_, recall, precision, accuracy


def plot_confusion_matrix(true_Y, pred_Y, classes,
                          thresholds=0.5,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cnf_matrix = confusion_matrix(true_Y, np.where(pred_Y > thresholds, 1, 0))

    auc_, recall, precision, accuracy = metric(true_Y, pred_Y)
    print('++++++++++++++++++++++++++++++++++')
    print('AUC: ', auc_)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('Accyracy: ', accuracy)
    print('++++++++++++++++++++++++++++++++++')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cnf_matrix)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_roc_curve(true_Y, pred_Y):
    """
    plot roc curve
    ----------------------------------
    Params
    pred_Y: prediction of model
    y: real data(testing sets)
    ----------------------------------
    plt object
    """
    fpr, tpr, _ = roc_curve(true_Y, pred_Y)
    c_stats = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, label="ROC curve")
    s = "AUC = %.4f" % c_stats
    plt.text(0.6, 0.2, s, bbox=dict(facecolor='red', alpha=0.5))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')#ROC 曲线
    plt.legend(loc='best')
    plt.show()


def ks_stats(true_Y, pred_Y, k=20):
    """
    plot K-S curve and output ks table
    ----------------------------------
    Params
    pred_Y: prediction of model
    true_Y: real data(testing sets)
    k: Section number 
    ----------------------------------
    ks_results: pandas dataframe 
    ks_ax: plt object, k-s curcve
    """
    # 检查true_Y是否是二元变量
    y_type = type_of_target(true_Y)
    if y_type not in ['binary']:
        raise ValueError('y必须是二元变量')
    # 合并true_Y与y_hat,并按pred_Y对数据进行降序排列
    datasets = pd.concat([pd.Series(true_Y, name='true_Y'), pd.Series(pred_Y, name='pred_Y')], axis=1)
    # datasets = pd.concat([true_Y, pd.Series(pred_Y, name='pred_Y', index=true_Y.index)], axis=1)
    datasets.columns = ["true_Y", "pred_Y"]
    datasets = datasets.sort_values(by="pred_Y", axis=0, ascending=True)
    # 计算正负案例数和行数,以及等分子集的行数n
    P = sum(true_Y)
    Nrows = datasets.shape[0]
    N = Nrows - P
    n = float(Nrows)/k
    # 重建索引，并将数据划分为子集，并计算每个子集的正例数和负例数
    datasets.index = np.arange(Nrows)
    ks_df = pd.DataFrame()
    rlt = {
            "tile":str(0),
            "Ptot":0,
            "Ntot":0}
    ks_df = ks_df.append(pd.Series(rlt), ignore_index=True)
    for i in range(k):
        lo = i*n
        up = (i+1)*n
        tile = datasets.ix[lo:(up-1), :]
        Ptot = sum(tile['true_Y'])
        Ntot = n-Ptot
        rlt = {
                "tile":str(i+1),
                "Ptot":Ptot,
                "Ntot":Ntot}
        ks_df = ks_df.append(pd.Series(rlt), ignore_index=True)
    # 计算各子集中的正负例比例,以及累积比例
    ks_df['PerP'] = ks_df['Ptot']/P
    ks_df['PerN'] = ks_df['Ntot']/N
    ks_df['PerP_cum'] = ks_df['PerP'].cumsum()
    ks_df['PerN_cum'] = ks_df['PerN'].cumsum()
    # 计算ks曲线以及ks值
    ks_df['ks'] = ks_df['PerN_cum'] - ks_df['PerP_cum']
    ks_value = ks_df['ks'].max()
    s = "KS value is %.4f" % ks_value
    # 整理得出ks统计表
    ks_results = ks_df.ix[1:,:]
    ks_results = ks_results[['tile', 'Ntot', 'Ptot', 'PerN', 'PerP', 'PerN_cum', 'PerP_cum', 'ks']]
    ks_results.columns = ['子集','负例数','正例数','负例比例','正例比例','累积负例比例','累积正例比例', 'ks']
    # 获取ks值所在的数据点
    ks_point = ks_results.ix[:,['子集','ks']]
    ks_point = ks_point.ix[ks_point['ks']==ks_point['ks'].max(),:]
    # 绘制KS曲线
    ks_ax = _ks_plot(ks_df=ks_df, ks_label='ks', good_label='PerN_cum', bad_label='PerP_cum', 
                    k=k, point=ks_point, s=s)
    return ks_results, ks_ax


def _ks_plot(ks_df, ks_label, good_label, bad_label, k, point, s):
    """
    middle function for ks_stats, plot k-s curve
    """
    plt.plot(ks_df['tile'], ks_df[ks_label], "r-.", label="ks_curve", lw=1.2)
    plt.plot(ks_df['tile'], ks_df[good_label], "g-.", label="good", lw=1.2)
    plt.plot(ks_df['tile'], ks_df[bad_label], "m-.", label="bad", lw=1.2)
    #plt.plot(point[0], point[1], 'o', markerfacecolor="red",
             #markeredgecolor='k', markersize=6)
    plt.legend(loc=0)
    plt.plot([0, k], [0, 1], linestyle='--', lw=0.8, color='k', label='Luck')
    plt.xlabel("decilis")#等份子集
    plt.title(s)#KS曲线图
    plt.show()    



"""
提升图和洛伦茨曲线
"""
def lift_lorenz(true_Y, pred_Y, k=10):
    """
    plot lift_lorenz curve 
    ----------------------------------
    Params
    pred_Y: prediction of model
    true_Y: real data(testing sets)
    k: Section number 
    ----------------------------------
    lift_ax: lift chart
    lorenz_ax: lorenz curve
    """
    # 检查y是否是二元变量
    y_type = type_of_target(true_Y)
    if y_type not in ['binary']:
        raise ValueError('y必须是二元变量')
    # 合并y与y_hat,并按prob_y对数据进行降序排列
    datasets = pd.concat([pd.Series(true_Y, name='true_Y'), pd.Series(pred_Y, name='pred_Y')], axis=1)
    datasets.columns = ["true_Y", "pred_Y"]
    datasets = datasets.sort_values(by="pred_Y", axis=0, ascending=False)
    # 计算正案例数和行数,以及等分子集的行数n
    P = sum(true_Y)
    Nrows = datasets.shape[0]
    n = float(Nrows)/k
    # 重建索引，并将数据划分为子集，并计算每个子集的正例数和负例数
    datasets.index = np.arange(Nrows)
    lift_df = pd.DataFrame()
    rlt = {
            "tile":str(0),
            "Ptot":0,
          }
    lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
    for i in range(k):
        lo = i*n
        up = (i+1)*n
        tile = datasets.ix[lo:(up-1), :]
        Ptot = sum(tile['true_Y'])
        rlt = {
                "tile":str(i+1),
                "Ptot":Ptot,
                }
        lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
    # 计算正例比例&累积正例比例
    lift_df['PerP'] = lift_df['Ptot']/P
    lift_df['PerP_cum'] = lift_df['PerP'].cumsum()
    # 计算随机正例数、正例率以及累积随机正例率
    lift_df['randP'] = float(P)/k
    lift_df['PerRandP'] = lift_df['randP']/P
    lift_df.ix[0,:]=0
    lift_df['PerRandP_cum'] = lift_df['PerRandP'].cumsum()
    lift_ax = lift_Chart(lift_df, k)
    lorenz_ax = lorenz_cruve(lift_df)
    return lift_ax, lorenz_ax


def lift_Chart(df, k):
    """
    middle function for lift_lorenz, plot lift Chart
    """
    #绘图变量
    PerP = df['PerP'][1:]
    PerRandP = df['PerRandP'][1:]
    #绘图参数
    fig, ax = plt.subplots()
    index = np.arange(k+1)[1:]
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, PerP, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Per_p')#正例比例
    rects2 = plt.bar(index + bar_width, PerRandP, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='random_P')#随机比例
    plt.xlabel('Group')
    plt.ylabel('Percent')
    plt.title('lift_Chart')
    plt.xticks(index + bar_width / 2, tuple(index))
    plt.legend()
    plt.tight_layout()
    plt.show()

def lorenz_cruve(df):
    """
    middle function for lift_lorenz, plot lorenz cruve
    """
    #准备绘图所需变量
    PerP_cum = df['PerP_cum']
    PerRandP_cum = df['PerRandP_cum']
    decilies = df['tile']
    #绘制洛伦茨曲线
    plt.plot(decilies, PerP_cum, 'm-^', label='lorenz_cruve')#lorenz曲线
    plt.plot(decilies, PerRandP_cum, 'k-.', label='random')#随机
    plt.legend()
    plt.xlabel("decilis")#等份子集
    plt.title("lorenz_cruve", fontsize=10)#洛伦茨曲线
    plt.show()  