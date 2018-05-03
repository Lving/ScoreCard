"""
评分卡特征分箱工具
https://blog.csdn.net/mr_tyting/article/details/75212250
"""
import pandas as pd
import numpy as np


def calcCut(df, col, target, cutPoints, output_type='dataframe'):
    """
    :param df: 
    :param col: splitted column
    :param target: 
    :param cutPoints: cutPoints
    :param output_type: dtaframe or dict
    :return res, discrete_df: 返回woe的信息, 离散化之后的列, 便于做woe trans
    woe的信息可以选择以dataframe或者dict的形式输出
    """

    # 补齐首尾便于cut
    if cutPoints[0] == df[col].min():
        cutPoints = [*cutPoints, np.inf]
    else:
        cutPoints = [0, *cutPoints, np.inf]
    labels = genLabels(cutPoints)

    discrete_df = pd.DataFrame()   # 离散化的新df


    binName = "%s_Bins" % col  # 为离散化的列附上新名称

    discrete_df[binName] = pd.cut(df[col], cutPoints, labels=labels, right=False)
    discrete_df[target] = df[target].copy()

    if output_type == 'dataframe':  # True：返回WOE的dataframe, 便于在excel中分析
        res = CalcWOE(discrete_df, binName, target=target, output_type=output_type)
        return res, discrete_df

    elif output_type == 'dict':  # False: 返回dict, 便于后续的实施
        res = CalcWOE(discrete_df, binName, target=target, output_type=output_type)
        res['CUTPOINTS'] = cutPoints
        return res, discrete_df 
    else:
        return


def genLabels(cutOffPoints):
    """
    将最优分箱点，生成list作为label
    """
    labels = []
    for i in range(len(cutOffPoints) - 1):
        interval = (cutOffPoints[i], cutOffPoints[i+1])
        labels.append(str(interval))
    return labels

def Chi2(df, total_col, bad_col, overallRate):
    """
    :param df: the dataset containing the total count and bad count
    :param total_col: total count of each value in the variable
    :param bad_col: bad count of each value in the variable
    :param overallRate: the overall bad rate of the training set
    :return: the chi-square value
    """
    df2 = df.copy()
    df2['expected'] = df[total_col].apply(lambda x: x*overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0]-i[1])**2/i[0] for i in combined]
    chi2 = sum(chi)
    return chi2


### ChiMerge_MaxInterval: split the continuous variable using Chi-square value by specifying the max number of intervals
def ChiMerge_MaxInterval_Original(df, col, target, max_interval = 5):
    """
    :param df: the dataframe containing splitted column, and target column with 1-0
    :param col: splitted column
    :param target: target column with 1-0
    :param max_interval: the maximum number of intervals. If the raw column has attributes less than this parameter, the function will not work
    :return: the combined bins
    """
    # 太多的小数点影响速度， 保留一位
    #colLevels = set(df[col])
    colLevels = set(df[col].round(1))
    colLevels = sorted(list(colLevels)) # 先对这列数据进行排序，然后在计算分箱
    N_distinct = len(colLevels)
    if N_distinct <= max_interval:  #If the raw column has attributes less than this parameter, the function will not work
        print("The number of original levels for {} is less than or equal to max intervals".format(col))
        return colLevels[:-1]
    else:
        #Step 1: group the dataset by col and work out the total count & bad count in each level of the raw column
        total = df.groupby([col])[target].count()
        total = pd.DataFrame({'total':total})
        bad = df.groupby([col])[target].sum()
        bad = pd.DataFrame({'bad':bad})
        regroup =  total.merge(bad,left_index=True,right_index=True, how='left')##将左侧，右侧的索引用作其连接键。
        regroup.reset_index(level=0, inplace=True)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        #the overall bad rate will be used in calculating expected bad count
        overallRate = B * 1.0 / N ##　统计坏样本率
        # initially, each single attribute forms a single interval
        groupIntervals = [[i] for i in colLevels]## 类似于[[1],[2],[3,4]]其中每个[.]为一箱
        groupNum = len(groupIntervals)
        while(len(groupIntervals)>max_interval):   #the termination condition: the number of intervals is equal to the pre-specified threshold
            # in each step of iteration, we calcualte the chi-square value of each atttribute
            chisqList = []
            for interval in groupIntervals:
                df2 = regroup.loc[regroup[col].isin(interval)]
                chisq = Chi2(df2, 'total','bad',overallRate)
                chisqList.append(chisq)
            #find the interval corresponding to minimum chi-square, and combine with the neighbore with smaller chi-square
            min_position = chisqList.index(min(chisqList))
            if min_position == 0:## 如果最小位置为0,则要与其结合的位置为１
                combinedPosition = 1
            elif min_position == groupNum - 1:
                combinedPosition = min_position -1
            else:## 如果在中间，则选择左右两边卡方值较小的与其结合
                if chisqList[min_position - 1]<=chisqList[min_position + 1]:
                    combinedPosition = min_position - 1
                else:
                    combinedPosition = min_position + 1
            groupIntervals[min_position] = groupIntervals[min_position]+groupIntervals[combinedPosition]
            # after combining two intervals, we need to remove one of them
            groupIntervals.remove(groupIntervals[combinedPosition])
            groupNum = len(groupIntervals)
        groupIntervals = [sorted(i) for i in groupIntervals]  ## 对每组的数据安从小到大排序
        cutOffPoints = [i[-1] for i in groupIntervals[:-1]]  ## 提取出每组的最大值，也就是分割点
        return cutOffPoints


def CalcWOE(df, col, target, output_type='dataframe'):
    '''
    :param df: dataframe containing feature and target
    :param col: 注意col这列已经经过分箱了，现在计算每箱的WOE和总的IV。
    :param target: good/bad indicator
    :return: 返回每箱的WOE(字典类型）和总的IV之和。
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pct'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pct'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['PctRec'] = regroup['total'].map(lambda x: x * 1.0 / N)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pct*1.0/(x.bad_pct+0.001)),axis = 1)  # 防止zerodivision错误
    regroup['ori_iv'] = regroup.apply(lambda x: (x.good_pct-x.bad_pct)*np.log(x.good_pct*1.0/(x.bad_pct+0.001)),axis = 1)
    regroup['sum_iv'] = sum(regroup['ori_iv'])
    regroup.rename(index=str, columns={col: 'interval'}, inplace=True)
    if output_type == 'dict':

        WOE_dict = regroup[['interval','WOE']].set_index('interval').to_dict(orient='index')
        # IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
        IV = sum(regroup['ori_iv'])
        return {"WOE": WOE_dict, 'IV':IV}
    elif output_type == 'dataframe':
        return regroup
        # return {"WOE": WOE_dict, 'IV':IV}

    else:
        return

def AssignBin(x, cutOffPoints):
    '''
    :param x: the value of variable
    :param cutOffPoints: 每组的最大值，也就是分割点
    :return: bin number, indexing from 0
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    '''
    numBin = len(cutOffPoints) + 1
    if x<=cutOffPoints[0]:
        return 'Bin 0'
    elif x > cutOffPoints[-1]:
        return 'Bin {}'.format(numBin-1)
    else:
        for i in range(0,numBin-1):
            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                return 'Bin {}'.format(i+1)


# determine whether the bad rate is monotone along the sortByVar
# 检查分箱的单调性
def BadRateMonotone(df, sortByVar, target):
    # df[sortByVar]这列数据已经经过分箱
    df2 = df.sort([sortByVar])
    total = df2.groupby([sortByVar])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df2.groupby([sortByVar])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateMonotone = [badRate[i]<badRate[i+1] for i in range(len(badRate)-1)]
    Monotone = len(set(badRateMonotone))
    if Monotone == 1:
        return True
    else:
        return False

# 检查最大分箱数据的占比, 占比超过90%弃用变量
def MaximumBinPcnt(df,col):
    N = df.shape[0]
    total = df.groupby([col])[col].count()
    pcnt = total*1.0/N
    return max(pcnt)

# 当某个或者几个类别的bad rate为0时,需要和最小的非0bad rate的箱进行合并
# If we find any categories with 0 bad, then we combine these categories with that having smallest non-zero bad rate
def MergeBad0(df,col,target):
    '''
     :param df: dataframe containing feature and target
     :param col: the feature that needs to be calculated the WOE and iv, usually categorical type
     :param target: good/bad indicator
     :return: WOE and IV in a dictionary
     '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad*1.0/x.total,axis = 1)
    regroup = regroup.sort_values(by = 'bad_rate')
    col_regroup = [[i] for i in regroup[col]]
    for i in range(regroup.shape[0]):
        col_regroup[1] = col_regroup[0] + col_regroup[1]
        col_regroup.pop(0)
        if regroup['bad_rate'][i+1] > 0:
            break
    newGroup = {}
    for i in range(len(col_regroup)):
        for g2 in col_regroup[i]:
            newGroup[g2] = 'Bin '+str(i)
    return newGroup
